"""
Reusable helpers for detecting sources in images and comparing them to catalogs.

This module was extracted from ``util.log_bright_source_flux_comparison`` so the
same logic can be reused for:
- catalog-based astrometric QC (Phase 2+ in the tuning plan)
- diagnostics/logging of bright source alignment

Conventions
-----------
- Pixel coordinates are always ``(x, y)`` with ``x`` increasing along numpy axis 1
  (columns) and ``y`` increasing along numpy axis 0 (rows).
- Image arrays are 2D numpy arrays in standard (y, x) indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

logger = logging.getLogger(__name__)


def _prefer_bdsf_subprocess() -> bool:
    """
    PyBDSF uses multiprocessing with fork; that is unsafe once JAX has started threads.

    When unset, run PyBDSF out-of-process if ``jax`` is already imported.

    Override with environment variable:

    - ``IMAGE_PLANE_CORRECTION_BDSF_SUBPROCESS=1`` — always use subprocess.
    - ``IMAGE_PLANE_CORRECTION_BDSF_SUBPROCESS=0`` — never use subprocess (may warn with JAX).
    """
    v = os.environ.get("IMAGE_PLANE_CORRECTION_BDSF_SUBPROCESS", "").strip()
    if v == "0":
        return False
    if v == "1":
        return True
    return sys.modules.get("jax") is not None


def _run_bdsf_via_subprocess(
    fits_path: str,
    *,
    process_kw: dict[str, Any],
    min_separation_px: float,
    n_sources_max: int | None,
) -> np.ndarray:
    """Run ``python -m image_plane_correction.pybdsf_worker`` with ``src`` on PYTHONPATH."""
    fd_out, out_npy = tempfile.mkstemp(suffix=".npy")
    os.close(fd_out)
    out_npy = str(out_npy)
    payload = dict(process_kw)
    payload["__min_separation_px"] = float(min_separation_px)
    payload["__n_sources_max"] = n_sources_max
    serializable: dict[str, Any] = {}
    for k, v in payload.items():
        if k == "beam":
            serializable[k] = [float(v[0]), float(v[1]), float(v[2])]
        else:
            serializable[k] = v

    src_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root if not prev else f"{src_root}{os.pathsep}{prev}"

    cmd = [
        sys.executable,
        "-m",
        "image_plane_correction.pybdsf_worker",
        fits_path,
        out_npy,
        json.dumps(serializable),
    ]
    try:
        # Do not use PIPE here: PyBDSF can be verbose; filling stderr/stdout pipes
        # blocks the child and appears as a hang (classic subprocess deadlock).
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning(
                "PyBDSF subprocess failed with exit code %s "
                "(set IMAGE_PLANE_CORRECTION_BDSF_SUBPROCESS=0 for in-process errors).",
                proc.returncode,
            )
            return np.zeros((0, 2), dtype=np.float64)
        return np.load(out_npy).astype(np.float64, copy=False)
    finally:
        try:
            os.unlink(out_npy)
        except OSError:
            pass


@dataclass(frozen=True)
class CatalogData:
    """Prepared catalog data in both sky and pixel coordinates."""

    sky: Any  # astropy.coordinates.SkyCoord
    flux: np.ndarray  # (N,) float, may contain NaNs
    xy: np.ndarray  # (N,2) float pixel coordinates (x,y)
    meta: Mapping[str, np.ndarray]  # optional extra columns aligned to N


def catalog_to_pixel(
    skycoord: Any,
    imwcs: Any,
    *,
    image_shape: tuple[int, int] | None = None,
    wcs_shape_scale: bool = True,
) -> np.ndarray:
    """
    Convert a SkyCoord catalog to pixel coordinates (x,y) for an image.

    If ``wcs_shape_scale`` is True and ``imwcs.pixel_shape`` is set, scale the
    returned pixel coordinates to match ``image_shape`` when the WCS was defined
    on a different pixel grid.
    """
    from astropy.wcs.utils import skycoord_to_pixel

    xy = np.stack(skycoord_to_pixel(skycoord, imwcs), axis=1).astype(np.float64)
    if not wcs_shape_scale:
        return xy

    if image_shape is None:
        return xy

    if getattr(imwcs, "pixel_shape", None) is None:
        return xy

    h, w = int(image_shape[0]), int(image_shape[1])
    wcs_h = int(imwcs.pixel_shape[1])
    wcs_w = int(imwcs.pixel_shape[0])
    if wcs_h <= 0 or wcs_w <= 0:
        return xy

    # Preserve the existing behavior in util.py: treat WCS as square-ish scale.
    # If you ever need anisotropic scaling, promote this to per-axis scaling.
    scale = float(h) / float(wcs_h)
    return xy * scale


def load_reference_catalog(
    catalog: str | Any | np.ndarray,
    *,
    imwcs: Any | None = None,
    catalog_path: str | None = None,
    min_flux: float = 0.0,
    pointlike_axis_ratio_max: float = 1.8,
    image_shape: tuple[int, int] | None = None,
) -> CatalogData:
    """
    Load/prepare a reference catalog.

    Parameters
    ----------
    catalog
        - ``str``: catalog key handled by ``image_plane_correction.catalogs.reference_sources``.
        - ``SkyCoord``: sky positions directly.
        - ``(N,2)`` array: interpreted as pixel ``(x,y)`` positions; sky positions will be None-like.
    imwcs
        Required for ``catalog`` keys and for converting SkyCoord to pixels.
    image_shape
        Used to scale pixel coordinates if WCS was defined on a different grid.
    """
    meta: MutableMapping[str, np.ndarray] = {}

    if isinstance(catalog, str):
        if imwcs is None:
            raise ValueError("`imwcs` is required when `catalog` is a catalog key string.")
        from .catalogs import reference_sources

        sky, flux = reference_sources(catalog, min_flux=min_flux, path=catalog_path)
        flux_np = np.asarray(flux, dtype=float)
        xy = catalog_to_pixel(sky, imwcs, image_shape=image_shape, wcs_shape_scale=True)

        # Optional morphology filtering: if catalog files have major/minor axes,
        # attempt to filter to point-like sources, mirroring legacy behavior.
        # This keeps the loader generic, and failures fall back to "no filter".
        if catalog in ("NVSS", "VLSSR"):
            try:
                import pandas as pd
                from .catalogs import NVSS_CATALOG, VLSSR_CATALOG

                resolved_path = catalog_path
                if resolved_path is None:
                    resolved_path = NVSS_CATALOG if catalog == "NVSS" else VLSSR_CATALOG
                raw = (
                    pd.read_csv(resolved_path, sep=r"\s+")
                    if catalog == "NVSS"
                    else pd.read_csv(resolved_path, sep=" ")
                )
                raw = raw.sort_values(by=["f"] if catalog == "NVSS" else "PEAK INT")
                cols_upper = {c.upper().replace(" ", "_"): c for c in raw.columns}
                maj_col = None
                min_col = None
                for candidate in ("MAJ", "MAJOR", "BMAJ", "MAJ_AX", "MAJAX"):
                    if candidate in cols_upper:
                        maj_col = cols_upper[candidate]
                        break
                for candidate in ("MIN", "MINOR", "BMIN", "MIN_AX", "MINAX"):
                    if candidate in cols_upper:
                        min_col = cols_upper[candidate]
                        break
                if maj_col is not None and min_col is not None and raw.shape[0] >= xy.shape[0]:
                    maj = raw[maj_col].to_numpy(dtype=float)[-xy.shape[0] :]
                    min_ax = raw[min_col].to_numpy(dtype=float)[-xy.shape[0] :]
                    finite_axes = np.isfinite(maj) & np.isfinite(min_ax) & (maj > 0) & (min_ax > 0)
                    axis_ratio = np.full_like(maj, np.inf, dtype=float)
                    axis_ratio[finite_axes] = maj[finite_axes] / min_ax[finite_axes]
                    keep = finite_axes & (axis_ratio <= float(pointlike_axis_ratio_max))
                    sky = sky[keep]
                    flux_np = flux_np[keep]
                    xy = xy[keep]
            except Exception:
                pass

        return CatalogData(sky=sky, flux=flux_np, xy=xy, meta=meta)

    # SkyCoord-like input
    if hasattr(catalog, "ra") and hasattr(catalog, "dec"):
        if imwcs is None:
            raise ValueError("`imwcs` is required to convert SkyCoord catalog to pixel coordinates.")
        sky = catalog
        flux_np = np.full(len(sky), np.nan, dtype=float)
        xy = catalog_to_pixel(sky, imwcs, image_shape=image_shape, wcs_shape_scale=True)
        return CatalogData(sky=sky, flux=flux_np, xy=xy, meta=meta)

    # Pixel catalog input
    xy = np.asarray(catalog, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("catalog must be a catalog key string, SkyCoord, or an Nx2 pixel position array.")
    flux_np = np.full(xy.shape[0], np.nan, dtype=float)
    return CatalogData(sky=None, flux=flux_np, xy=xy.astype(np.float64), meta=meta)


def filter_catalog_to_image(
    cat_xy: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Boolean mask selecting catalog entries whose pixel positions fall inside the image."""
    h, w = int(image_shape[0]), int(image_shape[1])
    cx = cat_xy[:, 0]
    cy = cat_xy[:, 1]
    return (
        np.isfinite(cx)
        & np.isfinite(cy)
        & (cx >= 0)
        & (cx < w)
        & (cy >= 0)
        & (cy < h)
    )


def select_top_catalog_sources(
    cat_xy: np.ndarray,
    flux: np.ndarray,
    n_sources: int,
) -> np.ndarray:
    """Deterministically select up to ``n_sources`` catalog entries (preferring higher flux when available)."""
    if n_sources <= 0:
        raise ValueError("n_sources must be positive.")
    if cat_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if np.any(np.isfinite(flux)):
        order = np.lexsort(
            (
                cat_xy[:, 1],
                cat_xy[:, 0],
                -np.nan_to_num(flux, nan=-np.inf),
            )
        )
    else:
        order = np.lexsort((cat_xy[:, 1], cat_xy[:, 0]))
    take = min(int(n_sources), int(cat_xy.shape[0]))
    return cat_xy[order][:take].astype(np.float64)


def _sanitize_array_for_bdsf(image: np.ndarray) -> np.ndarray:
    """Finite pixels unchanged; NaN/inf set to 0 so PyBDSF sees blank sky rather than NaN islands."""
    z = np.asarray(image, dtype=np.float32)
    return np.where(np.isfinite(z), z, 0.0)


def _fits_header_for_bdsf(
    imwcs: Any,
    shape_yx: tuple[int, int],
    *,
    beam_major_deg: float,
    beam_minor_deg: float,
    beam_pa_deg: float,
    restfreq_hz: float | None,
) -> Any:
    """Build a minimal FITS header PyBDSF can read (WCS, beam, frequency)."""
    hdr = imwcs.to_header(relax=True)
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = int(shape_yx[1])
    hdr["NAXIS2"] = int(shape_yx[0])
    hdr["BUNIT"] = hdr.get("BUNIT", "Jy/beam")
    hdr["BMAJ"] = float(beam_major_deg)
    hdr["BMIN"] = float(beam_minor_deg)
    hdr["BPA"] = float(beam_pa_deg)
    rf = restfreq_hz
    if rf is None:
        rf = float(hdr.get("RESTFREQ", hdr.get("RESTFRQ", 74e6)))
    hdr["RESTFREQ"] = float(rf)
    hdr["RESTFRQ"] = float(rf)
    return hdr


def _gaussian_rows_from_bdsf_image(img: Any) -> tuple[np.ndarray, np.ndarray]:
    """Collect fitted Gaussian centres (x,y) and sorting flux from a PyBDSF Image object."""
    xy_list: list[tuple[float, float]] = []
    flux_list: list[float] = []
    for src in getattr(img, "sources", []) or []:
        for g in getattr(src, "gaussians", []) or []:
            cp = getattr(g, "centre_pix", None)
            if cp is None or len(cp) < 2:
                continue
            x, y = float(cp[0]), float(cp[1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            pf = getattr(g, "peak_flux", None)
            tf = getattr(g, "total_flux", None)
            score = float(pf) if pf is not None and np.isfinite(float(pf)) else float(tf or 0.0)
            xy_list.append((x, y))
            flux_list.append(score)
    return np.asarray(xy_list, dtype=np.float64), np.asarray(flux_list, dtype=np.float64)


def _nms_xy_flux_sorted(
    xy: np.ndarray,
    flux: np.ndarray,
    *,
    min_separation_px: float,
    max_keep: int | None,
) -> np.ndarray:
    """Greedy NMS on flux-sorted components (brightest first)."""
    if xy.size == 0:
        return xy.reshape((0, 2)).astype(np.float64)
    order = np.argsort(-flux)
    kept: list[tuple[float, float]] = []
    min_sep = float(max(0.0, min_separation_px))
    for i in order:
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if min_sep > 0 and kept:
            kxy = np.asarray(kept, dtype=float)
            if np.any(np.hypot(kxy[:, 0] - x, kxy[:, 1] - y) < min_sep):
                continue
        kept.append((x, y))
        if max_keep is not None and len(kept) >= int(max_keep):
            break
    return np.asarray(kept, dtype=np.float64)


def detect_sources_bdsf_xy(
    image: np.ndarray,
    imwcs: Any,
    *,
    beam_major_deg: float,
    beam_minor_deg: float,
    beam_pa_deg: float = 0.0,
    n_sources_max: int | None,
    min_separation_px: float = 15.0,
    thresh: str | None = "hard",
    thresh_isl: float = 20.0,
    thresh_pix: float = 5.0,
    minpix_isl: int | None = None,
    quiet: bool = True,
    restfreq_hz: float | None = None,
    ncores: int = 4,
) -> np.ndarray:
    """
    Run PyBDSF on a 2D array and return up to ``n_sources_max`` Gaussian centres (x, y), 0-based pixels.

    Non-finite image values are replaced with 0 before processing.

    PyBDSF ``ncores`` defaults to 4 (passed through to ``bdsf.process_image``).

    If ``jax`` is already imported in this interpreter, PyBDSF is run in a subprocess
    by default (PyBDSF uses fork-based multiprocessing, which conflicts with JAX threads).
    Set ``IMAGE_PLANE_CORRECTION_BDSF_SUBPROCESS=0`` to force in-process execution.
    Subprocess stdout/stderr are discarded to avoid pipe deadlock on verbose runs.
    """
    z = _sanitize_array_for_bdsf(image)
    if z.ndim != 2:
        raise ValueError("image must be 2D.")

    ncores_used = max(1, int(ncores))

    hdr = _fits_header_for_bdsf(
        imwcs,
        z.shape,
        beam_major_deg=float(beam_major_deg),
        beam_minor_deg=float(beam_minor_deg),
        beam_pa_deg=float(beam_pa_deg),
        restfreq_hz=restfreq_hz,
    )

    fd, path = tempfile.mkstemp(suffix=".fits")
    os.close(fd)
    path = str(path)
    try:
        from astropy.io import fits

        fits.writeto(path, z, hdr, overwrite=True)
        beam = (float(beam_major_deg), float(beam_minor_deg), float(beam_pa_deg))
        process_kw: dict[str, Any] = dict(
            beam=beam,
            thresh_isl=float(thresh_isl),
            thresh_pix=float(thresh_pix),
            atrous_do=False,
            psf_vary_do=False,
            quiet=bool(quiet),
            ncores=ncores_used,
        )
        if thresh is not None:
            process_kw["thresh"] = str(thresh)
        if minpix_isl is not None:
            process_kw["minpix_isl"] = int(minpix_isl)

        if _prefer_bdsf_subprocess():
            return _run_bdsf_via_subprocess(
                path,
                process_kw=process_kw,
                min_separation_px=float(min_separation_px),
                n_sources_max=n_sources_max,
            )

        import bdsf

        try:
            img = bdsf.process_image(path, **process_kw)
        except Exception as exc:
            logger.warning("PyBDSF process_image failed: %s", exc)
            return np.zeros((0, 2), dtype=np.float64)

        xy_all, flux_all = _gaussian_rows_from_bdsf_image(img)
        return _nms_xy_flux_sorted(
            xy_all,
            flux_all,
            min_separation_px=min_separation_px,
            max_keep=n_sources_max,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def identify_sources_bdsf(
    input_image: str | os.PathLike[str],
    imwcs: Any,
    work_dir: str | None = None,
    *,
    beam_fwhm_deg: tuple[float, float] | None = None,
    beam_pa_deg: float = 0.0,
    n_sources_max: int | None = None,
    min_separation_px: float = 0.0,
    thresh: str | None = "hard",
    thresh_isl: float = 10.0,
    thresh_pix: float = 5.0,
    minpix_isl: int | None = None,
    quiet: bool = True,
    restfreq_hz: float | None = None,
    ncores: int = 4,
) -> Any:
    """
    Run PyBDSF on a FITS image path; return ``SkyCoord`` (ICRS) for fitted Gaussian centres.

    ``work_dir`` is accepted for API compatibility with legacy scripts and ignored.

    If ``beam_fwhm_deg`` is omitted, ``BMAJ`` / ``BMIN`` are read from the FITS header (degrees).
    """
    _ = work_dir  # unused
    from astropy.io import fits as fits_io

    path = Path(input_image)
    with fits_io.open(path, memmap=False) as hdul:
        data = np.squeeze(np.asarray(hdul[0].data, dtype=np.float32))
        header = hdul[0].header

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D image array after squeezing; got shape {data.shape}.")

    if beam_fwhm_deg is None:
        if "BMAJ" not in header or "BMIN" not in header:
            raise ValueError("beam_fwhm_deg required when FITS header lacks BMAJ/BMIN.")
        bmaj = float(header["BMAJ"])
        bmin = float(header["BMIN"])
    else:
        bmaj, bmin = float(beam_fwhm_deg[0]), float(beam_fwhm_deg[1])

    rf = restfreq_hz
    if rf is None and "RESTFREQ" in header:
        rf = float(header["RESTFREQ"])
    elif rf is None and "RESTFRQ" in header:
        rf = float(header["RESTFRQ"])

    xy = detect_sources_bdsf_xy(
        data,
        imwcs,
        beam_major_deg=bmaj,
        beam_minor_deg=bmin,
        beam_pa_deg=float(header.get("BPA", beam_pa_deg)),
        n_sources_max=n_sources_max,
        min_separation_px=min_separation_px,
        thresh=thresh,
        thresh_isl=thresh_isl,
        thresh_pix=thresh_pix,
        minpix_isl=minpix_isl,
        quiet=quiet,
        restfreq_hz=rf,
        ncores=ncores,
    )
    return pixels_to_skycoord(xy, imwcs, origin=0)


def match_xy_nearest(
    measured_xy: np.ndarray,
    catalog_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nearest-neighbor match in pixel coordinates.

    Returns ``(idx_cat, dist_px)`` for each measured point, where ``idx_cat`` is the
    index of the nearest catalog point.
    """
    m = np.asarray(measured_xy, dtype=float)
    c = np.asarray(catalog_xy, dtype=float)
    if m.size == 0 or c.size == 0:
        return np.full(m.shape[0], -1, dtype=int), np.full(m.shape[0], np.inf, dtype=float)
    d2 = np.sum((m[:, None, :] - c[None, :, :]) ** 2, axis=2)
    idx = np.argmin(d2, axis=1).astype(int)
    dist = np.sqrt(d2[np.arange(m.shape[0]), idx])
    return idx, dist.astype(float)


def summarize_separations(dist: np.ndarray) -> Mapping[str, float]:
    """Robust summary stats for separations; returns finite-only stats."""
    d = np.asarray(dist, dtype=float)
    finite = np.isfinite(d)
    if not np.any(finite):
        return {"median": float("nan"), "rms": float("nan"), "p90": float("nan")}
    d = d[finite]
    return {
        "median": float(np.median(d)),
        "rms": float(np.sqrt(np.mean(d**2))),
        "p90": float(np.percentile(d, 90.0)),
    }


def pixels_to_skycoord(
    xy: np.ndarray,
    imwcs: Any,
    *,
    origin: int = 0,
) -> Any:
    """
    Convert pixel ``(x,y)`` positions to ``astropy.coordinates.SkyCoord`` via WCS.

    Parameters
    ----------
    xy
        ``(N,2)`` array of pixel positions (x,y).
    imwcs
        Astropy WCS.
    origin
        Pixel origin for WCS conversion (0 for numpy convention).
    """
    from astropy.wcs.utils import pixel_to_skycoord

    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("xy must be an (N,2) array of (x,y) pixel positions.")
    return pixel_to_skycoord(pts[:, 0], pts[:, 1], imwcs, origin=origin)


def match_sky_nearest(
    measured_sky: Any,
    catalog_sky: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nearest-neighbor match in sky coordinates using Astropy.

    Returns ``(idx_cat, sep_deg)`` for each measured source.
    """
    if measured_sky is None or catalog_sky is None:
        raise ValueError("measured_sky and catalog_sky must be SkyCoord objects (not None).")
    idx, sep2d, _ = measured_sky.match_to_catalog_sky(catalog_sky)
    return np.asarray(idx, dtype=int), np.asarray(sep2d.deg, dtype=float)


def summarize_angular_separations_deg(sep_deg: np.ndarray) -> Mapping[str, float]:
    """
    Robust summary stats for angular separations in degrees.

    Returns both degrees and arcseconds variants for convenience.
    """
    stats = summarize_separations(np.asarray(sep_deg, dtype=float))
    scale = 3600.0
    return {
        "median_deg": float(stats["median"]),
        "rms_deg": float(stats["rms"]),
        "p90_deg": float(stats["p90"]),
        "median_arcsec": float(stats["median"] * scale),
        "rms_arcsec": float(stats["rms"] * scale),
        "p90_arcsec": float(stats["p90"] * scale),
    }


def catalog_astrometry_qa(
    image: np.ndarray,
    *,
    imwcs: Any,
    catalog: str | Any | np.ndarray,
    catalog_path: str | None = None,
    min_flux: float = 0.0,
    n_catalog_sources: int = 50,
    n_measured_sources: int | None = None,
    min_separation_px: int = 15,
    beam_fwhm_deg: tuple[float | None, float | None] = (None, None),
    beam_pa_deg: float = 0.0,
    pointlike_axis_ratio_max: float = 1.8,
    max_sep_arcsec: float = 600.0,
    min_matches: int = 5,
    bdsf_thresh: str | None = "hard",
    bdsf_thresh_isl: float = 10.0,
    bdsf_thresh_pix: float = 5.0,
    bdsf_minpix_isl: int | None = None,
    bdsf_quiet: bool = True,
    restfreq_hz: float | None = None,
    bdsf_ncores: int = 4,
) -> MutableMapping[str, Any]:
    """
    End-to-end catalog-based astrometric QA using PyBDSF for source measurement.

    Steps:
    1) Load reference catalog and filter to image footprint.
    2) Select top ``n_catalog_sources`` catalog entries (by flux if available).
    3) Run PyBDSF on the image (non-finite pixels treated as blank); take brightest
       Gaussian components with optional separation-based deduplication.
    4) Match measured sky positions to the selected catalog (nearest neighbour).
    5) Summarize separations and match yield within ``max_sep_arcsec``.

    ``beam_fwhm_deg`` must provide both major and minor FWHM in degrees (synthesized beam).

    Returns a dict designed to be JSON-serializable (SkyCoord objects excluded).
    """
    z = np.asarray(image)
    if z.ndim != 2:
        raise ValueError("image must be 2D.")
    if imwcs is None:
        raise ValueError("imwcs is required.")
    if n_catalog_sources <= 0:
        raise ValueError("n_catalog_sources must be positive.")
    if min_matches < 0:
        raise ValueError("min_matches must be non-negative.")

    cat = load_reference_catalog(
        catalog,
        imwcs=imwcs,
        catalog_path=catalog_path,
        min_flux=min_flux,
        pointlike_axis_ratio_max=pointlike_axis_ratio_max,
        image_shape=z.shape,
    )

    keep = filter_catalog_to_image(cat.xy, z.shape)
    cat_xy = cat.xy[keep]
    cat_flux = cat.flux[keep]

    if cat_xy.size == 0:
        return {
            "ok": False,
            "reason": "no_visible_catalog_sources",
            "n_catalog_visible": 0,
            "n_catalog_used": 0,
            "n_measured": 0,
            "n_matched": 0,
            "matched_fraction": 0.0,
            "max_sep_arcsec": float(max_sep_arcsec),
            **summarize_angular_separations_deg(np.array([], dtype=float)),
        }

    selected_cat_xy = select_top_catalog_sources(cat_xy, cat_flux, n_sources=int(n_catalog_sources))
    if selected_cat_xy.size == 0:
        return {
            "ok": False,
            "reason": "no_catalog_sources_selected",
            "n_catalog_visible": int(cat_xy.shape[0]),
            "n_catalog_used": 0,
            "n_measured": 0,
            "n_matched": 0,
            "matched_fraction": 0.0,
            "max_sep_arcsec": float(max_sep_arcsec),
            **summarize_angular_separations_deg(np.array([], dtype=float)),
        }

    if n_measured_sources is None:
        n_measured_sources = int(selected_cat_xy.shape[0])
    if n_measured_sources <= 0:
        raise ValueError("n_measured_sources must be positive.")

    bmaj_deg, bmin_deg = beam_fwhm_deg
    if bmaj_deg is None or bmin_deg is None:
        raise ValueError("catalog_astrometry_qa requires beam_fwhm_deg=(bmaj_deg, bmin_deg) in degrees.")

    measured_xy = detect_sources_bdsf_xy(
        z,
        imwcs,
        beam_major_deg=float(bmaj_deg),
        beam_minor_deg=float(bmin_deg),
        beam_pa_deg=float(beam_pa_deg),
        n_sources_max=int(n_measured_sources),
        min_separation_px=float(min_separation_px),
        thresh=bdsf_thresh,
        thresh_isl=float(bdsf_thresh_isl),
        thresh_pix=float(bdsf_thresh_pix),
        minpix_isl=bdsf_minpix_isl,
        quiet=bool(bdsf_quiet),
        restfreq_hz=restfreq_hz,
        ncores=int(bdsf_ncores),
    )
    if measured_xy.size == 0:
        return {
            "ok": False,
            "reason": "no_measured_sources",
            "n_catalog_visible": int(cat_xy.shape[0]),
            "n_catalog_used": int(selected_cat_xy.shape[0]),
            "n_measured": 0,
            "n_matched": 0,
            "matched_fraction": 0.0,
            "max_sep_arcsec": float(max_sep_arcsec),
            **summarize_angular_separations_deg(np.array([], dtype=float)),
        }

    measured_sky = pixels_to_skycoord(measured_xy, imwcs, origin=0)

    # Convert selected catalog xy back to sky via WCS to avoid depending on the
    # original catalog sky ordering after filtering/selection.
    catalog_sky = pixels_to_skycoord(selected_cat_xy, imwcs, origin=0)

    _idx, sep_deg = match_sky_nearest(measured_sky, catalog_sky)
    sep_arcsec = np.asarray(sep_deg, dtype=float) * 3600.0
    matched = np.isfinite(sep_arcsec) & (sep_arcsec <= float(max_sep_arcsec))
    n_matched = int(np.count_nonzero(matched))
    frac = float(n_matched / max(1, measured_xy.shape[0]))

    stats = summarize_angular_separations_deg(sep_deg[matched])
    ok = bool(n_matched >= int(min_matches))

    return {
        "ok": ok,
        "reason": "ok" if ok else "too_few_matches",
        "n_catalog_visible": int(cat_xy.shape[0]),
        "n_catalog_used": int(selected_cat_xy.shape[0]),
        "n_measured": int(measured_xy.shape[0]),
        "n_matched": n_matched,
        "matched_fraction": frac,
        "max_sep_arcsec": float(max_sep_arcsec),
        **stats,
    }

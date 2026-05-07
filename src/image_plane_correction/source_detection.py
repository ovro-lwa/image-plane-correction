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
from typing import Any, Iterable, Literal, Mapping, MutableMapping, Sequence

import numpy as np


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


def detect_peaks(
    image: np.ndarray,
    *,
    n_peaks: int,
    min_separation_px: int = 15,
    finite_only: bool = True,
) -> np.ndarray:
    """
    Simple deterministic peak picker with non-maximum suppression.

    Returns an ``(M,2)`` array of peak positions in pixel coords (x,y), sorted by peak value desc.
    """
    if n_peaks <= 0:
        raise ValueError("n_peaks must be positive.")
    z = np.asarray(image)
    if z.ndim != 2:
        raise ValueError("image must be 2D.")

    flat_idx = np.argsort(z.ravel())[::-1]
    peaks: list[tuple[float, float]] = []
    for idx in flat_idx:
        y, x = np.unravel_index(idx, z.shape)
        if finite_only and not np.isfinite(z[y, x]):
            continue
        if peaks:
            peak_xy = np.asarray(peaks, dtype=float)
            deltas = peak_xy - np.array([x, y], dtype=float)
            if np.any(np.linalg.norm(deltas, axis=1) < float(min_separation_px)):
                continue
        peaks.append((float(x), float(y)))
        if len(peaks) >= int(n_peaks):
            break
    return np.asarray(peaks, dtype=np.float64)


def refine_centroids(
    image: np.ndarray,
    peaks_xy: np.ndarray,
    *,
    method: Literal["none", "centroid", "gaussian_fixed_beam"] = "gaussian_fixed_beam",
    imwcs: Any | None = None,
    beam_fwhm_deg: tuple[float | None, float | None] = (None, None),
) -> np.ndarray:
    """
    Refine peak positions to sub-pixel centroids.

    - ``none``: returns input.
    - ``centroid``: intensity-weighted centroid in a local window.
    - ``gaussian_fixed_beam``: least-squares fit of a fixed-width 2D Gaussian.
    """
    z = np.asarray(image, dtype=float)
    xy = np.asarray(peaks_xy, dtype=float)
    if xy.size == 0:
        return xy.reshape((0, 2)).astype(np.float64)

    if method == "none":
        return xy.astype(np.float64)

    if method == "centroid":
        out = []
        for x0, y0 in xy:
            ix = int(np.rint(x0))
            iy = int(np.rint(y0))
            half = 5
            y0w = max(0, iy - half)
            y1w = min(z.shape[0], iy + half + 1)
            x0w = max(0, ix - half)
            x1w = min(z.shape[1], ix + half + 1)
            patch = z[y0w:y1w, x0w:x1w]
            if patch.size == 0 or not np.all(np.isfinite(patch)):
                out.append((float(x0), float(y0)))
                continue
            yy, xx = np.indices(patch.shape, dtype=float)
            xx += x0w
            yy += y0w
            w = np.clip(patch, 0.0, np.inf)
            s = float(np.sum(w))
            if s <= 0:
                out.append((float(x0), float(y0)))
                continue
            xc = float(np.sum(xx * w) / s)
            yc = float(np.sum(yy * w) / s)
            out.append((xc, yc))
        return np.asarray(out, dtype=np.float64)

    if method != "gaussian_fixed_beam":
        raise ValueError(f"Unknown method: {method!r}")

    bmaj_deg, bmin_deg = beam_fwhm_deg
    if imwcs is None or bmaj_deg is None or bmin_deg is None:
        raise ValueError("imwcs, bmaj_deg, and bmin_deg are required for gaussian_fixed_beam refinement.")

    from astropy.wcs.utils import proj_plane_pixel_scales
    from scipy.optimize import least_squares

    pix_scales = np.abs(proj_plane_pixel_scales(imwcs))
    sigma_y = (float(bmaj_deg) / float(pix_scales[1])) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x = (float(bmin_deg) / float(pix_scales[0])) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = int(max(4, np.ceil(3.0 * max(sigma_x, sigma_y))))

    out = []
    for x0, y0 in xy:
        ix = int(np.rint(x0))
        iy = int(np.rint(y0))
        y0w = max(0, iy - half)
        y1w = min(z.shape[0], iy + half + 1)
        x0w = max(0, ix - half)
        x1w = min(z.shape[1], ix + half + 1)
        patch = z[y0w:y1w, x0w:x1w]
        if patch.size == 0 or not np.all(np.isfinite(patch)):
            out.append((float(x0), float(y0)))
            continue

        yy, xx = np.indices(patch.shape, dtype=float)
        xx += x0w
        yy += y0w
        amp = float(np.nanmax(patch))

        def residuals(params: Sequence[float]) -> np.ndarray:
            xc, yc = params
            model = amp * np.exp(-0.5 * (((xx - xc) / sigma_x) ** 2 + ((yy - yc) / sigma_y) ** 2))
            return (patch - model).ravel()

        fit = least_squares(
            residuals,
            x0=np.array([x0, y0], dtype=float),
            bounds=(
                np.array([x0w, y0w], dtype=float),
                np.array([x1w - 1, y1w - 1], dtype=float),
            ),
        )
        out.append((float(fit.x[0]), float(fit.x[1])))
    return np.asarray(out, dtype=np.float64)


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


def catalog_astrometry_qc(
    image: np.ndarray,
    *,
    imwcs: Any,
    catalog: str | Any | np.ndarray,
    catalog_path: str | None = None,
    min_flux: float = 0.0,
    n_catalog_sources: int = 50,
    n_measured_sources: int | None = None,
    min_separation_px: int = 15,
    centroid_method: Literal["none", "centroid", "gaussian_fixed_beam"] = "gaussian_fixed_beam",
    beam_fwhm_deg: tuple[float | None, float | None] = (None, None),
    pointlike_axis_ratio_max: float = 1.8,
    max_sep_arcsec: float = 120.0,
    min_matches: int = 5,
) -> MutableMapping[str, Any]:
    """
    End-to-end catalog-based astrometric QC.

    Steps:
    1) Load reference catalog and filter to image footprint.
    2) Select top ``n_catalog_sources`` catalog entries (by flux if available).
    3) Detect bright peaks in the image (with NMS).
    4) Optionally refine centroids.
    5) Convert measured peaks to SkyCoord and match to catalog SkyCoord.
    6) Summarize separations and compute match yield within ``max_sep_arcsec``.

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

    peaks_xy = detect_peaks(
        z,
        n_peaks=int(n_measured_sources),
        min_separation_px=int(min_separation_px),
        finite_only=True,
    )
    if peaks_xy.size == 0:
        return {
            "ok": False,
            "reason": "no_measured_peaks",
            "n_catalog_visible": int(cat_xy.shape[0]),
            "n_catalog_used": int(selected_cat_xy.shape[0]),
            "n_measured": 0,
            "n_matched": 0,
            "matched_fraction": 0.0,
            "max_sep_arcsec": float(max_sep_arcsec),
            **summarize_angular_separations_deg(np.array([], dtype=float)),
        }

    if centroid_method == "gaussian_fixed_beam":
        bmaj_deg, bmin_deg = beam_fwhm_deg
        if bmaj_deg is None or bmin_deg is None:
            raise ValueError("beam_fwhm_deg must be provided for gaussian_fixed_beam centroiding.")

    refined_xy = refine_centroids(
        z,
        peaks_xy,
        method=centroid_method,
        imwcs=imwcs,
        beam_fwhm_deg=beam_fwhm_deg,
    )

    measured_sky = pixels_to_skycoord(refined_xy, imwcs, origin=0)

    # Convert selected catalog xy back to sky via WCS to avoid depending on the
    # original catalog sky ordering after filtering/selection.
    catalog_sky = pixels_to_skycoord(selected_cat_xy, imwcs, origin=0)

    _idx, sep_deg = match_sky_nearest(measured_sky, catalog_sky)
    sep_arcsec = np.asarray(sep_deg, dtype=float) * 3600.0
    matched = np.isfinite(sep_arcsec) & (sep_arcsec <= float(max_sep_arcsec))
    n_matched = int(np.count_nonzero(matched))
    frac = float(n_matched / max(1, refined_xy.shape[0]))

    stats = summarize_angular_separations_deg(sep_deg[matched])
    ok = bool(n_matched >= int(min_matches))

    return {
        "ok": ok,
        "reason": "ok" if ok else "too_few_matches",
        "n_catalog_visible": int(cat_xy.shape[0]),
        "n_catalog_used": int(selected_cat_xy.shape[0]),
        "n_measured": int(refined_xy.shape[0]),
        "n_matched": n_matched,
        "matched_fraction": frac,
        "max_sep_arcsec": float(max_sep_arcsec),
        **stats,
    }

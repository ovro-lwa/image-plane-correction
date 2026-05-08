"""
Structured quality assurance (QA) for dewarped vs raw images (catalog astrometry, logging helpers).

Use from :func:`~image_plane_correction.flow.calcflow` or from tuning harnesses;
keep imaging logic in ``source_detection`` / ``util.runqa``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, MutableMapping, Union

import numpy as np

from . import source_detection

logger = logging.getLogger(__name__)


def check_reference_sky(reference_sky: Any, *, label: str = "reference_sky") -> dict[str, Any]:
    """
    Validate the reference sky image before it is used for flow solving.

    Fails fast if the map contains any non-finite values or if it is entirely zeros.
    Returns a small summary dict and logs a one-line summary at INFO level.
    """
    arr = np.asarray(reference_sky)
    if arr.size == 0:
        raise ValueError(f"{label} is empty.")

    finite = np.isfinite(arr)
    finite_frac = float(np.mean(finite))
    if finite_frac < 1.0:
        n_bad = int(arr.size - int(np.count_nonzero(finite)))
        raise ValueError(f"{label} contains non-finite values (n_bad={n_bad}, finite_frac={finite_frac:.6f}).")

    nonzero = arr != 0
    nonzero_frac = float(np.mean(nonzero))
    if nonzero_frac == 0.0:
        raise ValueError(f"{label} is all zeros (shape={arr.shape}).")

    mn = float(np.min(arr))
    mx = float(np.max(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    summary = {
        "shape": tuple(int(x) for x in arr.shape),
        "dtype": str(arr.dtype),
        "finite_frac": finite_frac,
        "nonzero_frac": nonzero_frac,
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
    }
    logger.info(
        "QA %s: shape=%s dtype=%s finite_frac=%.6f nonzero_frac=%.6f min=%.6g max=%.6g mean=%.6g std=%.6g",
        label,
        summary["shape"],
        summary["dtype"],
        finite_frac,
        nonzero_frac,
        mn,
        mx,
        mean,
        std,
    )
    return summary


def beam_fwhm_deg_for_catalog_qa(
    image_fn: str,
    *,
    bmaj_deg_override: float | None,
    bmin_deg_override: float | None,
) -> tuple[float | None, float | None]:
    """FWHM major/minor in degrees from overrides or FITS ``BMAJ`` / ``BMIN``."""
    if bmaj_deg_override is not None and bmin_deg_override is not None:
        return float(bmaj_deg_override), float(bmin_deg_override)
    from astropy.io import fits

    hdr = fits.getheader(image_fn)
    if "BMAJ" not in hdr or "BMIN" not in hdr:
        raise ValueError(
            "catalog QA requires BMAJ/BMIN in the image FITS header, or beam overrides "
            "(degrees) via CatalogAstrometryQAParams / --catalog-qa-beam-deg."
        )
    return float(hdr["BMAJ"]), float(hdr["BMIN"])


def _attach_prefixed(target: MutableMapping[str, Any], prefix: str, d: Mapping[str, Any]) -> None:
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, (float, np.floating)):
            target[key] = float(v)
        elif isinstance(v, (int, np.integer)):
            target[key] = int(v)
        elif isinstance(v, bool):
            target[key] = bool(v)
        else:
            target[key] = v


@dataclass
class CatalogAstrometryQAParams:
    """Parameters for :func:`catalog_astrometry_metrics_pair` (PyBDSF-based measurement)."""

    max_sep_arcsec: float = 600.0
    min_matches: int = 5
    n_catalog_sources: int = 50
    n_measured_sources: int | None = None
    min_separation_px: int = 15
    min_flux: float = 0.0
    pointlike_axis_ratio_max: float = 1.8
    bmaj_deg: float | None = None
    bmin_deg: float | None = None
    beam_pa_deg: float = 0.0
    bdsf_thresh: str | None = "hard"
    bdsf_thresh_isl: float = 10.0
    bdsf_thresh_pix: float = 5.0
    bdsf_minpix_isl: int | None = None
    bdsf_quiet: bool = True
    restfreq_hz: float | None = None
    bdsf_ncores: int = 4


CatalogLike = Union[str, Any, np.ndarray]


def catalog_astrometry_metrics_pair(
    image_raw: np.ndarray,
    image_dewarped: np.ndarray,
    imwcs: Any,
    catalog: CatalogLike,
    catalog_path: str | None,
    image_fn: str,
    params: CatalogAstrometryQAParams | None = None,
) -> dict[str, Any]:
    """
    Run catalog astrometry QA on raw and dewarped images; return flat metrics for JSON rows.

    Keys match the tuning harness: ``catalog_qa_raw_*``, ``catalog_qa_dewarped_*``,
    ``catalog_qa_delta_median_arcsec``.
    """
    p = params or CatalogAstrometryQAParams()
    beam_deg = beam_fwhm_deg_for_catalog_qa(
        image_fn,
        bmaj_deg_override=p.bmaj_deg,
        bmin_deg_override=p.bmin_deg,
    )
    thresh = None if (p.bdsf_thresh is not None and str(p.bdsf_thresh).lower() == "auto") else p.bdsf_thresh
    common_kw = dict(
        imwcs=imwcs,
        catalog=catalog,
        catalog_path=catalog_path,
        min_flux=float(p.min_flux),
        n_catalog_sources=int(p.n_catalog_sources),
        n_measured_sources=p.n_measured_sources,
        min_separation_px=int(p.min_separation_px),
        beam_fwhm_deg=beam_deg,
        beam_pa_deg=float(p.beam_pa_deg),
        max_sep_arcsec=float(p.max_sep_arcsec),
        min_matches=int(p.min_matches),
        pointlike_axis_ratio_max=float(p.pointlike_axis_ratio_max),
        bdsf_thresh=thresh,
        bdsf_thresh_isl=float(p.bdsf_thresh_isl),
        bdsf_thresh_pix=float(p.bdsf_thresh_pix),
        bdsf_minpix_isl=p.bdsf_minpix_isl,
        bdsf_quiet=bool(p.bdsf_quiet),
        restfreq_hz=p.restfreq_hz,
        bdsf_ncores=int(p.bdsf_ncores),
    )
    raw_qa = source_detection.catalog_astrometry_qa(np.asarray(image_raw), **common_kw)
    dew_qa = source_detection.catalog_astrometry_qa(np.asarray(image_dewarped), **common_kw)

    out: dict[str, Any] = {}
    _attach_prefixed(out, "catalog_qa_raw_", dict(raw_qa))
    _attach_prefixed(out, "catalog_qa_dewarped_", dict(dew_qa))

    if bool(raw_qa.get("ok")) and bool(dew_qa.get("ok")):
        raw_med = raw_qa.get("median_arcsec")
        dew_med = dew_qa.get("median_arcsec")
        if (
            raw_med is not None
            and dew_med is not None
            and np.isfinite(float(raw_med))
            and np.isfinite(float(dew_med))
        ):
            out["catalog_qa_delta_median_arcsec"] = float(raw_med) - float(dew_med)
        else:
            out["catalog_qa_delta_median_arcsec"] = None
    else:
        out["catalog_qa_delta_median_arcsec"] = None
    return out


def bright_source_qa_kwargs(
    imwcs: Any,
    header: Mapping[str, Any],
    *,
    catalog: str,
    catalog_path: str | None,
    n_sources: int,
) -> dict[str, Any]:
    """Keyword dict for :func:`~image_plane_correction.util.log_bright_source_flux_comparison`."""
    return {
        "imwcs": imwcs,
        "catalog": catalog,
        "catalog_path": catalog_path,
        "n_sources": int(n_sources),
        "bmaj_deg": float(header["BMAJ"]) if "BMAJ" in header else None,
        "bmin_deg": float(header["BMIN"]) if "BMIN" in header else None,
        "bpa_deg": float(header["BPA"]) if "BPA" in header else 0.0,
    }


def log_bright_source_alignment(dewarped: Any, **kwargs: Any) -> None:
    """Dispatch to legacy printer (PyBDSF positions vs catalog)."""
    from .util import log_bright_source_flux_comparison

    log_bright_source_flux_comparison(dewarped, **kwargs)


"""
Spectral and differential metrics for dense optical-flow fields.

Offsets follow the same convention as ``Flow.offsets``: shape ``(..., 2)`` with
``[..., 0]`` = displacement along increasing column index (x / RA-like axis) and
``[..., 1]`` = displacement along increasing row index (y / Dec-like axis).
"""

from __future__ import annotations

from typing import Any, MutableMapping, Tuple, Union

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


ArrayLike = Union[np.ndarray, Any]


def _as_numpy_offsets(offsets: Union[ArrayLike, Any]) -> np.ndarray:
    """Return ``(H, W, 2)`` float array of offsets."""
    if hasattr(offsets, "offsets"):
        arr = offsets.offsets
    else:
        arr = offsets
    if hasattr(arr, "__array__"):
        out = np.asarray(arr.__array__(), dtype=np.float64)
    else:
        out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 3 or out.shape[-1] != 2:
        raise ValueError(f"offsets must have shape (H, W, 2); got {out.shape}")
    return out


def deg_per_pix_from_wcs(imwcs: WCS) -> float:
    """
    Mean absolute pixel scale of the WCS in degrees per pixel.

    Uses ``proj_plane_pixel_scales`` (same approach as ``calcflow`` / beam code).
    """
    scales = proj_plane_pixel_scales(imwcs)
    if hasattr(scales, "to"):
        deg = np.asarray(scales.to(u.deg).value, dtype=np.float64)
    else:
        deg = np.abs(np.asarray(scales, dtype=np.float64))
    if deg.size == 0:
        raise ValueError("Empty proj_plane_pixel_scales output.")
    return float(np.mean(deg))


def flow_div_curl(offsets: Union[ArrayLike, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divergence and scalar curl of a 2D offset field.

    ``div = ∂u/∂x + ∂v/∂y``, ``curl = ∂v/∂x - ∂u/∂y`` with ``u`` = column offset,
    ``v`` = row offset (numpy axis 1 = x, axis 0 = y).
    """
    o = _as_numpy_offsets(offsets)
    uu = o[..., 0]
    vv = o[..., 1]
    gu_y, gu_x = np.gradient(uu)
    gv_y, gv_x = np.gradient(vv)
    div = gu_x + gv_y
    curl = gv_x - gu_y
    return div.astype(np.float64), curl.astype(np.float64)


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float64)


def _fft_power_kdeg(
    field: np.ndarray,
    deg_per_pix: float,
    *,
    mask: np.ndarray | None = None,
    taper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(power, k_deg)`` for the 2D FFT of ``field`` (|FFT|^2 per mode).

    Frequencies are mapped to cycles per degree via ``k_deg = k_pix / deg_per_pix``.
    """
    z = np.asarray(field, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"field must be 2D; got shape {z.shape}")
    h, w = z.shape
    if deg_per_pix <= 0 or not np.isfinite(deg_per_pix):
        raise ValueError(f"deg_per_pix must be finite and positive; got {deg_per_pix}")

    zz = z.copy()
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        if m.shape != z.shape:
            raise ValueError(f"mask shape {m.shape} != field shape {z.shape}")
        zz = zz * m

    if taper:
        zz = zz * _hann2d(h, w)

    fz = np.fft.fft2(zz)
    power = (np.abs(fz) ** 2).astype(np.float64)

    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    kx, ky = np.meshgrid(fx, fy)
    k_pix = np.sqrt(kx**2 + ky**2)
    k_deg = (k_pix / float(deg_per_pix)).astype(np.float64)
    return power, k_deg


def radial_power_spectrum_2d(
    field: np.ndarray,
    deg_per_pix: float,
    *,
    mask: np.ndarray | None = None,
    n_bins: int = 128,
    taper: bool = True,
    exclude_dc: bool = True,
    dc_kdeg_eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radially averaged power vs spatial frequency ``k`` in cycles / degree.

    Returns ``(k_centers, mean_power_per_bin)`` where ``mean_power_per_bin`` is the
    mean of ``|FFT|^2`` over modes falling in each ``k_deg`` shell (not a calibrated
    PSD); band fractions should use :func:`band_power_from_field`.
    """
    power, k_deg = _fft_power_kdeg(field, deg_per_pix, mask=mask, taper=taper)

    if exclude_dc:
        power = power.copy()
        power[k_deg <= dc_kdeg_eps] = 0.0

    kmax_deg = float(np.max(k_deg))
    if not np.isfinite(kmax_deg) or kmax_deg <= 0:
        return np.zeros(n_bins, dtype=np.float64), np.zeros(n_bins, dtype=np.float64)

    edges = np.linspace(0.0, kmax_deg, n_bins + 1)
    k_flat = k_deg.ravel()
    p_flat = power.ravel()

    sum_p, _ = np.histogram(k_flat, bins=edges, weights=p_flat)
    counts, _ = np.histogram(k_flat, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = np.where(counts > 0, sum_p / counts, 0.0).astype(np.float64)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, profile


def radial_mean_power_kdeg(
    field: np.ndarray,
    deg_per_pix: float,
    *,
    mask: np.ndarray | None = None,
    n_bins: int = 128,
    taper: bool = True,
    exclude_dc: bool = True,
    dc_kdeg_eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for :func:`radial_power_spectrum_2d`."""
    return radial_power_spectrum_2d(
        field,
        deg_per_pix,
        mask=mask,
        n_bins=n_bins,
        taper=taper,
        exclude_dc=exclude_dc,
        dc_kdeg_eps=dc_kdeg_eps,
    )


def band_power_from_field(
    field: np.ndarray,
    deg_per_pix: float,
    band_deg: Tuple[float, float],
    *,
    mask: np.ndarray | None = None,
    taper: bool = True,
    exclude_dc: bool = True,
    dc_kdeg_eps: float = 1e-9,
) -> Tuple[float, float]:
    """
    Return ``(power_in_band, power_total)`` Summing ``|FFT|^2`` over angular wavenumber.

    Band is inclusive in wavelength: ``band_deg = (lambda_min, lambda_max)`` in degrees,
    corresponding to ``k_deg`` in ``[1/lambda_max, 1/lambda_min]``.

    Very long wavelengths (large ``lambda_deg / deg_per_pix`` vs image size) are poorly
    resolved on a finite grid; band fractions should be interpreted accordingly.
    """
    lam0, lam1 = float(band_deg[0]), float(band_deg[1])
    if lam0 <= 0 or lam1 <= 0 or lam1 < lam0:
        raise ValueError(f"band_deg must be (lambda_min, lambda_max) with 0 < min <= max; got {band_deg}")
    k_lo = 1.0 / lam1
    k_hi = 1.0 / lam0

    power, k_deg = _fft_power_kdeg(field, deg_per_pix, mask=mask, taper=taper)

    valid = np.ones_like(k_deg, dtype=bool)
    if exclude_dc:
        valid &= k_deg > dc_kdeg_eps

    in_band = valid & (k_deg >= k_lo) & (k_deg <= k_hi)
    p_band = float(np.sum(power[in_band]))
    p_tot = float(np.sum(power[valid]))
    return p_band, p_tot


def structure_score(
    offsets: Union[ArrayLike, Any],
    imwcs: WCS,
    band_deg: Tuple[float, float] = (10.0, 160.0),
    mask: np.ndarray | None = None,
    eps: float = 1e-18,
) -> MutableMapping[str, Any]:
    """
    Aggregate ionospheric-structure metrics for a dense flow (higher score is better).

    Combines in-band fraction of divergence power, curl-to-divergence ratio in-band,
    and out-of-band divergence power fraction.

    Parameters
    ----------
    offsets
        ``(H, W, 2)`` array or object with ``.offsets``.
    imwcs
        Astropy WCS for pixel scale (degrees per pixel).
    band_deg
        Target wavelength band ``(lambda_min, lambda_max)`` in degrees (default 20–100).
    mask
        Optional ``(H, W)`` weights for FFT windowing (0–1).

    Returns
    -------
    dict
        Includes ``band_power_frac_div``, ``band_power_frac_curl``,
        ``curl_to_div_ratio_band``, ``out_of_band_penalty``, ``structure_score``,
        ``deg_per_pix``, ``k_deg_band``.
    """
    dpp = deg_per_pix_from_wcs(imwcs)
    div, curl = flow_div_curl(offsets)

    bd_div, td_div = band_power_from_field(div, dpp, band_deg, mask=mask)
    bd_curl, td_curl = band_power_from_field(curl, dpp, band_deg, mask=mask)

    lam0, lam1 = float(band_deg[0]), float(band_deg[1])
    k_band = (1.0 / lam1, 1.0 / lam0)

    if td_div <= eps and td_curl <= eps:
        out: MutableMapping[str, Any] = {
            "deg_per_pix": dpp,
            "k_deg_band": k_band,
            "band_power_frac_div": 0.0,
            "band_power_frac_curl": 0.0,
            "curl_to_div_ratio_band": 0.0,
            "out_of_band_penalty": 0.0,
            "structure_score": 0.0,
            "band_power_div": bd_div,
            "total_power_div": td_div,
            "band_power_curl": bd_curl,
            "total_power_curl": td_curl,
        }
        return out

    frac_div = bd_div / (td_div + eps)
    frac_curl = bd_curl / (td_curl + eps) if td_curl > eps else 0.0
    ratio_band = bd_curl / (bd_div + eps)
    oob = max(0.0, (td_div - bd_div) / (td_div + eps))

    # Higher is better: prefer divergence concentrated in band and low curl in band
    score = float(frac_div * (1.0 / (1.0 + ratio_band)) * max(0.0, 1.0 - min(1.0, oob)))

    return {
        "deg_per_pix": dpp,
        "k_deg_band": k_band,
        "band_power_frac_div": float(frac_div),
        "band_power_frac_curl": float(frac_curl),
        "curl_to_div_ratio_band": float(ratio_band),
        "out_of_band_penalty": float(oob),
        "structure_score": score,
        "band_power_div": float(bd_div),
        "total_power_div": float(td_div),
        "band_power_curl": float(bd_curl),
        "total_power_curl": float(td_curl),
    }



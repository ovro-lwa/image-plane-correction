"""Panel + HoloViews live tip-tilt / dewarp vector-field app.

Celestial Mollweide quiver (RA increasing left) from a tip-tilt named-array
``.npz``, with optional active-beam pointing overlays.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import param

from image_plane_correction.live_map import (
    BeamPointingRecord,
    TipTiltMap,
    load_tiptilt_npz,
    resolve_tiptilt_path,
)
from image_plane_correction.tiptilt_map import default_location
from image_plane_correction.viz.beam_overlay import (
    BeamMapRow,
    _hover_opts,
    _strip_text_from_hover,
    build_beam_overlay,
    collect_beam_rows,
    load_demo_pointings,
    rows_fingerprint,
)

LOG = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Mollweide plot limits (standard equal-area ellipsoid).
_MOLL_X_MAX = 2.0 * np.sqrt(2.0)
_MOLL_Y_MAX = np.sqrt(2.0)
# Same pad on X/Y so the ellipse is not clipped by mismatched aspect.
_MOLL_PAD = 1.18
_MOLL_PLOT_XLIM = _MOLL_X_MAX * _MOLL_PAD
_MOLL_PLOT_YLIM = _MOLL_Y_MAX * _MOLL_PAD
# 2:1 data aspect (X_MAX/Y_MAX == 2); frame matches exactly.
_MOLL_FRAME_W = 880
_MOLL_FRAME_H = 440
_MOLL_WIDTH = 960
# Extra pane height so the south limb / Dec labels are not clipped.
_MOLL_HEIGHT = 620
# Equal-area celestial sample spacing (deg) at the equator; stride multiplies this.
_BASE_SAMPLE_SPACING_DEG = 2.5

# Typical OVRO-LWA cleaned BMAJ (~0.20°) — PSF major-axis FWHM.
_BMAJ_ARCMIN = 12.0
# Physical |shift| that maps to full arrow length (and clips above).
_ARROW_MAX_ARCMIN = 10.0
_ARROW_REF_ARCMIN = 10.0
# Max displayed arrow length = Mollweide equatorial length of this angle.
_ARROW_MAX_DISPLAY_DEG = 20.0

# Discrete age slider: rightmost stop is unlimited ("all").
_POINTING_AGE_ALL = float("inf")
_POINTING_AGE_SLIDER_OPTIONS = {
    "1": 1.0,
    "10": 10.0,
    "30": 30.0,
    "60": 60.0,
    "120": 120.0,
    "300": 300.0,
    "600": 600.0,
    "1200": 1200.0,
    "1800": 1800.0,
    "3600": 3600.0,
    "all": _POINTING_AGE_ALL,
}
# Finite stand-in when calling APIs that reject non-finite age limits.
_POINTING_AGE_DISCOVER_FALLBACK_SEC = 1.0e9


def wrap_lon_offset_deg(lon_deg: np.ndarray, central_lon_deg: float = 0.0) -> np.ndarray:
    """Longitude relative to ``central_lon_deg``, wrapped to ``[-180, 180)``."""
    return (
        np.mod(np.asarray(lon_deg, dtype=np.float64) - float(central_lon_deg) + 180.0, 360.0)
        - 180.0
    )


def _mollweide_theta(lat_rad: np.ndarray, *, n_iter: int = 10) -> np.ndarray:
    """Solve Mollweide auxiliary angle θ from latitude (Newton)."""
    lat = np.asarray(lat_rad, dtype=np.float64)
    # Poles: θ = ±π/2.
    theta = lat.copy()
    finite = np.isfinite(lat) & (np.abs(lat) < (0.5 * np.pi - 1e-12))
    if not np.any(finite):
        return np.where(lat >= 0.0, 0.5 * np.pi, -0.5 * np.pi)
    th = lat[finite].copy()
    for _ in range(int(n_iter)):
        # 2θ + sin(2θ) = π sin(φ)
        f = 2.0 * th + np.sin(2.0 * th) - np.pi * np.sin(lat[finite])
        fp = 2.0 + 2.0 * np.cos(2.0 * th)
        th = th - f / np.where(np.abs(fp) > 1e-12, fp, 1.0)
    theta = np.full_like(lat, np.nan, dtype=np.float64)
    theta[finite] = th
    theta = np.where(lat >= (0.5 * np.pi - 1e-12), 0.5 * np.pi, theta)
    theta = np.where(lat <= (-0.5 * np.pi + 1e-12), -0.5 * np.pi, theta)
    return theta


def radec_to_mollweide(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    central_ra_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project RA/Dec to Mollweide ``(x, y)`` with **RA increasing to the left**.

    ``central_ra_deg`` is mapped to ``x=0``. Longitude offset uses
    ``-(RA − central)`` so increasing RA moves left (astronomical convention).
    """
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    # Negate so RA increases leftward.
    lam = np.deg2rad(wrap_lon_offset_deg(-ra, -float(central_ra_deg)))
    phi = np.deg2rad(dec)
    theta = _mollweide_theta(phi)
    x = (2.0 * np.sqrt(2.0) / np.pi) * lam * np.cos(theta)
    y = np.sqrt(2.0) * np.sin(theta)
    return x, y


def azel_to_radec(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    *,
    obstime_unix: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert AltAz (deg) at ``obstime_unix`` to ICRS RA/Dec (deg)."""
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time

    az = np.asarray(az_deg, dtype=np.float64)
    el = np.asarray(el_deg, dtype=np.float64)
    obstime = Time(float(obstime_unix), format="unix", scale="utc")
    aa = SkyCoord(
        az=az,
        alt=el,
        unit="deg",
        frame=AltAz(obstime=obstime, location=default_location()),
    )
    sky = aa.icrs
    return np.asarray(sky.ra.deg, dtype=np.float64), np.asarray(sky.dec.deg, dtype=np.float64)


def radec_to_azel(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    obstime_unix: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ICRS RA/Dec (deg) at ``obstime_unix`` to AltAz (deg)."""
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time

    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    obstime = Time(float(obstime_unix), format="unix", scale="utc")
    sky = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    aa = sky.transform_to(AltAz(obstime=obstime, location=default_location()))
    return (
        np.asarray(aa.az.deg, dtype=np.float64),
        np.asarray(aa.alt.deg, dtype=np.float64),
    )


def zenith_ra_deg(obstime_unix: float) -> float:
    """ICRS RA (deg) of the local zenith at ``obstime_unix``."""
    ra, _dec = azel_to_radec(0.0, 90.0, obstime_unix=obstime_unix)
    return float(np.asarray(ra).reshape(-1)[0])


# Back-compat aliases (previous ZEA / Mollweide az-el helpers).
def azel_to_zea(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    *,
    central_az_deg: float = 0.0,
    obstime_unix: float = 0.0,
    central_ra_deg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated: project az/el via celestial Mollweide (RA left)."""
    ra, dec = azel_to_radec(az_deg, el_deg, obstime_unix=obstime_unix)
    if central_ra_deg is None:
        central_ra_deg = zenith_ra_deg(obstime_unix) if obstime_unix else 0.0
    return radec_to_mollweide(ra, dec, central_ra_deg=float(central_ra_deg))


azel_to_mollweide = azel_to_zea


def _hv_curve(x: np.ndarray, y: np.ndarray, **opts):
    """HoloViews Curve with Mollweide axis names."""
    import holoviews as hv

    return hv.Curve((x, y), kdims=["moll_x"], vdims=["moll_y"]).opts(**opts)


def mollweide_outline(n: int = 256):
    """Mollweide projection limb (full-sky ellipse)."""
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
    return _hv_curve(
        _MOLL_X_MAX * np.cos(t),
        _MOLL_Y_MAX * np.sin(t),
        color="#444444",
        line_width=1.5,
    )


def mollweide_graticule(
    *,
    central_ra_deg: float = 0.0,
    ra_step_deg: float = 45.0,
    dec_step_deg: float = 30.0,
    n: int = 72,
):
    """RA/Dec graticule + labels in Mollweide coordinates (RA increases left)."""
    import holoviews as hv

    hv.extension("bokeh", logo=False)

    curves = []
    style = dict(color="#bbbbbb", line_width=0.6)
    # Constant-RA meridians.
    ras = np.arange(0.0, 360.0, float(ra_step_deg))
    dec_line = np.linspace(-90.0, 90.0, int(n))
    for ra in ras:
        x, y = radec_to_mollweide(
            np.full_like(dec_line, ra), dec_line, central_ra_deg=central_ra_deg
        )
        curves.append(_hv_curve(x, y, **style))
    # Constant-Dec parallels (skip poles).
    decs = np.arange(-90.0 + dec_step_deg, 90.0, float(dec_step_deg))
    ra_line = np.linspace(0.0, 360.0, int(n), endpoint=True)
    for dec in decs:
        x, y = radec_to_mollweide(
            ra_line, np.full_like(ra_line, dec), central_ra_deg=central_ra_deg
        )
        curves.append(_hv_curve(x, y, **style))

    # Labels: RA along the equator; Dec along a left-side meridian.
    lab_x: list[float] = []
    lab_y: list[float] = []
    lab_text: list[str] = []
    for ra in ras:
        x, y = radec_to_mollweide(
            np.asarray(ra), np.asarray(0.0), central_ra_deg=central_ra_deg
        )
        lab_x.append(float(np.asarray(x).reshape(-1)[0]))
        lab_y.append(float(np.asarray(y).reshape(-1)[0]) + 0.06)
        lab_text.append(f"RA {ra:g}°")
    # Left of centre (RA increases left → central+offset).
    dec_label_ra = float(central_ra_deg) + 100.0
    for dec in decs:
        if abs(float(dec)) < 1e-9:
            continue  # equator already has RA labels
        x, y = radec_to_mollweide(
            np.asarray(dec_label_ra),
            np.asarray(dec),
            central_ra_deg=central_ra_deg,
        )
        lab_x.append(float(np.asarray(x).reshape(-1)[0]))
        lab_y.append(float(np.asarray(y).reshape(-1)[0]))
        lab_text.append(f"Dec {dec:+g}°")
    labels = hv.Labels(
        {"moll_x": lab_x, "moll_y": lab_y, "label": lab_text},
        kdims=["moll_x", "moll_y"],
        vdims=["label"],
    ).opts(
        text_color="#666666",
        text_font_size="8pt",
        text_align="center",
        text_baseline="middle",
        show_legend=False,
    )
    return hv.Overlay(curves) * labels


def sample_spacing_deg(stride: int = 1) -> float:
    """Equatorial celestial sample spacing (deg) for equal-area display lattice."""
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be an int >= 1")
    return float(_BASE_SAMPLE_SPACING_DEG * stride)


def downsample_tiptilt_uv(
    tip: TipTiltMap,
    *,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return flat arrays ``(az, el, dra, ddec, magnitude)`` for quiver display.

    Legacy: every ``stride``-th az/el grid node. Prefer
    :func:`sample_tiptilt_equal_area` for Mollweide (uniform on-sky density).
    Magnitude is in **arcsec** (native NPZ units).
    """
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be an int >= 1")
    az_1d = tip.az_deg[::stride]
    el_1d = tip.el_deg[::stride]
    dra = tip.dra_arcsec[::stride, ::stride]
    ddec = tip.ddec_arcsec[::stride, ::stride]
    az_grid, el_grid = np.meshgrid(az_1d, el_1d, indexing="xy")
    mag = np.hypot(dra, ddec)
    finite = np.isfinite(dra) & np.isfinite(ddec) & np.isfinite(mag)
    return (
        az_grid[finite].astype(np.float64, copy=False),
        el_grid[finite].astype(np.float64, copy=False),
        dra[finite].astype(np.float64, copy=False),
        ddec[finite].astype(np.float64, copy=False),
        mag[finite].astype(np.float64, copy=False),
    )


def sample_tiptilt_equal_area(
    tip: TipTiltMap,
    *,
    spacing_deg: float = 5.0,
    min_el_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample tip-tilt on an equal-area celestial lattice.

    Declination every ``spacing_deg``; at latitude δ, RA step is
    ``spacing_deg / max(cos δ, ε)``. That is uniform solid angle, so the
    points appear evenly spaced on Mollweide (an equal-area projection).
    Each (RA, Dec) is converted to AltAz at the map obstime and bilinearly
    sampled from the tip-tilt NPZ; below-horizon / off-grid cells are dropped.

    Returns
    -------
    az, el, dra, ddec, magnitude
        Flat arrays; magnitude in arcsec.
    """
    spacing = float(spacing_deg)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_deg must be a positive finite float")

    obstime = float(tip.obstime_unix)
    # Half-step offset keeps poles from being over-weighted.
    decs = np.arange(-90.0 + 0.5 * spacing, 90.0, spacing, dtype=np.float64)
    ra_chunks: list[np.ndarray] = []
    dec_chunks: list[np.ndarray] = []
    for dec in decs:
        cos_dec = float(np.cos(np.deg2rad(dec)))
        ra_step = spacing / max(abs(cos_dec), 0.05)
        n_ra = max(1, int(round(360.0 / ra_step)))
        ras = (np.arange(n_ra, dtype=np.float64) + 0.5) * (360.0 / n_ra)
        ra_chunks.append(ras)
        dec_chunks.append(np.full(n_ra, float(dec), dtype=np.float64))

    if not ra_chunks:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    ra_all = np.concatenate(ra_chunks)
    dec_all = np.concatenate(dec_chunks)
    az_all, el_all = radec_to_azel(ra_all, dec_all, obstime_unix=obstime)
    above = np.isfinite(el_all) & (el_all >= float(min_el_deg))
    az_c = az_all[above]
    el_c = el_all[above]
    if az_c.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    # Batch bilinear sample via the map interpolators.
    pts = np.column_stack([el_c, az_c])
    dra_c = np.asarray(tip._interp_dra(pts), dtype=np.float64)
    ddec_c = np.asarray(tip._interp_ddec(pts), dtype=np.float64)
    ok = np.isfinite(dra_c) & np.isfinite(ddec_c)
    az_a = az_c[ok]
    el_a = el_c[ok]
    dra_a = dra_c[ok]
    ddec_a = ddec_c[ok]
    mag_a = np.hypot(dra_a, ddec_a)
    return az_a, el_a, dra_a, ddec_a, mag_a


def _grid_spacing_deg(tip: TipTiltMap, *, stride: int) -> float:
    """Display lattice spacing (deg); equal-area equatorial step."""
    _ = tip
    return sample_spacing_deg(stride)


def _arrow_max_display_length(display_deg: float = _ARROW_MAX_DISPLAY_DEG) -> float:
    """
    Max arrow length in Mollweide units for ``display_deg`` at the equator.

    Near Dec≈0, ``dx/dλ ≈ 2√2/π``. Physical |shift| = ``_ARROW_REF_ARCMIN``
    maps to this length; larger shifts clip.
    """
    spacing_rad = np.deg2rad(max(float(display_deg), 1e-3))
    return float((2.0 * np.sqrt(2.0) / np.pi) * spacing_rad)


def _moll_frame_opts(title: str = ""):
    """Fixed Mollweide frame; no Cartesian x/y axes (limb carries the context)."""
    return dict(
        title=title,
        width=_MOLL_WIDTH,
        height=_MOLL_HEIGHT,
        # frame_* alone (no aspect) — HoloViews ignores aspect when both width
        # and height are set, and mismatched pad was clipping the south limb.
        frame_width=_MOLL_FRAME_W,
        frame_height=_MOLL_FRAME_H,
        xlim=(-_MOLL_PLOT_XLIM, _MOLL_PLOT_XLIM),
        ylim=(-_MOLL_PLOT_YLIM, _MOLL_PLOT_YLIM),
        xlabel="",
        ylabel="",
        xaxis=None,
        yaxis=None,
        axiswise=False,
        shared_axes=True,
    )


def _display_uv_from_tip_tilt(
    az: np.ndarray,
    el: np.ndarray,
    dra: np.ndarray,
    ddec: np.ndarray,
    mag: np.ndarray,
    *,
    obstime_unix: float,
    central_ra_deg: float,
    arrow_max_len: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(x, y, u_disp, v_disp, mag_arcmin)`` for Mollweide display.

    Direction follows the projected celestial ``(dra, ddec)`` step. Length maps
    ``|shift|`` in arcmin onto ``[0, arrow_max_len]`` with
    ``_ARROW_REF_ARCMIN`` → ``arrow_max_len`` and clip at ``_ARROW_MAX_ARCMIN``.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra0, dec0 = azel_to_radec(az, el, obstime_unix=obstime_unix)
    x0, y0 = radec_to_mollweide(ra0, dec0, central_ra_deg=central_ra_deg)

    c0 = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg, frame="icrs")
    c1 = c0.spherical_offsets_by(dra * u.arcsec, ddec * u.arcsec)
    x1, y1 = radec_to_mollweide(
        np.asarray(c1.ra.deg, dtype=np.float64),
        np.asarray(c1.dec.deg, dtype=np.float64),
        central_ra_deg=central_ra_deg,
    )
    u_raw = x1 - x0
    v_raw = y1 - y0
    raw_len = np.hypot(u_raw, v_raw)
    angle = np.arctan2(v_raw, u_raw)
    tiny = raw_len < 1e-15
    if np.any(tiny):
        # Fallback: east-of-north from (dra, ddec); Mollweide x flips with RA.
        angle = np.where(tiny, np.arctan2(ddec, -dra), angle)

    mag_arcmin = np.asarray(mag, dtype=np.float64) / 60.0
    # Fixed physical range (arcmin): ref → full spacing-safe length; clip max.
    scale = float(arrow_max_len) / max(float(_ARROW_REF_ARCMIN), 1e-6)
    disp_len = np.clip(mag_arcmin * scale, 0.0, float(arrow_max_len))
    over = mag_arcmin > float(_ARROW_MAX_ARCMIN)
    if np.any(over):
        disp_len = np.where(over, float(arrow_max_len), disp_len)
    disp_len = np.where(np.isfinite(disp_len), disp_len, 0.0)
    u_disp = disp_len * np.cos(angle)
    v_disp = disp_len * np.sin(angle)
    return x0, y0, u_disp, v_disp, mag_arcmin


def build_vector_element(
    tip: TipTiltMap,
    *,
    stride: int = 1,
    central_ra_deg: Optional[float] = None,
):
    """
    Build tip-tilt ``VectorField`` plus an invisible hover ``Points`` layer.

    HoloViews' Bokeh VectorField hover only ships ``x``/``y``/``Angle``/
    ``Magnitude`` (and tiles each arrow three times for heads), so custom
    vdims show as ``???``. Hover is therefore handled by Points at each
    sample with az, el, |shift| (arcmin), and celestial angle.
    """
    import holoviews as hv

    obstime = float(tip.obstime_unix)
    if central_ra_deg is None:
        central_ra_deg = 0.0
    spacing = sample_spacing_deg(stride)
    az, el, dra, ddec, mag = sample_tiptilt_equal_area(tip, spacing_deg=spacing)
    if az.size == 0:
        return hv.VectorField(
            ([], [], [], []),
            kdims=["moll_x", "moll_y"],
            vdims=["Angle", "Magnitude"],
        ).opts(alpha=0.0)

    arrow_max = _arrow_max_display_length(_ARROW_MAX_DISPLAY_DEG)
    x0, y0, u, v, mag_am = _display_uv_from_tip_tilt(
        az,
        el,
        dra,
        ddec,
        mag,
        obstime_unix=obstime,
        central_ra_deg=float(central_ra_deg),
        arrow_max_len=arrow_max,
    )
    # Celestial offset angle: 0° = +ddec (north), 90° = +dra (east).
    angle_deg = np.rad2deg(np.arctan2(dra, ddec))
    # Length alone encodes |shift|; no redundant magnitude color scale.
    field = hv.VectorField.from_uv(
        {
            "moll_x": x0,
            "moll_y": y0,
            "u": u,
            "v": v,
        },
        kdims=["moll_x", "moll_y"],
        vdims=["u", "v"],
    ).opts(
        color="#4c78a8",
        magnitude="Magnitude",
        # Keep u/v lengths as Mollweide data units (10′ → 10° equatorial).
        rescale_lengths=False,
        normalize_lengths=False,
        tools=[],
        default_tools=["save", "pan", "wheel_zoom", "box_zoom", "reset"],
        line_width=1.0,
    )
    # Invisible hit targets at sample centres (not along the arrow shaft).
    hover_pts = hv.Points(
        {
            "moll_x": x0,
            "moll_y": y0,
            "az_deg": az,
            "el_deg": el,
            "shift_arcmin": mag_am,
            "angle_deg": angle_deg,
        },
        kdims=["moll_x", "moll_y"],
        vdims=["az_deg", "el_deg", "shift_arcmin", "angle_deg"],
    ).opts(
        size=6,
        alpha=0.0,
        **_hover_opts(),
    )
    return field * hover_pts


def build_vectorfield(
    tip: TipTiltMap,
    *,
    stride: int = 1,
    title: str = "",
    central_ra_deg: Optional[float] = None,
    beam_rows: Optional[List[BeamMapRow]] = None,
    show_nominal_marks: bool = True,
):
    """
    Build a celestial Mollweide HoloViews overlay.

    Limb + RA/Dec graticule + tip-tilt vectors (+ optional beam marks).
    """
    import holoviews as hv

    obstime = float(tip.obstime_unix)
    if central_ra_deg is None:
        central_ra_deg = 0.0
    outline = mollweide_outline()
    graticule = mollweide_graticule(central_ra_deg=float(central_ra_deg))
    field = build_vector_element(
        tip, stride=stride, central_ra_deg=float(central_ra_deg)
    )
    overlay = outline * graticule * field
    if beam_rows is not None:
        overlay = overlay * build_beam_overlay(
            beam_rows,
            obstime_unix=obstime,
            central_ra_deg=float(central_ra_deg),
            show_nominal_when_commanded=show_nominal_marks,
        )
    # Final strip: composing with the graticule/field Overlay can re-merge
    # HoverTools and put Labels/Text back into renderers (→ "???" on beams).
    frame_opts = _moll_frame_opts(title)
    hooks = list(frame_opts.get("hooks") or [])
    hooks.append(_strip_text_from_hover)
    frame_opts["hooks"] = hooks
    return overlay.opts(**frame_opts)


def _format_unix(ts: float) -> str:
    if not np.isfinite(ts):
        return "?"
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _fmt_opt(value: Optional[float], fmt: str = "{:.2f}") -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return fmt.format(float(value))


def _format_age_sec(age_sec: Optional[float]) -> str:
    """Human-readable age matching the pointing-age slider scale."""
    if age_sec is None or not np.isfinite(age_sec):
        return "—"
    age = max(0.0, float(age_sec))
    if age < 60.0:
        return f"{age:.0f} s"
    if age < 3600.0:
        minutes = int(age // 60.0)
        seconds = int(round(age - 60.0 * minutes))
        if seconds:
            return f"{minutes} m {seconds} s"
        return f"{minutes} m"
    hours = int(age // 3600.0)
    minutes = int((age - 3600.0 * hours) // 60.0)
    if minutes:
        return f"{hours} h {minutes} m"
    return f"{hours} h"


def _beam_table_html(rows: List[BeamMapRow]) -> str:
    """HTML beam table; stale rows are greyed (age threshold set by the slider)."""
    if not rows:
        return "<em>No active beams</em>"

    lines = [
        "<table style='border-collapse:collapse;font-size:13px'>",
        "<thead><tr>"
        "<th style='text-align:left;padding:2px 8px'>beam</th>"
        "<th style='text-align:left;padding:2px 8px'>status</th>"
        "<th style='text-align:right;padding:2px 8px'>az</th>"
        "<th style='text-align:right;padding:2px 8px'>el</th>"
        "<th style='text-align:right;padding:2px 8px'>nom az</th>"
        "<th style='text-align:right;padding:2px 8px'>nom el</th>"
        "<th style='text-align:right;padding:2px 8px'>cmd?</th>"
        "<th style='text-align:right;padding:2px 8px'>dra′</th>"
        "<th style='text-align:right;padding:2px 8px'>ddec′</th>"
        "<th style='text-align:right;padding:2px 8px'>|shift|′</th>"
        "<th style='text-align:right;padding:2px 8px'>age</th>"
        "</tr></thead><tbody>",
    ]
    for row in rows:
        dra_am = None if row.dra_arcsec is None else row.dra_arcsec / 60.0
        ddec_am = None if row.ddec_arcsec is None else row.ddec_arcsec / 60.0
        shift_am = None if row.shift_arcsec is None else row.shift_arcsec / 60.0
        if row.status == "stale":
            row_style = "background:#eeeeee;color:#777777"
            status_txt = "stale"
        elif row.status == "no_pointing":
            row_style = "background:#f5f5f5;color:#666"
            status_txt = "no pointing"
        else:
            row_style = ""
            status_txt = "ok"
        lines.append(
            f"<tr style='{row_style}'>"
            f"<td style='padding:2px 8px'>dr{row.beam}</td>"
            f"<td style='padding:2px 8px'>{status_txt}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.display_az_deg)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.display_el_deg)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.nominal_az_deg)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.nominal_el_deg)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>"
            f"{'yes' if row.uses_commanded else '—'}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(dra_am)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(ddec_am)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(shift_am)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_format_age_sec(row.age_sec)}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


class DewarpMapApp(param.Parameterized):
    """Param-driven live tip-tilt map view for one frequency / warp path."""

    warp_path = param.String(default="", doc="Path to tip-tilt .npz (empty → etcd resolve)")
    frequency_mhz = param.Integer(default=55, bounds=(1, 300), doc="App frequency MHz")
    poll_seconds = param.Number(default=3.0, bounds=(0.5, 60.0))
    stride = param.Integer(
        default=3,
        bounds=(1, 32),
        doc="Equal-area sample spacing multiplier (spacing = 2.5° × stride)",
    )
    central_ra_deg = param.Number(
        default=0.0,
        softbounds=(0.0, 360.0),
        doc="Mollweide central RA (deg); 0 puts the RA=0 meridian at centre",
    )
    max_pointing_age_sec = param.Selector(
        default=120.0,
        objects=list(_POINTING_AGE_SLIDER_OPTIONS.values()),
        label="Max pointing age sec",
        doc="Pointing older than this is marked stale; 'all' = never stale",
    )
    show_nominal_marks = param.Boolean(
        default=True,
        doc="When commanded≠nominal, also mark nominal az/el",
    )
    status = param.String(default="starting", doc="Status line for the UI")
    header = param.String(default="", doc="Markdown header with obstime / mtime")
    # Bumped on every successful map / beam reload so reactive panes re-render.
    revision = param.Integer(default=0, doc="Map revision counter")
    beam_table_html = param.String(default="", doc="HTML table of beam warp samples")

    def __init__(
        self,
        demo_pointings: Optional[Dict[int, BeamPointingRecord]] = None,
        **params,
    ):
        super().__init__(**params)
        self._tip: Optional[TipTiltMap] = None
        self._loaded_path: Optional[Path] = None
        self._loaded_mtime: Optional[float] = None
        self._loaded_obstime: Optional[float] = None
        self._demo_pointings = demo_pointings
        self._beam_rows: List[BeamMapRow] = []
        self._beam_fp: tuple = ()
        self._plot_obj = None
        self._plot_pipe = None
        self._plot_dmap = None
        self._plot_pane = None
        self._callback = None
        self._poll_scheduled = False
        self._pn = None
        self._hv = None
        self._layout = None
        self._ensure_extensions()
        self.reload(force=True)
        self.refresh_beams(force=True)
        self._compose_plot()

    def _ensure_extensions(self) -> None:
        import holoviews as hv
        import panel as pn

        self._pn = pn
        self._hv = hv
        hv.extension("bokeh", logo=False)
        pn.extension(sizing_mode="stretch_width")

    def resolve_path(self) -> Optional[Path]:
        """Return the NPZ path from ``warp_path`` or tip-tilt etcd for ``frequency_mhz``."""
        raw = (self.warp_path or "").strip()
        if raw:
            return Path(raw).expanduser()
        resolved = resolve_tiptilt_path(int(self.frequency_mhz))
        if resolved:
            return Path(resolved)
        return None

    def _resolved_central_ra(self) -> float:
        return float(np.mod(float(self.central_ra_deg), 360.0))

    def _effective_max_age_sec(self) -> float:
        """Return age limit; non-finite means unlimited (UI value ``all``)."""
        return float(self.max_pointing_age_sec)

    def _discover_age_limit_sec(self) -> float:
        """Finite age for recorder discovery APIs that reject ``inf``."""
        age = self._effective_max_age_sec()
        if not np.isfinite(age):
            return float(_POINTING_AGE_DISCOVER_FALLBACK_SEC)
        return age

    def _set_header(self, path: Optional[Path], mtime: float, obstime: float) -> None:
        name = path.name if path is not None else "?"
        self.header = (
            f"### Tip-tilt **{self.frequency_mhz} MHz** — `{name}`\n\n"
            f"- **map obstime:** {_format_unix(obstime)}\n"
            f"- **file mtime:** {_format_unix(mtime)}\n"
            f"- **revision:** {self.revision}\n"
            f"- **units:** arcmin; **1 BMAJ ≈ {_BMAJ_ARCMIN:g}′** "
            f"(typical PSF major-axis FWHM ≈ 0.20°)\n"
            f"- **arrow scale:** |shift| {_ARROW_REF_ARCMIN:g}′ → "
            f"{_ARROW_MAX_DISPLAY_DEG:g}° on Mollweide "
            f"(clip {_ARROW_MAX_ARCMIN:g}′)"
        )

    def reload(self, *, force: bool = False) -> bool:
        """
        Reload the map if the path/mtime/obstime changed (or ``force``).

        Returns True when the in-memory map was replaced.
        """
        path = self.resolve_path()
        if path is None:
            self.status = (
                f"No warp path: set --warp-path or publish tip-tilt for "
                f"{self.frequency_mhz} MHz"
            )
            return False
        if not path.is_file():
            self.status = f"Waiting for map file: {path}"
            return False
        try:
            st = path.stat()
            mtime = float(st.st_mtime)
            size = int(st.st_size)
        except OSError as exc:
            self.status = f"Cannot stat {path}: {exc}"
            return False
        if (
            not force
            and self._tip is not None
            and self._loaded_path == path
            and self._loaded_mtime is not None
            and abs(self._loaded_mtime - mtime) < 1e-6
        ):
            return False
        try:
            tip = load_tiptilt_npz(path)
        except Exception as exc:
            # Mid-replace races: keep previous map and retry next poll.
            LOG.warning("Failed to load tip-tilt map %s (size=%s): %s", path, size, exc)
            self.status = f"Load failed ({path.name}): {exc}"
            return False

        same_content = (
            not force
            and self._loaded_obstime is not None
            and abs(self._loaded_obstime - tip.obstime_unix) < 1e-9
            and self._loaded_path == path
        )
        self._tip = tip
        self._loaded_path = path
        self._loaded_mtime = mtime
        self._loaded_obstime = float(tip.obstime_unix)
        self._set_header(path, mtime, tip.obstime_unix)
        self.status = (
            f"{'Reload' if not same_content else 'Touch'} {path.name}  "
            f"grid={tip.az_deg.size}×{tip.el_deg.size}  "
            f"stride={self.stride}  obstime={_format_unix(tip.obstime_unix)}  "
            f"@ {_format_unix(time.time())}"
        )
        return True

    def refresh_beams(self, *, force: bool = False) -> bool:
        """Re-read active beam pointings; return True when the table changed."""
        try:
            rows = collect_beam_rows(
                frequency_mhz=int(self.frequency_mhz),
                tip=self._tip,
                max_age_sec=self._effective_max_age_sec(),
                demo_pointings=self._demo_pointings,
            )
        except Exception as exc:
            LOG.warning("beam refresh failed: %s", exc)
            rows = list(self._beam_rows)
        fp = rows_fingerprint(rows)
        if not force and fp == self._beam_fp:
            return False
        self._beam_rows = rows
        self._beam_fp = fp
        self.beam_table_html = _beam_table_html(rows)
        return True

    def poll(self) -> None:
        """Periodic callback: refresh map + beams; always tick status."""
        try:
            map_changed = self.reload(force=False)
            beams_changed = self.refresh_beams(force=False)
            if map_changed or beams_changed:
                self._compose_plot()
            path = self._loaded_path
            name = path.name if path is not None else "?"
            n_ok = sum(1 for r in self._beam_rows if r.status == "ok")
            n_stale = sum(1 for r in self._beam_rows if r.status == "stale")
            n_miss = sum(1 for r in self._beam_rows if r.status == "no_pointing")
            mode = "demo" if self._demo_pointings is not None else "etcd"
            self.status = (
                f"Watching {name}  beams[{mode}]="
                f"{n_ok} ok/{n_stale} stale/{n_miss} missing  "
                f"obstime={_format_unix(self._loaded_obstime or float('nan'))}  "
                f"poll@{_format_unix(time.time())}"
            )
        except Exception as exc:
            LOG.exception("poll failed")
            self.status = f"Poll error: {exc}"

    def _plot_payload(self) -> dict:
        path = self._loaded_path
        mtime = self._loaded_mtime or float("nan")
        obstime = (
            float(self._tip.obstime_unix)
            if self._tip is not None
            else float("nan")
        )
        cra = self._resolved_central_ra()
        title = (
            f"Tip-tilt {self.frequency_mhz} MHz | "
            f"obstime {_format_unix(obstime)} | "
            f"mtime {_format_unix(mtime)} | "
            f"Mollweide RA/Dec (RA←, center RA={cra:.1f}°)"
        )
        return {
            "tip": self._tip,
            "beam_rows": list(self._beam_rows),
            "stride": int(self.stride),
            "central_ra_deg": float(cra),
            "show_nominal_marks": bool(self.show_nominal_marks),
            "title": title,
        }

    def _render_payload(self, data: Optional[dict] = None):
        """Build overlay from a Pipe payload (DynamicMap; kwarg must be ``data``)."""
        import holoviews as hv

        payload = data
        if not payload or payload.get("tip") is None:
            empty = hv.Text(0.0, 0.0, "No map loaded yet")
            return (mollweide_outline() * empty).opts(**_moll_frame_opts("No map loaded"))
        return build_vectorfield(
            payload["tip"],
            stride=int(payload.get("stride", 1)),
            title=str(payload.get("title") or ""),
            central_ra_deg=float(payload.get("central_ra_deg", 0.0)),
            beam_rows=payload.get("beam_rows") or [],
            show_nominal_marks=bool(payload.get("show_nominal_marks", True)),
        )

    def _ensure_dynamic_plot(self) -> None:
        """Create a single DynamicMap+Pipe so refresh updates data, not axes."""
        if self._plot_dmap is not None:
            return
        import holoviews as hv
        from holoviews.streams import Pipe

        initial = self._plot_payload()
        self._plot_pipe = Pipe(data=initial)
        self._plot_dmap = hv.DynamicMap(self._render_payload, streams=[self._plot_pipe])
        self._plot_obj = self._plot_dmap

    def _compose_plot(self) -> None:
        """Push latest map/beams into the DynamicMap Pipe (preserves x/y limits)."""
        self._ensure_dynamic_plot()
        payload = self._plot_payload()
        assert self._plot_pipe is not None
        self._plot_pipe.send(payload)
        self._plot_obj = self._plot_dmap
        self.revision = int(self.revision) + 1

    @param.depends(
        "stride", "central_ra_deg", "show_nominal_marks", "max_pointing_age_sec",
        watch=True,
    )
    def _on_display_params(self) -> None:
        self.refresh_beams(force=True)
        self._compose_plot()
        age = self._effective_max_age_sec()
        age_txt = "all" if not np.isfinite(age) else f"{age:g}s"
        self.status = (
            f"Redraw stride={self.stride} central_ra={self._resolved_central_ra():.1f} "
            f"max_age={age_txt}"
        )

    @param.depends("status")
    def status_view(self):
        return f"**Status:** {self.status}"

    @param.depends("header")
    def header_view(self):
        return self.header or "_No map metadata yet_"

    @param.depends("beam_table_html")
    def beam_table_view(self):
        html = self.beam_table_html or "<em>No active beams</em>"
        return f"### Beams\n\n{html}"

    def _start_periodic_callback(self) -> None:
        """Schedule exactly one poll callback for this session via ``onload``.

        Starting synchronously while the Bokeh document is held (layout build)
        can register the same periodic-callback ID twice when the document is
        later unheld — hence the deferral to ``pn.state.onload``.
        """
        pn = self._pn
        if getattr(self, "_poll_scheduled", False):
            return
        self._poll_scheduled = True
        period_ms = int(max(0.5, float(self.poll_seconds)) * 1000)

        def _start():
            if self._callback is not None:
                return
            try:
                self._callback = pn.state.add_periodic_callback(
                    self.poll, period=period_ms, start=True
                )
            except ValueError as exc:
                # Session already has this callback (e.g. duplicate onload).
                LOG.warning("Poll callback not started: %s", exc)
                return
            LOG.info(
                "Started dewarp-map poll every %s ms (freq=%s)",
                period_ms,
                self.frequency_mhz,
            )

        pn.state.onload(_start)

    def panel(self):
        """Return (and cache) the Panel layout; schedule the poll callback."""
        pn = self._pn
        if self._layout is not None:
            return self._layout

        controls = pn.Param(
            self,
            parameters=[
                "frequency_mhz",
                "stride",
                "central_ra_deg",
                "max_pointing_age_sec",
                "show_nominal_marks",
                "poll_seconds",
            ],
            widgets={
                "frequency_mhz": {"disabled": True},
                "poll_seconds": {"disabled": True},
                "max_pointing_age_sec": pn.widgets.DiscreteSlider(
                    label="Max pointing age sec",
                    options=dict(_POINTING_AGE_SLIDER_OPTIONS),
                    value=120.0,
                ),
            },
            name="Dewarp map",
            width=280,
            sizing_mode="fixed",
        )
        demo_note = (
            " (demo pointings)"
            if self._demo_pointings is not None
            else " (live etcd pointings)"
        )
        self._ensure_dynamic_plot()
        self._plot_pane = pn.pane.HoloViews(
            self._plot_dmap,
            width=_MOLL_WIDTH,
            height=_MOLL_HEIGHT,
            sizing_mode="fixed",
            linked_axes=False,
        )
        plot_row = pn.Row(
            controls,
            self._plot_pane,
            sizing_mode="fixed",
        )
        meta_row = pn.Row(
            pn.panel(self.header_view),
            pn.panel(self.beam_table_view),
            sizing_mode="stretch_width",
        )
        self._layout = pn.Column(
            pn.pane.Markdown(
                f"# Live dewarp / tip-tilt map\n"
                f"Frequency **{self.frequency_mhz} MHz** — "
                f"path `{self.warp_path or '(from etcd)'}`{demo_note}\n\n"
                f"Celestial **Mollweide** centred on **RA=0** (RA increases left). "
                f"Samples use an **equal-area** RA/Dec lattice (uniform on Mollweide). "
                f"Arrow **length** encodes |shift| in **arcmin**; "
                f"**1 BMAJ ≈ {_BMAJ_ARCMIN:g}′** (PSF major-axis FWHM). "
                f"Arrow length: {_ARROW_REF_ARCMIN:g}′ → {_ARROW_MAX_DISPLAY_DEG:g}° "
                f"on Mollweide (clip {_ARROW_MAX_ARCMIN:g}′)."
            ),
            meta_row,
            pn.panel(self.status_view),
            plot_row,
            sizing_mode="stretch_width",
        )
        self._start_periodic_callback()
        return self._layout

    def servable(self, title: str = "Dewarp map"):
        """Mark the layout servable for ``panel serve``."""
        return self.panel().servable(title=title)


def create_app(
    *,
    warp_path: Optional[PathLike] = None,
    frequency_mhz: int = 55,
    poll_seconds: float = 3.0,
    stride: int = 3,
    central_ra_deg: float = 0.0,
    central_az_deg: Optional[float] = None,
    max_pointing_age_sec: float = 120.0,
    show_nominal_marks: bool = True,
    demo_pointings_path: Optional[PathLike] = None,
) -> DewarpMapApp:
    """Factory used by the serve script."""
    demo = None
    if demo_pointings_path is not None:
        demo = load_demo_pointings(demo_pointings_path)
    # central_az_deg kept as ignored alias for older call sites / CLI.
    _ = central_az_deg
    return DewarpMapApp(
        warp_path="" if warp_path is None else str(Path(warp_path)),
        frequency_mhz=int(frequency_mhz),
        poll_seconds=float(poll_seconds),
        stride=int(stride),
        central_ra_deg=float(central_ra_deg),
        max_pointing_age_sec=float(max_pointing_age_sec),
        show_nominal_marks=bool(show_nominal_marks),
        demo_pointings=demo,
    )

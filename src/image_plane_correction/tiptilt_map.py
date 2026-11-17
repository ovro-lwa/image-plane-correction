"""Convert calcflow pixel offsets to ``mnc.tiptilt`` named-array NPZ maps.

The tip-tilt consumer (:mod:`mnc.tiptilt`) expects a ``.npz`` (or dict-like
``.npy``) with:

- ``az_deg`` — ``(n_az,)``, strictly increasing AltAz azimuth (deg)
- ``el_deg`` — ``(n_el,)``, strictly increasing altitude (deg)
- ``dra_arcsec`` / ``ddec_arcsec`` — ``(n_el, n_az)`` FK5 true-angle offsets
- ``obstime_unix`` — scalar UTC unix seconds (AltAz epoch of the grid)

Pixel ``(dx, dy)`` offsets from :class:`~image_plane_correction.flow.Flow` are
sampled on a regular az/el grid using the active image WCS. At each grid cell
the sky offset from pixel ``(x, y)`` to ``(x + dx, y + dy)`` is stored as
``(dra, ddec)`` via :meth:`~astropy.coordinates.SkyCoord.spherical_offsets_to`.
That matches backwards :meth:`~image_plane_correction.flow.Flow.apply`: content
for reference pixel ``p`` is taken from ``p + offset``, so adding this sky
shift moves a catalog direction toward the apparent image location.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

PathLike = Union[str, os.PathLike]

_FREQUENCY_IN_NAME = re.compile(r"_(\d+)MHz", re.IGNORECASE)

# OVRO-LWA geodetic fallback if ``lwa_antpos`` is unavailable (matches mnc tests).
_OVRO_FALLBACK = dict(lat=37.23335, lon=-118.28167, height=1222.0)

# Power-beam subband names (MHz). Matches ``mnc.xengine_beamformer_control``.
DEFAULT_SUBBAND_FREQUENCIES_MHZ: Tuple[int, ...] = (
    13,
    18,
    23,
    27,
    32,
    36,
    41,
    46,
    50,
    55,
    59,
    64,
    69,
    73,
    78,
    82,
)


def default_location() -> EarthLocation:
    """Return the OVRO station ``EarthLocation``."""
    try:
        from lwa_antpos.station import ovro
        import astropy.units as u

        return EarthLocation.from_geocentric(*ovro.ecef, unit=u.m)
    except Exception:
        import astropy.units as u

        return EarthLocation(
            lat=_OVRO_FALLBACK["lat"] * u.deg,
            lon=_OVRO_FALLBACK["lon"] * u.deg,
            height=_OVRO_FALLBACK["height"] * u.m,
        )


#: Default directory for per-subband tip-tilt warp NPZ files.
DEFAULT_WARP_DIR = Path("/fast/claw/warps")

#: Max concurrent beamformer subbands the live-dewarp UI expects (OVRO-LWA).
MAX_WARP_SUBBANDS = 16


def frequency_mhz_from_filename(path: PathLike) -> int:
    """Return the positive integer between ``_`` and ``MHz`` in the basename."""
    found = _FREQUENCY_IN_NAME.findall(os.path.basename(os.fspath(path)))
    if len(found) != 1:
        raise ValueError(
            "filename must contain exactly one _<integer>MHz field "
            f"(got {os.path.basename(os.fspath(path))!r})"
        )
    frequency_mhz = int(found[0])
    if frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive integer")
    return frequency_mhz


def warp_npz_basename(frequency_mhz: int) -> str:
    """Return the conventional ``warp_<N>MHz.npz`` basename for a subband."""
    if type(frequency_mhz) is not int or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive int")
    return f"warp_{frequency_mhz}MHz.npz"


def warp_path_for_subband(
    frequency_mhz: int,
    *,
    warp_dir: PathLike = DEFAULT_WARP_DIR,
) -> Path:
    """Return ``warp_dir / warp_<N>MHz.npz`` for beamformer subband ``N``."""
    return Path(warp_dir).expanduser() / warp_npz_basename(frequency_mhz)


def discover_warp_subbands_mhz(
    warp_dir: PathLike,
    *,
    max_subbands: int = MAX_WARP_SUBBANDS,
) -> Tuple[int, ...]:
    """
    Discover subband MHz values from ``warp_<N>MHz.npz`` files in ``warp_dir``.

    Returns a sorted tuple of at most ``max_subbands`` frequencies (default 16).
    Missing / unreadable directories yield an empty tuple.
    """
    if type(max_subbands) is not int or max_subbands < 1:
        raise ValueError("max_subbands must be an int >= 1")
    root = Path(warp_dir).expanduser()
    if not root.is_dir():
        return ()
    freqs: set[int] = set()
    for path in root.glob("warp_*MHz.npz"):
        if not path.is_file():
            continue
        try:
            freqs.add(frequency_mhz_from_filename(path))
        except ValueError:
            continue
    return tuple(sorted(freqs)[: int(max_subbands)])


def subband_frequencies_mhz() -> Tuple[int, ...]:
    """
    Return OVRO-LWA tip-tilt / beamformer subband names in MHz.

    Prefers ``mnc.xengine_beamformer_control.SUBBAND_FREQUENCIES_MHZ`` when
    importable so publish stays aligned with live BeamTracker; otherwise uses
    :data:`DEFAULT_SUBBAND_FREQUENCIES_MHZ`.
    """
    try:
        from mnc.xengine_beamformer_control import SUBBAND_FREQUENCIES_MHZ
    except ImportError:
        return DEFAULT_SUBBAND_FREQUENCIES_MHZ
    return tuple(int(x) for x in SUBBAND_FREQUENCIES_MHZ)


def resolve_publish_frequencies_mhz(
    *,
    warp_path: PathLike,
    frequency_mhz: Optional[int] = None,
    publish_all_frequencies: bool = False,
    frequencies_mhz: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """
    Choose which tip-tilt catalog keys should point at ``warp_path``.

    Precedence: explicit ``frequencies_mhz`` → all subbands when
    ``publish_all_frequencies`` → single ``frequency_mhz`` → parse from
    ``warp_path`` basename.
    """
    if frequencies_mhz is not None:
        freqs = tuple(int(x) for x in frequencies_mhz)
        if not freqs:
            raise ValueError("frequencies_mhz must be non-empty")
        if any(f <= 0 for f in freqs):
            raise ValueError("frequencies_mhz values must be positive ints")
        return freqs
    if publish_all_frequencies:
        return subband_frequencies_mhz()
    if frequency_mhz is not None:
        freq = int(frequency_mhz)
        if freq <= 0:
            raise ValueError("frequency_mhz must be a positive int")
        return (freq,)
    return (frequency_mhz_from_filename(warp_path),)


def obstime_unix_from_header(header) -> float:
    """Parse UTC unix seconds from a FITS header (``DATE-OBS`` or ``MJD-OBS``)."""
    if header.get("DATE-OBS"):
        t = Time(header["DATE-OBS"], scale="utc")
    elif header.get("MJD-OBS") is not None:
        t = Time(header["MJD-OBS"], format="mjd", scale="utc")
    else:
        raise ValueError("header must provide DATE-OBS or MJD-OBS for tip-tilt obstime")
    return float(t.unix)


def _az_el_axes(
    *,
    az_deg: Optional[np.ndarray],
    el_deg: Optional[np.ndarray],
    az_step_deg: float,
    el_step_deg: float,
    el_min_deg: float,
    el_max_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if az_deg is None:
        if not (az_step_deg > 0.0):
            raise ValueError("az_step_deg must be positive")
        az_deg = np.arange(0.0, 360.0 + 0.5 * az_step_deg, az_step_deg, dtype=np.float64)
        if az_deg.size < 2 or az_deg[-1] < 360.0 - 1e-9:
            az_deg = np.unique(np.append(az_deg, 360.0))
    else:
        az_deg = np.asarray(az_deg, dtype=np.float64)
    if el_deg is None:
        if not (el_step_deg > 0.0):
            raise ValueError("el_step_deg must be positive")
        el_deg = np.arange(
            float(el_min_deg),
            float(el_max_deg) + 0.5 * el_step_deg,
            el_step_deg,
            dtype=np.float64,
        )
        if el_deg.size == 0 or el_deg[-1] < float(el_max_deg) - 1e-9:
            el_deg = np.unique(np.append(el_deg, float(el_max_deg)))
    else:
        el_deg = np.asarray(el_deg, dtype=np.float64)

    if az_deg.ndim != 1 or el_deg.ndim != 1:
        raise ValueError("az_deg and el_deg must be 1-d")
    if az_deg.size < 2 or el_deg.size < 2:
        raise ValueError("az_deg and el_deg must each have at least two points")
    if not np.all(np.diff(az_deg) > 0.0) or not np.all(np.diff(el_deg) > 0.0):
        raise ValueError("az_deg and el_deg must be strictly increasing")
    return az_deg, el_deg


def pixel_flow_to_tiptilt_arrays(
    offsets: np.ndarray,
    imwcs: WCS,
    obstime_unix: float,
    *,
    location: Optional[EarthLocation] = None,
    az_deg: Optional[np.ndarray] = None,
    el_deg: Optional[np.ndarray] = None,
    az_step_deg: float = 5.0,
    el_step_deg: float = 2.5,
    el_min_deg: float = 0.0,
    el_max_deg: float = 90.0,
) -> dict:
    """
    Build tip-tilt named arrays from a pixel ``(H, W, 2)`` flow field.

    Parameters
    ----------
    offsets :
        Flow offsets with ``[..., 0] = dx`` (column) and ``[..., 1] = dy`` (row).
    imwcs :
        Celestial WCS of the offset grid (same shape as ``offsets`` spatial axes).
    obstime_unix :
        UTC unix seconds used as the AltAz epoch (and stored in the map).

    Returns
    -------
    dict
        ``az_deg``, ``el_deg``, ``dra_arcsec``, ``ddec_arcsec``, ``obstime_unix``.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError(f"offsets must have shape (H, W, 2), got {offsets.shape}")
    n_y, n_x = int(offsets.shape[0]), int(offsets.shape[1])
    if n_y < 2 or n_x < 2:
        raise ValueError("offsets spatial axes must be at least 2 pixels")

    az_axis, el_axis = _az_el_axes(
        az_deg=az_deg,
        el_deg=el_deg,
        az_step_deg=az_step_deg,
        el_step_deg=el_step_deg,
        el_min_deg=el_min_deg,
        el_max_deg=el_max_deg,
    )
    if location is None:
        location = default_location()

    obstime = Time(float(obstime_unix), format="unix", scale="utc")
    az_grid, el_grid = np.meshgrid(az_axis, el_axis)  # (n_el, n_az)
    aa = SkyCoord(
        az=az_grid,
        alt=el_grid,
        unit="deg",
        frame=AltAz(obstime=obstime, location=location),
    )
    # Pixelize grid directions in FK5 (LWA snapshot WCS is typically RA/Dec SIN).
    sky = aa.transform_to("fk5")
    x, y = imwcs.world_to_pixel(sky)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Bilinear sample of (dx, dy); outside → NaN.
    coords = np.vstack([y.ravel(), x.ravel()])
    dx = map_coordinates(
        offsets[:, :, 0], coords, order=1, mode="constant", cval=np.nan, prefilter=False
    ).reshape(el_axis.size, az_axis.size)
    dy = map_coordinates(
        offsets[:, :, 1], coords, order=1, mode="constant", cval=np.nan, prefilter=False
    ).reshape(el_axis.size, az_axis.size)

    # Invalidate samples that land outside the pixel grid before offsetting.
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= (n_x - 1))
        & (y >= 0.0)
        & (y <= (n_y - 1))
    )
    dx = np.where(inside, dx, np.nan)
    dy = np.where(inside, dy, np.nan)

    x1 = x + dx
    y1 = y + dy
    still_ok = (
        inside
        & np.isfinite(dx)
        & np.isfinite(dy)
        & np.isfinite(x1)
        & np.isfinite(y1)
        & (x1 >= 0.0)
        & (x1 <= (n_x - 1))
        & (y1 >= 0.0)
        & (y1 <= (n_y - 1))
    )

    dra = np.full(az_grid.shape, np.nan, dtype=np.float64)
    ddec = np.full(az_grid.shape, np.nan, dtype=np.float64)

    if np.any(still_ok):
        sky0 = imwcs.pixel_to_world(x[still_ok], y[still_ok])
        sky1 = imwcs.pixel_to_world(x1[still_ok], y1[still_ok])
        lon, lat = sky0.spherical_offsets_to(sky1)
        dra_v = np.asarray(lon.to_value("arcsec"), dtype=np.float64)
        ddec_v = np.asarray(lat.to_value("arcsec"), dtype=np.float64)
        dra[still_ok] = dra_v
        ddec[still_ok] = ddec_v

    return {
        "az_deg": az_axis.astype(np.float64, copy=False),
        "el_deg": el_axis.astype(np.float64, copy=False),
        "dra_arcsec": dra,
        "ddec_arcsec": ddec,
        "obstime_unix": np.asarray(float(obstime_unix), dtype=np.float64),
    }


def write_tiptilt_npz(path: PathLike, arrays: dict) -> Path:
    """Atomically write a tip-tilt named-array ``.npz`` file."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"tip-tilt map path must end with .npz, got {path.name!r}")
    required = ("az_deg", "el_deg", "dra_arcsec", "ddec_arcsec", "obstime_unix")
    missing = [k for k in required if k not in arrays]
    if missing:
        raise ValueError(f"arrays missing keys: {missing}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Write via file object so NumPy does not append an extra ``.npz`` suffix.
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            az_deg=np.asarray(arrays["az_deg"], dtype=np.float64),
            el_deg=np.asarray(arrays["el_deg"], dtype=np.float64),
            dra_arcsec=np.asarray(arrays["dra_arcsec"], dtype=np.float64),
            ddec_arcsec=np.asarray(arrays["ddec_arcsec"], dtype=np.float64),
            obstime_unix=np.asarray(arrays["obstime_unix"], dtype=np.float64),
        )
    os.replace(tmp, path)
    return path


def pixel_flow_to_tiptilt_npz(
    path: PathLike,
    offsets: np.ndarray,
    imwcs: WCS,
    obstime_unix: float,
    **kwargs,
) -> Path:
    """Convert pixel offsets to tip-tilt arrays and write ``path`` as ``.npz``."""
    arrays = pixel_flow_to_tiptilt_arrays(offsets, imwcs, obstime_unix, **kwargs)
    return write_tiptilt_npz(path, arrays)

"""Load tip-tilt NPZ maps and beam pointing monitors for live UI consumers.

Tip-tilt maps match :mod:`mnc.tiptilt` named-array ``.npz`` files. Beam pointing
JSON matches ``mnc.xengine_beamformer_control.publish_commanded_pointing``::

    {"timestamp": <unix>, "unit": "deg",
     "value": {
         "nominal": {"az_deg": ..., "el_deg": ...},
         "commanded_by_frequency_mhz": {
             "<mhz>": {"az_deg": ..., "el_deg": ...}, ...
         }
     }}

``commanded_by_frequency_mhz`` is optional. Display pointing for app frequency
``F`` prefers ``commanded_by_frequency_mhz[str(F)]``, else ``nominal``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

PathLike = Union[str, os.PathLike]

#: Same template as ``mnc.xengine_beamformer_control.POINTING_ETCD_KEY_TEMPLATE``.
POINTING_ETCD_KEY_TEMPLATE = "/mon/beamformer/%d/pointing"

_DR_POWER = re.compile(r"^dr(\d+)$")
_REQUIRED_MAP_KEYS = ("az_deg", "el_deg", "dra_arcsec", "ddec_arcsec", "obstime_unix")


@dataclass(frozen=True)
class BeamPointingRecord:
    """Parsed ``/mon/beamformer/{N}/pointing`` monitor value."""

    beam: int
    timestamp: float
    nominal_az_deg: float
    nominal_el_deg: float
    commanded_by_frequency_mhz: Dict[int, Tuple[float, float]] = field(
        default_factory=dict
    )
    unit: str = "deg"

    @property
    def nominal(self) -> Tuple[float, float]:
        return (self.nominal_az_deg, self.nominal_el_deg)


@dataclass
class TipTiltMap:
    """In-memory tip-tilt named-array map with az/el interpolators."""

    path: Path
    az_deg: np.ndarray
    el_deg: np.ndarray
    dra_arcsec: np.ndarray
    ddec_arcsec: np.ndarray
    obstime_unix: float
    mtime: float
    _interp_dra: Any = field(repr=False, compare=False)
    _interp_ddec: Any = field(repr=False, compare=False)

    def sample(self, az_deg: float, el_deg: float) -> Tuple[float, float, bool]:
        """
        Return ``(dra_arcsec, ddec_arcsec, outside)``.

        ``outside`` is True when the query is off the az/el grid; samples are
        then NaN (caller may treat as zero shift for display).
        """
        az = float(az_deg)
        el = float(el_deg)
        if (
            az < float(self.az_deg[0])
            or az > float(self.az_deg[-1])
            or el < float(self.el_deg[0])
            or el > float(self.el_deg[-1])
        ):
            return float("nan"), float("nan"), True
        dra = float(self._interp_dra((el, az)))
        ddec = float(self._interp_ddec((el, az)))
        if not (np.isfinite(dra) and np.isfinite(ddec)):
            return dra, ddec, True
        return dra, ddec, False

    def magnitude_arcsec(self, az_deg: float, el_deg: float) -> float:
        dra, ddec, outside = self.sample(az_deg, el_deg)
        if outside or not (np.isfinite(dra) and np.isfinite(ddec)):
            return float("nan")
        return float(np.hypot(dra, ddec))


def load_tiptilt_npz(path: PathLike) -> TipTiltMap:
    """Load a tip-tilt named-array ``.npz`` from disk."""
    path = Path(path)
    with np.load(path) as data:
        missing = [k for k in _REQUIRED_MAP_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        az_deg = np.array(data["az_deg"], dtype=np.float64, copy=True)
        el_deg = np.array(data["el_deg"], dtype=np.float64, copy=True)
        dra = np.array(data["dra_arcsec"], dtype=np.float64, copy=True)
        ddec = np.array(data["ddec_arcsec"], dtype=np.float64, copy=True)
        obstime = float(np.asarray(data["obstime_unix"], dtype=np.float64))

    if az_deg.ndim != 1 or el_deg.ndim != 1:
        raise ValueError("az_deg and el_deg must be 1-d")
    if az_deg.size < 2 or el_deg.size < 2:
        raise ValueError("az_deg and el_deg must each have at least two points")
    if dra.shape != (el_deg.size, az_deg.size) or ddec.shape != dra.shape:
        raise ValueError("dra_arcsec/ddec_arcsec must have shape (n_el, n_az)")
    if not np.all(np.diff(az_deg) > 0.0) or not np.all(np.diff(el_deg) > 0.0):
        raise ValueError("az_deg and el_deg must be strictly increasing")

    interp_dra = RegularGridInterpolator(
        (el_deg, az_deg), dra, bounds_error=False, fill_value=np.nan
    )
    interp_ddec = RegularGridInterpolator(
        (el_deg, az_deg), ddec, bounds_error=False, fill_value=np.nan
    )
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = float("nan")
    return TipTiltMap(
        path=path,
        az_deg=az_deg,
        el_deg=el_deg,
        dra_arcsec=dra,
        ddec_arcsec=ddec,
        obstime_unix=obstime,
        mtime=mtime,
        _interp_dra=interp_dra,
        _interp_ddec=interp_ddec,
    )


def resolve_tiptilt_path(frequency_mhz: int, client=None) -> Optional[str]:
    """Return published tip-tilt map path for ``frequency_mhz``, or ``None``."""
    if type(frequency_mhz) is not int or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive int")
    try:
        from mnc.tiptilt import get_tiptilt_path
    except Exception:
        return None
    try:
        return get_tiptilt_path(frequency_mhz, client=client)
    except Exception:
        return None


def _az_el_pair(obj: Any, label: str) -> Tuple[float, float]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"{label} must be an object with az_deg and el_deg")
    try:
        az = float(obj["az_deg"])
        el = float(obj["el_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must provide finite az_deg and el_deg") from exc
    if not (np.isfinite(az) and np.isfinite(el)):
        raise ValueError(f"{label} az_deg/el_deg must be finite")
    return az, el


def parse_beam_pointing_payload(
    payload: Union[str, bytes, Mapping[str, Any]],
    *,
    beam: int,
) -> BeamPointingRecord:
    """Parse MCS JSON (or already-decoded dict) into :class:`BeamPointingRecord`."""
    if type(beam) is not int or beam <= 0:
        raise ValueError("beam must be a positive int")
    if isinstance(payload, bytes):
        try:
            payload = payload.decode()
        except UnicodeError as exc:
            raise ValueError("pointing payload is not valid UTF-8") from exc
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("pointing payload is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("pointing payload must be a JSON object")

    try:
        timestamp = float(payload["timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pointing payload missing finite timestamp") from exc
    if not np.isfinite(timestamp):
        raise ValueError("pointing payload missing finite timestamp")

    unit = payload.get("unit", "deg")
    if unit is not None and unit != "deg":
        raise ValueError(f"unsupported pointing unit {unit!r}; expected 'deg'")

    value = payload.get("value")
    if not isinstance(value, Mapping):
        raise ValueError("pointing value must be an object")
    if "nominal" not in value:
        raise ValueError("pointing value must include nominal {az_deg, el_deg}")
    nom_az, nom_el = _az_el_pair(value["nominal"], "nominal")

    commanded: Dict[int, Tuple[float, float]] = {}
    raw_cmd = value.get("commanded_by_frequency_mhz")
    if raw_cmd is not None:
        if not isinstance(raw_cmd, Mapping):
            raise ValueError("commanded_by_frequency_mhz must be an object")
        for key, direction in raw_cmd.items():
            try:
                freq = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"commanded_by_frequency_mhz key must be integer MHz, got {key!r}"
                ) from exc
            if freq <= 0 or str(freq) != str(key):
                raise ValueError(
                    f"commanded_by_frequency_mhz key must be integer MHz, got {key!r}"
                )
            commanded[freq] = _az_el_pair(direction, f"commanded[{freq}]")

    return BeamPointingRecord(
        beam=beam,
        timestamp=timestamp,
        nominal_az_deg=nom_az,
        nominal_el_deg=nom_el,
        commanded_by_frequency_mhz=commanded,
        unit="deg",
    )


def display_az_el(
    record: BeamPointingRecord, frequency_mhz: int
) -> Tuple[float, float]:
    """
    Az/el used for map marks at app frequency ``frequency_mhz``.

    Prefers ``commanded_by_frequency_mhz[frequency_mhz]`` when present;
    otherwise returns ``nominal``.
    """
    if type(frequency_mhz) is not int or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be a positive int")
    commanded = record.commanded_by_frequency_mhz.get(frequency_mhz)
    if commanded is not None:
        return commanded
    return record.nominal


def pointing_age_sec(record: BeamPointingRecord, now: Optional[float] = None) -> float:
    """Seconds since ``record.timestamp`` (``now - timestamp``)."""
    if now is None:
        now = time.time()
    return float(now) - float(record.timestamp)


def is_pointing_stale(
    record: BeamPointingRecord,
    *,
    max_age_sec: float = 120.0,
    now: Optional[float] = None,
) -> bool:
    """True when pointing age exceeds ``max_age_sec``.

    Non-finite ``max_age_sec`` (e.g. ``inf`` / UI ``all``) means never stale.
    """
    age_limit = float(max_age_sec)
    if not np.isfinite(age_limit):
        return False
    if not (age_limit > 0.0):
        raise ValueError("max_age_sec must be positive or non-finite (all)")
    return pointing_age_sec(record, now=now) > age_limit


def read_beam_pointing(
    beam: int,
    *,
    client=None,
    key_template: str = POINTING_ETCD_KEY_TEMPLATE,
) -> Optional[BeamPointingRecord]:
    """
    Read and parse ``/mon/beamformer/{beam}/pointing``.

    Returns ``None`` when the key is missing, etcd is unreachable, or the
    payload is unusable. ``client`` is an etcd-like object with ``get`` (and
    optional ``close``); the default connects via ``mnc`` / ``etcd3``.
    """
    if type(beam) is not int or beam <= 0:
        raise ValueError("beam must be a positive int")
    key = key_template % int(beam)
    own = False
    etcd = client
    try:
        if etcd is None:
            import etcd3
            from mnc.common import ETCD_HOST, ETCD_PORT

            etcd = etcd3.client(host=ETCD_HOST, port=ETCD_PORT)
            own = True
        got = etcd.get(key)
        value = got[0] if isinstance(got, tuple) else got
        if value is None:
            return None
        return parse_beam_pointing_payload(value, beam=beam)
    except Exception:
        return None
    finally:
        if own and etcd is not None:
            try:
                etcd.close()
            except Exception:
                pass


def beam_number_from_recorder_id(recorder_id: str) -> Optional[int]:
    """Map MCS id ``drN`` → ``N``; return ``None`` for non-power beam ids."""
    if not isinstance(recorder_id, str):
        return None
    matched = _DR_POWER.match(recorder_id.strip())
    if matched is None:
        return None
    return int(matched.group(1))


def list_active_beam_numbers(
    *,
    age_limit: float = 120.0,
    discover_recorders=None,
) -> list:
    """
    Active power-beam numbers from recorder discovery.

    By default calls ``mnc.ezdr.discover_recorders(type='power')``. Pass
    ``discover_recorders`` in tests to inject a fake callable.
    """
    if discover_recorders is None:
        from mnc.ezdr import discover_recorders as _discover

        discover_recorders = _discover
    found = discover_recorders(type="power", age_limit=age_limit)
    beams = []
    seen = set()
    for recorder_id in found:
        beam = beam_number_from_recorder_id(recorder_id)
        if beam is None or beam in seen:
            continue
        seen.add(beam)
        beams.append(beam)
    return sorted(beams)


_POINTING_KEY = re.compile(r"^/mon/beamformer/(\d+)/pointing$")


def list_beam_numbers_from_pointing_keys(*, client=None) -> list:
    """
    Beam numbers that have a ``/mon/beamformer/{N}/pointing`` etcd key.

    Fallback when ``discover_recorders`` fails (e.g. etcd message-size limits).
    ``client`` is an etcd-like object with ``get_prefix`` (and optional ``close``).
    """
    own = False
    etcd = client
    try:
        if etcd is None:
            import etcd3
            from mnc.common import ETCD_HOST, ETCD_PORT

            etcd = etcd3.client(host=ETCD_HOST, port=ETCD_PORT)
            own = True
        beams = []
        seen = set()
        for _value, meta in etcd.get_prefix("/mon/beamformer/"):
            key = meta.key.decode() if hasattr(meta.key, "decode") else str(meta.key)
            matched = _POINTING_KEY.match(key)
            if matched is None:
                continue
            beam = int(matched.group(1))
            if beam in seen:
                continue
            seen.add(beam)
            beams.append(beam)
        return sorted(beams)
    except Exception:
        return []
    finally:
        if own and etcd is not None:
            try:
                etcd.close()
            except Exception:
                pass

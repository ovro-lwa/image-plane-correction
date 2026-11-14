"""Beam pointing rows and zenithal equal-area overlays for the dewarp map app."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from image_plane_correction.live_map import (
    BeamPointingRecord,
    TipTiltMap,
    display_az_el,
    is_pointing_stale,
    list_active_beam_numbers,
    parse_beam_pointing_payload,
    pointing_age_sec,
    read_beam_pointing,
)

PathLike = Union[str, Path]


@dataclass
class BeamMapRow:
    """One beam's display state for the map table / overlays."""

    beam: int
    status: str  # ok | stale | no_pointing
    display_az_deg: Optional[float]
    display_el_deg: Optional[float]
    nominal_az_deg: Optional[float]
    nominal_el_deg: Optional[float]
    commanded_az_deg: Optional[float]
    commanded_el_deg: Optional[float]
    uses_commanded: bool
    dra_arcsec: Optional[float]
    ddec_arcsec: Optional[float]
    shift_arcsec: Optional[float]
    age_sec: Optional[float]
    outside_map: bool = False

    def fingerprint(self) -> tuple:
        return (
            self.beam,
            self.status,
            self.display_az_deg,
            self.display_el_deg,
            self.nominal_az_deg,
            self.nominal_el_deg,
            self.commanded_az_deg,
            self.commanded_el_deg,
            self.uses_commanded,
            self.dra_arcsec,
            self.ddec_arcsec,
            self.shift_arcsec,
            None if self.age_sec is None else round(float(self.age_sec), 1),
            self.outside_map,
        )


def load_demo_pointings(path: PathLike) -> Dict[int, BeamPointingRecord]:
    """
    Load demo ``/mon/beamformer/{N}/pointing`` payloads from JSON.

    Accepted shapes::

        {"1": {timestamp, value: {nominal: ...}}, "3": {...}}
        {"beams": {"1": {...}, "3": {...}}}
        [{"beam": 1, "timestamp": ..., "value": {...}}, ...]
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    items: Dict[int, Any] = {}
    if isinstance(raw, Mapping) and "beams" in raw and isinstance(raw["beams"], Mapping):
        raw = raw["beams"]
    if isinstance(raw, Mapping):
        for key, payload in raw.items():
            if isinstance(payload, Mapping) and "beam" in payload:
                beam = int(payload["beam"])
            else:
                beam = int(key)
            items[beam] = payload
    elif isinstance(raw, list):
        for payload in raw:
            if not isinstance(payload, Mapping) or "beam" not in payload:
                raise ValueError("demo pointing list entries need a beam field")
            items[int(payload["beam"])] = payload
    else:
        raise ValueError("demo pointings JSON must be an object or list")

    out: Dict[int, BeamPointingRecord] = {}
    for beam, payload in items.items():
        # Allow omitting timestamp in fixtures → "now".
        if isinstance(payload, Mapping) and "timestamp" not in payload:
            payload = dict(payload)
            payload["timestamp"] = time.time()
        out[int(beam)] = parse_beam_pointing_payload(payload, beam=int(beam))
    return out


def beam_row_from_record(
    record: BeamPointingRecord,
    *,
    frequency_mhz: int,
    tip: Optional[TipTiltMap],
    max_age_sec: float,
    now: Optional[float] = None,
) -> BeamMapRow:
    """Build a :class:`BeamMapRow` from a parsed pointing record."""
    if now is None:
        now = time.time()
    age = pointing_age_sec(record, now=now)
    stale = is_pointing_stale(record, max_age_sec=max_age_sec, now=now)
    disp_az, disp_el = display_az_el(record, frequency_mhz)
    commanded = record.commanded_by_frequency_mhz.get(frequency_mhz)
    uses_commanded = commanded is not None
    cmd_az = float(commanded[0]) if commanded is not None else None
    cmd_el = float(commanded[1]) if commanded is not None else None

    dra = ddec = shift = None
    outside = False
    if tip is not None:
        dra_v, ddec_v, outside = tip.sample(disp_az, disp_el)
        if np.isfinite(dra_v) and np.isfinite(ddec_v):
            dra, ddec = float(dra_v), float(ddec_v)
            shift = float(np.hypot(dra, ddec))
        else:
            outside = True

    return BeamMapRow(
        beam=record.beam,
        status="stale" if stale else "ok",
        display_az_deg=float(disp_az),
        display_el_deg=float(disp_el),
        nominal_az_deg=float(record.nominal_az_deg),
        nominal_el_deg=float(record.nominal_el_deg),
        commanded_az_deg=cmd_az,
        commanded_el_deg=cmd_el,
        uses_commanded=uses_commanded,
        dra_arcsec=dra,
        ddec_arcsec=ddec,
        shift_arcsec=shift,
        age_sec=float(age),
        outside_map=outside,
    )


def beam_row_missing(beam: int) -> BeamMapRow:
    return BeamMapRow(
        beam=beam,
        status="no_pointing",
        display_az_deg=None,
        display_el_deg=None,
        nominal_az_deg=None,
        nominal_el_deg=None,
        commanded_az_deg=None,
        commanded_el_deg=None,
        uses_commanded=False,
        dra_arcsec=None,
        ddec_arcsec=None,
        shift_arcsec=None,
        age_sec=None,
        outside_map=False,
    )


def collect_beam_rows(
    *,
    frequency_mhz: int,
    tip: Optional[TipTiltMap],
    max_age_sec: float = 120.0,
    demo_pointings: Optional[Mapping[int, BeamPointingRecord]] = None,
    demo_beam_numbers: Optional[Sequence[int]] = None,
    discover_recorders=None,
    read_pointing=None,
    now: Optional[float] = None,
) -> List[BeamMapRow]:
    """
    Collect beam rows for the UI.

    Demo mode (``demo_pointings`` set): beams default to the demo keys (or
    ``demo_beam_numbers``). Live mode: discover ``drN`` and read etcd keys.
    """
    if now is None:
        now = time.time()
    if read_pointing is None:
        read_pointing = read_beam_pointing

    if demo_pointings is not None:
        if demo_beam_numbers is None:
            beams = sorted(demo_pointings.keys())
        else:
            beams = [int(b) for b in demo_beam_numbers]
        rows = []
        for beam in beams:
            record = demo_pointings.get(beam)
            if record is None:
                rows.append(beam_row_missing(beam))
            else:
                rows.append(
                    beam_row_from_record(
                        record,
                        frequency_mhz=frequency_mhz,
                        tip=tip,
                        max_age_sec=max_age_sec,
                        now=now,
                    )
                )
        return rows

    try:
        beams = list_active_beam_numbers(
            age_limit=max_age_sec, discover_recorders=discover_recorders
        )
    except Exception:
        beams = []

    rows = []
    for beam in beams:
        try:
            record = read_pointing(beam)
        except Exception:
            record = None
        if record is None:
            rows.append(beam_row_missing(beam))
        else:
            rows.append(
                beam_row_from_record(
                    record,
                    frequency_mhz=frequency_mhz,
                    tip=tip,
                    max_age_sec=max_age_sec,
                    now=now,
                )
            )
    return rows


def rows_fingerprint(rows: Sequence[BeamMapRow]) -> tuple:
    return tuple(r.fingerprint() for r in rows)


def rows_as_dicts(rows: Sequence[BeamMapRow]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in rows]


def build_beam_overlay(
    rows: Sequence[BeamMapRow],
    *,
    central_az_deg: float = 0.0,
    show_nominal_when_commanded: bool = True,
):
    """
    HoloViews Points (+ labels) for beams with a display az/el.

    Fresh commanded/nominal marks are teal; stale marks are gray. When
    commanded differs from nominal, a smaller orange nominal mark is optional.
    """
    import holoviews as hv

    hv.extension("bokeh", logo=False)

    # Late import avoids a circular dependency with dewarp_map_app.
    from image_plane_correction.viz.dewarp_map_app import azel_to_zea

    ok_az, ok_el, ok_label = [], [], []
    stale_az, stale_el, stale_label = [], [], []
    nom_az, nom_el, nom_label = [], [], []

    for row in rows:
        if row.display_az_deg is None or row.display_el_deg is None:
            continue
        if row.status == "ok":
            ok_az.append(row.display_az_deg)
            ok_el.append(row.display_el_deg)
            ok_label.append(f"dr{row.beam}")
            if (
                show_nominal_when_commanded
                and row.uses_commanded
                and row.nominal_az_deg is not None
                and row.nominal_el_deg is not None
                and (
                    abs(row.nominal_az_deg - row.display_az_deg) > 1e-4
                    or abs(row.nominal_el_deg - row.display_el_deg) > 1e-4
                )
            ):
                nom_az.append(row.nominal_az_deg)
                nom_el.append(row.nominal_el_deg)
                nom_label.append(f"dr{row.beam} nom")
        elif row.status == "stale":
            stale_az.append(row.display_az_deg)
            stale_el.append(row.display_el_deg)
            stale_label.append(f"dr{row.beam}?")

    layers = []

    def _points(az, el, labels, *, color, size, marker="circle"):
        if not az:
            return None
        x, y = azel_to_zea(
            np.asarray(az, dtype=np.float64),
            np.asarray(el, dtype=np.float64),
            central_az_deg=central_az_deg,
        )
        pts = hv.Points(
            {
                "zea_x": x,
                "zea_y": y,
                "label": labels,
                "az_deg": az,
                "el_deg": el,
            },
            kdims=["zea_x", "zea_y"],
            vdims=["label", "az_deg", "el_deg"],
        ).opts(
            color=color,
            size=size,
            marker=marker,
            tools=["hover"],
            show_legend=False,
        )
        labels_el = hv.Labels(
            {"zea_x": x, "zea_y": y, "label": labels},
            kdims=["zea_x", "zea_y"],
            vdims=["label"],
        ).opts(text_color=color, text_font_size="9pt", yoffset=0.06)
        return pts * labels_el

    ok = _points(ok_az, ok_el, ok_label, color="#00a3a3", size=12)
    if ok is not None:
        layers.append(ok)
    stale = _points(stale_az, stale_el, stale_label, color="#888888", size=10)
    if stale is not None:
        layers.append(stale)
    nom = _points(
        nom_az, nom_el, nom_label, color="#e67e22", size=7, marker="triangle"
    )
    if nom is not None:
        layers.append(nom)

    if not layers:
        return hv.Points(
            ([], []), kdims=["zea_x", "zea_y"]
        ).opts(alpha=0.0, show_legend=False)
    overlay = layers[0]
    for layer in layers[1:]:
        overlay = overlay * layer
    return overlay

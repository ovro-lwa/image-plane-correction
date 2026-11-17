"""Beam pointing rows and celestial Mollweide overlays for the dewarp map app."""

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
    list_active_beam_numbers,
    list_beam_numbers_from_pointing_keys,
    parse_beam_pointing_payload,
    read_beam_pointing,
)

PathLike = Union[str, Path]

# Shared by tip-tilt arrow samples and beam marks.
# Numeral.js formats (Bokeh's default) — not printf "%.1f", which Holoviews
# rewrites to @{field}{%.1f} while formatter keys stay @field and then fall
# through to numeral as a broken percent format.
# Units live in the label so unicode degree/prime suffixes are not required.
_HOVER_TOOLTIPS = [
    ("beam", "@label"),
    ("az (deg)", "@az_deg{0.0}"),
    ("el (deg)", "@el_deg{0.0}"),
    ("|shift| (arcmin)", "@shift_arcmin{0.00}"),
    ("angle (deg)", "@angle_deg{0.0}"),
]


def _hover_opts() -> dict:
    """Opts shared by every glyph that should show tip-tilt hover fields."""
    return {
        "tools": ["hover"],
        "default_tools": [],
        "hover_tooltips": list(_HOVER_TOOLTIPS),
        "show_legend": False,
    }


def _strip_text_from_hover(plot, _element) -> None:
    """
    Restrict HoverTool to glyphs that carry tip-tilt hover columns.

    HoloViews Overlay/DynamicMap updates append *every* new glyph_renderer
    (including Labels/Text) to the shared HoverTool. Text CDS only has
    ``label``/x/y, so hovering beam name labels — especially small orange
    nominal triangles — shows ``???`` for az/el/shift/angle.
    """
    from bokeh.models import HoverTool

    fig = plot.state
    good = [
        r
        for r in fig.renderers
        if getattr(getattr(r, "data_source", None), "data", None) is not None
        and "az_deg" in r.data_source.data
        and "el_deg" in r.data_source.data
        and "shift_arcmin" in r.data_source.data
        and "angle_deg" in r.data_source.data
    ]
    for tool in fig.select(type=HoverTool):
        tool.renderers = list(good)
        tool.tooltips = list(_HOVER_TOOLTIPS)


# Tip-tilt correction status shown in the beam table / overlays.
STATUS_APPLIED = "applied"
STATUS_NOT_APPLIED = "not applied"
STATUS_STALE = "stale"


def _warp_correction_age_sec(
    tip: Optional[TipTiltMap], *, now: float
) -> Optional[float]:
    """Age of the loaded tip-tilt warp correction (obstime, else file mtime)."""
    if tip is None:
        return None
    epoch = float(tip.obstime_unix)
    if not np.isfinite(epoch):
        epoch = float(tip.mtime)
    if not np.isfinite(epoch):
        return None
    return max(0.0, float(now) - epoch)


def _correction_status(
    *,
    tip: Optional[TipTiltMap],
    uses_commanded: bool,
    warp_age_sec: Optional[float],
    max_age_sec: float,
) -> str:
    """
    Classify tip-tilt correction state for a beam.

    ``stale`` — warp map older than ``max_age_sec`` (when the limit is finite).
    ``applied`` — fresh warp and beamformer has commanded tip-tilt for the subband.
    ``not applied`` — no commanded tip-tilt (or no warp map yet).
    """
    age_limit = float(max_age_sec)
    if tip is not None and np.isfinite(age_limit):
        if age_limit <= 0.0:
            raise ValueError("max_age_sec must be positive or non-finite (all)")
        if warp_age_sec is not None and warp_age_sec > age_limit:
            return STATUS_STALE
    if uses_commanded and tip is not None:
        return STATUS_APPLIED
    return STATUS_NOT_APPLIED


@dataclass
class BeamMapRow:
    """One beam's display state for the map table / overlays."""

    beam: int
    status: str  # applied | not applied | stale
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
    age_sec: Optional[float]  # warp correction age (not pointing age)
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

    warp_age = _warp_correction_age_sec(tip, now=float(now))
    status = _correction_status(
        tip=tip,
        uses_commanded=uses_commanded,
        warp_age_sec=warp_age,
        max_age_sec=max_age_sec,
    )
    return BeamMapRow(
        beam=record.beam,
        status=status,
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
        age_sec=warp_age,
        outside_map=outside,
    )


def beam_row_missing(
    beam: int,
    *,
    tip: Optional[TipTiltMap] = None,
    max_age_sec: float = 120.0,
    now: Optional[float] = None,
) -> BeamMapRow:
    if now is None:
        now = time.time()
    warp_age = _warp_correction_age_sec(tip, now=float(now))
    status = _correction_status(
        tip=tip,
        uses_commanded=False,
        warp_age_sec=warp_age,
        max_age_sec=max_age_sec,
    )
    return BeamMapRow(
        beam=beam,
        status=status,
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
        age_sec=warp_age,
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
    list_pointing_beams=None,
    read_pointing=None,
    now: Optional[float] = None,
) -> List[BeamMapRow]:
    """
    Collect beam rows for the UI.

    Demo mode (``demo_pointings`` set): beams default to the demo keys (or
    ``demo_beam_numbers``). Live mode: discover ``drN`` and read etcd keys.
    If discovery fails/returns empty, fall back to etcd pointing-key scan
    (``list_pointing_beams`` / :func:`list_beam_numbers_from_pointing_keys`).
    """
    if now is None:
        now = time.time()
    if read_pointing is None:
        read_pointing = read_beam_pointing
    if list_pointing_beams is None:
        list_pointing_beams = list_beam_numbers_from_pointing_keys

    if demo_pointings is not None:
        if demo_beam_numbers is None:
            beams = sorted(demo_pointings.keys())
        else:
            beams = [int(b) for b in demo_beam_numbers]
        rows = []
        for beam in beams:
            record = demo_pointings.get(beam)
            if record is None:
                rows.append(
                    beam_row_missing(
                        beam, tip=tip, max_age_sec=max_age_sec, now=now
                    )
                )
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
        # discover_recorders may reject non-finite age_limit (UI "all" → inf).
        discover_age = (
            float(max_age_sec)
            if np.isfinite(float(max_age_sec))
            else 1.0e9
        )
        beams = list_active_beam_numbers(
            age_limit=discover_age, discover_recorders=discover_recorders
        )
    except Exception:
        beams = []

    # Fallback: etcd pointing keys (discover_recorders often fails with
    # RESOURCE_EXHAUSTED on large etcd payloads).
    if not beams:
        try:
            beams = list(list_pointing_beams())
        except Exception:
            beams = []

    rows = []
    for beam in beams:
        try:
            record = read_pointing(beam)
        except Exception:
            record = None
        if record is None:
            rows.append(
                beam_row_missing(
                    beam, tip=tip, max_age_sec=max_age_sec, now=now
                )
            )
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


def _shift_angle_arcmin(
    dra_arcsec: Optional[float], ddec_arcsec: Optional[float]
) -> tuple[float, float]:
    """Return ``(shift_arcmin, angle_deg)``; NaN when shift is unavailable."""
    if (
        dra_arcsec is None
        or ddec_arcsec is None
        or not (np.isfinite(dra_arcsec) and np.isfinite(ddec_arcsec))
    ):
        return float("nan"), float("nan")
    shift_am = float(np.hypot(dra_arcsec, ddec_arcsec) / 60.0)
    # Same convention as arrow hover: 0° = +ddec (north), 90° = +dra (east).
    angle_deg = float(np.rad2deg(np.arctan2(dra_arcsec, ddec_arcsec)))
    return shift_am, angle_deg


def build_beam_overlay(
    rows: Sequence[BeamMapRow],
    *,
    obstime_unix: float,
    central_ra_deg: float = 0.0,
    show_nominal_when_commanded: bool = True,
    # Deprecated ZEA kwarg kept for older call sites.
    central_az_deg: Optional[float] = None,
):
    """
    HoloViews Points (+ labels) for beams with a display az/el.

    Applied corrections are teal; not-applied marks are muted teal; stale
    warp corrections are gray. When commanded differs from nominal, a smaller
    orange nominal mark is optional. Hover fields match tip-tilt arrows:
    az, el, |shift|′, angle.
    """
    import holoviews as hv

    hv.extension("bokeh", logo=False)

    # Late import avoids a circular dependency with dewarp_map_app.
    from image_plane_correction.viz.dewarp_map_app import (
        azel_to_radec,
        radec_to_mollweide,
    )

    _ = central_az_deg

    def _pack(row: BeamMapRow, label: str, az: float, el: float):
        shift_am, angle_deg = _shift_angle_arcmin(row.dra_arcsec, row.ddec_arcsec)
        return {
            "az": float(az),
            "el": float(el),
            "label": label,
            "shift_arcmin": shift_am,
            "angle_deg": angle_deg,
        }

    applied_pts: list[dict] = []
    not_applied_pts: list[dict] = []
    stale_pts: list[dict] = []
    nom_pts: list[dict] = []

    for row in rows:
        if row.display_az_deg is None or row.display_el_deg is None:
            continue
        if row.status == STATUS_APPLIED:
            applied_pts.append(
                _pack(row, f"dr{row.beam}", row.display_az_deg, row.display_el_deg)
            )
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
                # Nominal mark: same hover fields; shift still from display sample.
                nom_pts.append(
                    _pack(
                        row,
                        f"dr{row.beam} nom",
                        row.nominal_az_deg,
                        row.nominal_el_deg,
                    )
                )
        elif row.status == STATUS_NOT_APPLIED:
            not_applied_pts.append(
                _pack(row, f"dr{row.beam}", row.display_az_deg, row.display_el_deg)
            )
        elif row.status == STATUS_STALE:
            stale_pts.append(
                _pack(row, f"dr{row.beam}?", row.display_az_deg, row.display_el_deg)
            )

    layers = []

    def _points(pts: list[dict], *, color, size, marker="circle", with_labels=True):
        if not pts:
            return None
        az = np.asarray([p["az"] for p in pts], dtype=np.float64)
        el = np.asarray([p["el"] for p in pts], dtype=np.float64)
        labels = [p["label"] for p in pts]
        shift_am = np.asarray([p["shift_arcmin"] for p in pts], dtype=np.float64)
        angle_deg = np.asarray([p["angle_deg"] for p in pts], dtype=np.float64)
        ra, dec = azel_to_radec(az, el, obstime_unix=float(obstime_unix))
        x, y = radec_to_mollweide(ra, dec, central_ra_deg=float(central_ra_deg))
        data = {
            "moll_x": x,
            "moll_y": y,
            "label": labels,
            "az_deg": az,
            "el_deg": el,
            "shift_arcmin": shift_am,
            "angle_deg": angle_deg,
        }
        vdims = ["label", "az_deg", "el_deg", "shift_arcmin", "angle_deg"]
        scatter = hv.Points(
            data,
            kdims=["moll_x", "moll_y"],
            vdims=vdims,
        ).opts(
            color=color,
            size=size,
            marker=marker,
            **_hover_opts(),
        )
        if not with_labels:
            # Nominal triangles are small; a nearby Labels glyph steals hits
            # and (if re-attached to HoverTool on DynamicMap updates) shows ???.
            # Identity is still available via the shared "beam" hover field.
            return scatter
        labels_el = hv.Labels(
            {"moll_x": x, "moll_y": y, "label": labels},
            kdims=["moll_x", "moll_y"],
            vdims=["label"],
        ).opts(
            text_color=color,
            text_font_size="9pt",
            yoffset=0.06,
            tools=[],
            default_tools=[],
        )
        return scatter * labels_el

    applied = _points(applied_pts, color="#00a3a3", size=12)
    if applied is not None:
        layers.append(applied)
    not_applied = _points(not_applied_pts, color="#7fbfbf", size=11)
    if not_applied is not None:
        layers.append(not_applied)
    stale = _points(stale_pts, color="#888888", size=10)
    if stale is not None:
        layers.append(stale)
    # No map Labels on nominal marks — hover carries "drN nom".
    nom = _points(
        nom_pts, color="#e67e22", size=10, marker="triangle", with_labels=False
    )
    if nom is not None:
        layers.append(nom)

    if not layers:
        return hv.Points(
            ([], []), kdims=["moll_x", "moll_y"]
        ).opts(alpha=0.0, show_legend=False)
    overlay = layers[0]
    for layer in layers[1:]:
        overlay = overlay * layer
    # Hook must run on the composed Overlay (not per-Labels): Overlay merges
    # HoverTools after child hooks and can re-attach Text renderers.
    return overlay.opts(hooks=[_strip_text_from_hover])

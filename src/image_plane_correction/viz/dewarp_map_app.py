"""Panel + HoloViews live tip-tilt / dewarp vector-field app.

Az/el quiver from a tip-tilt named-array ``.npz`` in **zenithal equal-area**
(Lambert azimuthal) projection, with optional active-beam pointing overlays.
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
from image_plane_correction.viz.beam_overlay import (
    BeamMapRow,
    build_beam_overlay,
    collect_beam_rows,
    load_demo_pointings,
    rows_fingerprint,
)

LOG = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Horizon radius in ZEA units: R = 2 sin(z/2) with z=90° → √2.
_ZEA_R_HORIZON = np.sqrt(2.0)
_ZEA_PLOT_LIM = _ZEA_R_HORIZON * 1.08
# Fixed square frame (do not stretch — that breaks equal aspect).
_ZEA_WIDTH = 720
_ZEA_HEIGHT = 720
_ZEA_FRAME = 640
# Display arrow length: 90th-percentile |shift| maps to this fraction of horizon R.
_ARROW_REF_FRAC = 0.10
_ARROW_MAX_FRAC = 0.22


def wrap_az_offset_deg(az_deg: np.ndarray, central_az_deg: float = 0.0) -> np.ndarray:
    """Azimuth relative to ``central_az_deg``, wrapped to ``[-180, 180)``."""
    return (
        np.mod(np.asarray(az_deg, dtype=np.float64) - float(central_az_deg) + 180.0, 360.0)
        - 180.0
    )


def azel_to_zea(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    *,
    central_az_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project azimuth / elevation to zenithal equal-area ``(x, y)``.

    Lambert azimuthal equal-area with the zenith at the origin. ``central_az_deg``
    is the azimuth drawn toward **+y** (default 0° = north up). ``+x`` is 90°
    clockwise from that (east when north is up). Horizon (``el=0``) is the circle
    of radius ``√2``.
    """
    az = np.asarray(az_deg, dtype=np.float64)
    el = np.asarray(el_deg, dtype=np.float64)
    # Zenith distance; clamp below-horizon to the limb for display.
    z_deg = np.clip(90.0 - el, 0.0, 90.0)
    rho = 2.0 * np.sin(np.deg2rad(z_deg) / 2.0)
    az_rel = np.deg2rad(wrap_az_offset_deg(az, central_az_deg))
    x = rho * np.sin(az_rel)
    y = rho * np.cos(az_rel)
    return x, y


# Back-compat alias used during the Mollweide → ZEA switch.
azel_to_mollweide = azel_to_zea


def _hv_curve(x: np.ndarray, y: np.ndarray, **opts):
    """HoloViews Curve with ZEA axis names (single kdim)."""
    import holoviews as hv

    return hv.Curve((x, y), kdims=["zea_x"], vdims=["zea_y"]).opts(**opts)


def zea_outline(n: int = 256):
    """Horizon circle (``el=0``) in zenithal equal-area coordinates."""
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
    return _hv_curve(
        _ZEA_R_HORIZON * np.sin(t),
        _ZEA_R_HORIZON * np.cos(t),
        color="#444444",
        line_width=1.5,
    )


def zea_graticule(
    *,
    central_az_deg: float = 0.0,
    az_step_deg: float = 45.0,
    el_step_deg: float = 15.0,
    n: int = 72,
):
    """Az/el graticule in zenithal equal-area coordinates (rays + el circles)."""
    import holoviews as hv

    curves = []
    style = dict(color="#bbbbbb", line_width=0.6)
    # Constant-azimuth rays from zenith to horizon.
    azs = np.arange(0.0, 360.0, float(az_step_deg))
    el_line = np.linspace(0.0, 90.0, int(n))
    for az in azs:
        x, y = azel_to_zea(
            np.full_like(el_line, az), el_line, central_az_deg=central_az_deg
        )
        curves.append(_hv_curve(x, y, **style))
    # Constant-elevation circles (including horizon).
    els = np.arange(0.0, 90.0, float(el_step_deg))
    az_line = np.linspace(0.0, 360.0, int(n), endpoint=True)
    for el in els:
        x, y = azel_to_zea(
            az_line, np.full_like(az_line, el), central_az_deg=central_az_deg
        )
        curves.append(_hv_curve(x, y, **style))
    return hv.Overlay(curves)


def downsample_tiptilt_uv(
    tip: TipTiltMap,
    *,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return flat arrays ``(az, el, dra, ddec, magnitude)`` for quiver display.

    Samples every ``stride``-th az/el grid node and drops non-finite vectors.
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


def _zea_frame_opts(title: str = ""):
    """Fixed square axes; xlim/ylim always the full sky disk."""
    return dict(
        title=title,
        width=_ZEA_WIDTH,
        height=_ZEA_HEIGHT,
        frame_width=_ZEA_FRAME,
        frame_height=_ZEA_FRAME,
        aspect="equal",
        xlim=(-_ZEA_PLOT_LIM, _ZEA_PLOT_LIM),
        ylim=(-_ZEA_PLOT_LIM, _ZEA_PLOT_LIM),
        xlabel="ZEA X (E when N up)",
        ylabel="ZEA Y (N up)",
        # Keep user pan/zoom across DynamicMap updates when possible.
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
    central_az_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(x, y, u_disp, v_disp)`` with arrow lengths scaled for ZEA visibility.

    Direction follows the ZEA projection of a small ``(daz, del)`` step; length
    is set from celestial ``|shift|`` so typical arrows are ~10% of the horizon
    radius (true projected displacements are ~10⁻³ of that and invisible).
    """
    x0, y0 = azel_to_zea(az, el, central_az_deg=central_az_deg)
    x1, y1 = azel_to_zea(
        az + dra / 3600.0,
        el + ddec / 3600.0,
        central_az_deg=central_az_deg,
    )
    u_raw = x1 - x0
    v_raw = y1 - y0
    raw_len = np.hypot(u_raw, v_raw)
    # Fallback direction from (dra, ddec) if projection collapses (e.g. at zenith).
    angle = np.arctan2(v_raw, u_raw)
    tiny = raw_len < 1e-15
    if np.any(tiny):
        angle = np.where(tiny, np.arctan2(ddec, dra), angle)

    positive = mag[np.isfinite(mag) & (mag > 0)]
    if positive.size == 0:
        mag_ref = 1.0
    else:
        mag_ref = float(np.nanpercentile(positive, 90))
        if not np.isfinite(mag_ref) or mag_ref <= 0.0:
            mag_ref = float(np.nanmedian(positive)) or 1.0

    arrow_ref = _ARROW_REF_FRAC * _ZEA_R_HORIZON
    arrow_max = _ARROW_MAX_FRAC * _ZEA_R_HORIZON
    disp_len = np.clip((mag / mag_ref) * arrow_ref, 0.0, arrow_max)
    disp_len = np.where(np.isfinite(disp_len), disp_len, 0.0)
    u_disp = disp_len * np.cos(angle)
    v_disp = disp_len * np.sin(angle)
    return x0, y0, u_disp, v_disp


def build_vector_element(
    tip: TipTiltMap,
    *,
    stride: int = 1,
    central_az_deg: float = 0.0,
):
    """Build only the tip-tilt ``VectorField`` element (no frame / graticule)."""
    import holoviews as hv

    az, el, dra, ddec, mag = downsample_tiptilt_uv(tip, stride=stride)
    if az.size == 0:
        return hv.VectorField(
            ([], [], [], []),
            kdims=["zea_x", "zea_y"],
            vdims=["Angle", "Magnitude"],
        ).opts(alpha=0.0)

    x0, y0, u, v = _display_uv_from_tip_tilt(
        az, el, dra, ddec, mag, central_az_deg=central_az_deg
    )
    field = hv.VectorField.from_uv(
        {
            "zea_x": x0,
            "zea_y": y0,
            "u": u,
            "v": v,
            "az_deg": az,
            "el_deg": el,
            "shift_arcsec": mag,
        },
        kdims=["zea_x", "zea_y"],
        vdims=["u", "v", "az_deg", "el_deg", "shift_arcsec"],
    )
    return field.opts(
        color="shift_arcsec",
        cmap="plasma",
        colorbar=True,
        magnitude="Magnitude",
        # Lengths already chosen in ZEA data units — do not re-normalize to 1.
        rescale_lengths=False,
        clabel="|shift| (arcsec)",
        tools=["hover"],
        line_width=1.5,
    )


def build_vectorfield(
    tip: TipTiltMap,
    *,
    stride: int = 1,
    title: str = "",
    central_az_deg: float = 0.0,
    beam_rows: Optional[List[BeamMapRow]] = None,
    show_nominal_marks: bool = True,
):
    """
    Build a zenithal equal-area az/el HoloViews overlay.

    Horizon + graticule + tip-tilt vectors (+ optional beam marks).
    """
    import holoviews as hv

    outline = zea_outline()
    graticule = zea_graticule(central_az_deg=central_az_deg)
    field = build_vector_element(
        tip, stride=stride, central_az_deg=central_az_deg
    )
    overlay = outline * graticule * field
    if beam_rows is not None:
        overlay = overlay * build_beam_overlay(
            beam_rows,
            central_az_deg=central_az_deg,
            show_nominal_when_commanded=show_nominal_marks,
        )
    return overlay.opts(**_zea_frame_opts(title))


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


def _beam_table_html(rows: List[BeamMapRow]) -> str:
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
        "<th style='text-align:right;padding:2px 8px'>dra″</th>"
        "<th style='text-align:right;padding:2px 8px'>ddec″</th>"
        "<th style='text-align:right;padding:2px 8px'>|shift|″</th>"
        "<th style='text-align:right;padding:2px 8px'>age s</th>"
        "</tr></thead><tbody>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            f"<td style='padding:2px 8px'>dr{row.beam}</td>"
            f"<td style='padding:2px 8px'>{row.status}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.display_az_deg)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.display_el_deg)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.nominal_az_deg)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.nominal_el_deg)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>"
            f"{'yes' if row.uses_commanded else '—'}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.dra_arcsec)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.ddec_arcsec)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.shift_arcsec)}</td>"
            f"<td style='text-align:right;padding:2px 8px'>{_fmt_opt(row.age_sec, '{:.0f}')}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


class DewarpMapApp(param.Parameterized):
    """Param-driven live tip-tilt map view for one frequency / warp path."""

    warp_path = param.String(default="", doc="Path to tip-tilt .npz (empty → etcd resolve)")
    frequency_mhz = param.Integer(default=55, bounds=(1, 300), doc="App frequency MHz")
    poll_seconds = param.Number(default=3.0, bounds=(0.5, 60.0))
    stride = param.Integer(default=1, bounds=(1, 32), doc="Grid downsample stride")
    central_az_deg = param.Number(
        default=0.0,
        softbounds=(0.0, 360.0),
        doc="ZEA up-azimuth (deg); 0=N up, 90=E up, 180=S up, 270=W up",
    )
    max_pointing_age_sec = param.Number(
        default=120.0,
        bounds=(1.0, 3600.0),
        doc="Pointing older than this is marked stale",
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

    def _set_header(self, path: Optional[Path], mtime: float, obstime: float) -> None:
        name = path.name if path is not None else "?"
        self.header = (
            f"### Tip-tilt **{self.frequency_mhz} MHz** — `{name}`\n\n"
            f"- **map obstime:** {_format_unix(obstime)}\n"
            f"- **file mtime:** {_format_unix(mtime)}\n"
            f"- **revision:** {self.revision}"
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
                max_age_sec=float(self.max_pointing_age_sec),
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
        title = (
            f"Tip-tilt {self.frequency_mhz} MHz | "
            f"obstime {_format_unix(obstime)} | "
            f"mtime {_format_unix(mtime)} | "
            f"ZEA (up az={self.central_az_deg:g}°)"
        )
        return {
            "tip": self._tip,
            "beam_rows": list(self._beam_rows),
            "stride": int(self.stride),
            "central_az_deg": float(self.central_az_deg),
            "show_nominal_marks": bool(self.show_nominal_marks),
            "title": title,
        }

    def _render_payload(self, data: Optional[dict] = None):
        """Build overlay from a Pipe payload (DynamicMap; kwarg must be ``data``)."""
        import holoviews as hv

        payload = data
        if not payload or payload.get("tip") is None:
            empty = hv.Text(0.0, 0.0, "No map loaded yet")
            return (zea_outline() * empty).opts(**_zea_frame_opts("No map loaded"))
        return build_vectorfield(
            payload["tip"],
            stride=int(payload.get("stride", 1)),
            title=str(payload.get("title") or ""),
            central_az_deg=float(payload.get("central_az_deg", 0.0)),
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

    @param.depends("stride", "central_az_deg", "show_nominal_marks", watch=True)
    def _on_display_params(self) -> None:
        self._compose_plot()
        self.status = (
            f"Redraw stride={self.stride} central_az={self.central_az_deg:g}"
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
                "central_az_deg",
                "max_pointing_age_sec",
                "show_nominal_marks",
                "poll_seconds",
            ],
            widgets={
                "frequency_mhz": {"disabled": True},
                "poll_seconds": {"disabled": True},
            },
            name="Dewarp map",
        )
        demo_note = (
            " (demo pointings)"
            if self._demo_pointings is not None
            else " (live etcd pointings)"
        )
        self._ensure_dynamic_plot()
        # Fixed square pane — stretch_both was warping the ZEA disk.
        self._plot_pane = pn.pane.HoloViews(
            self._plot_dmap,
            width=_ZEA_WIDTH,
            height=_ZEA_HEIGHT,
            sizing_mode="fixed",
            linked_axes=False,
        )
        self._layout = pn.Column(
            pn.pane.Markdown(
                f"# Live dewarp / tip-tilt map\n"
                f"Frequency **{self.frequency_mhz} MHz** — "
                f"path `{self.warp_path or '(from etcd)'}`{demo_note}"
            ),
            # pn.panel() wraps @param.depends methods so they re-render.
            pn.panel(self.header_view),
            controls,
            pn.panel(self.status_view),
            self._plot_pane,
            pn.pane.Markdown(self.beam_table_view),
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
    stride: int = 1,
    central_az_deg: float = 0.0,
    max_pointing_age_sec: float = 120.0,
    show_nominal_marks: bool = True,
    demo_pointings_path: Optional[PathLike] = None,
) -> DewarpMapApp:
    """Factory used by the serve script."""
    demo = None
    if demo_pointings_path is not None:
        demo = load_demo_pointings(demo_pointings_path)
    return DewarpMapApp(
        warp_path="" if warp_path is None else str(Path(warp_path)),
        frequency_mhz=int(frequency_mhz),
        poll_seconds=float(poll_seconds),
        stride=int(stride),
        central_az_deg=float(central_az_deg),
        max_pointing_age_sec=float(max_pointing_age_sec),
        show_nominal_marks=bool(show_nominal_marks),
        demo_pointings=demo,
    )

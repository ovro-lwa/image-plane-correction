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
_ZEA_WIDTH = 700
_ZEA_HEIGHT = 700


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
    return dict(
        title=title,
        width=_ZEA_WIDTH,
        height=_ZEA_HEIGHT,
        aspect="equal",
        xlim=(-_ZEA_PLOT_LIM, _ZEA_PLOT_LIM),
        ylim=(-_ZEA_PLOT_LIM, _ZEA_PLOT_LIM),
        xlabel="ZEA X (E when N up)",
        ylabel="ZEA Y (N up)",
    )


def build_vectorfield(
    tip: TipTiltMap,
    *,
    stride: int = 1,
    title: str = "",
    central_az_deg: float = 0.0,
):
    """
    Build a zenithal equal-area az/el HoloViews overlay (horizon + graticule + vectors).

    Tip-tilt ``(dra, ddec)`` arcsec are projected by finite-differencing the ZEA
    map after treating them as small ``(daz, del)`` displacements (same display
    convention as the prior rectangular / Mollweide quivers).
    """
    import holoviews as hv

    outline = zea_outline()
    graticule = zea_graticule(central_az_deg=central_az_deg)
    frame = _zea_frame_opts(title or "No finite tip-tilt samples")
    az, el, dra, ddec, mag = downsample_tiptilt_uv(tip, stride=stride)
    if az.size == 0:
        empty = hv.VectorField(
            ([], [], [], []),
            kdims=["zea_x", "zea_y"],
            vdims=["Angle", "Magnitude"],
        )
        return (outline * graticule * empty).opts(
            hv.opts.VectorField(
                title=frame["title"],
                xlabel=frame["xlabel"],
                ylabel=frame["ylabel"],
            ),
            hv.opts.Overlay(**frame),
        )

    x0, y0 = azel_to_zea(az, el, central_az_deg=central_az_deg)
    # Display convention: map celestial tip-tilt arcsec onto local az/el axes.
    daz_deg = dra / 3600.0
    del_deg = ddec / 3600.0
    x1, y1 = azel_to_zea(
        az + daz_deg, el + del_deg, central_az_deg=central_az_deg
    )
    u = x1 - x0
    v = y1 - y0

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
    # Color by celestial |shift|; arrow length from projected (u, v).
    field = field.opts(
        color="shift_arcsec",
        cmap="plasma",
        colorbar=True,
        magnitude="Magnitude",
        rescale_lengths=True,
        clabel="|shift| (arcsec)",
        tools=["hover"],
        xlabel=frame["xlabel"],
        ylabel=frame["ylabel"],
    )
    overlay = outline * graticule * field
    return overlay.opts(**{**frame, "title": title})


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
        self._callback = None
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

    def _compose_plot(self) -> None:
        """Rebuild ZEA vector field + beam overlays and bump revision."""
        if self._tip is None:
            self._plot_obj = None
            self.revision = int(self.revision) + 1
            return
        path = self._loaded_path
        mtime = self._loaded_mtime or float("nan")
        title = (
            f"Tip-tilt {self.frequency_mhz} MHz | "
            f"obstime {_format_unix(self._tip.obstime_unix)} | "
            f"mtime {_format_unix(mtime)} | "
            f"ZEA (up az={self.central_az_deg:g}°)"
        )
        base = build_vectorfield(
            self._tip,
            stride=int(self.stride),
            title=title,
            central_az_deg=float(self.central_az_deg),
        )
        beams = build_beam_overlay(
            self._beam_rows,
            central_az_deg=float(self.central_az_deg),
            show_nominal_when_commanded=bool(self.show_nominal_marks),
        )
        self._plot_obj = (base * beams).opts(**_zea_frame_opts(title))
        self.revision = int(self.revision) + 1

    @param.depends("stride", "central_az_deg", "show_nominal_marks", watch=True)
    def _on_display_params(self) -> None:
        self._compose_plot()
        self.status = (
            f"Redraw stride={self.stride} central_az={self.central_az_deg:g}"
        )

    @param.depends("revision", "stride", "central_az_deg", "show_nominal_marks")
    def plot_view(self):
        """Reactive HoloViews plot; re-runs when ``revision`` bumps."""
        if self._plot_obj is None:
            return self._hv.Text(0.5, 0.5, "No map loaded yet").opts(
                width=_ZEA_WIDTH, height=_ZEA_HEIGHT
            )
        return self._plot_obj

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
        pn = self._pn
        if self._callback is not None:
            return
        period_ms = int(max(0.5, float(self.poll_seconds)) * 1000)

        def _start():
            if self._callback is not None:
                return
            # Attach to the live Bokeh document for this session.
            self._callback = pn.state.add_periodic_callback(
                self.poll, period=period_ms, start=True
            )
            LOG.info(
                "Started dewarp-map poll every %s ms (freq=%s)",
                period_ms,
                self.frequency_mhz,
            )

        # If a document already exists, start now; otherwise wait for session onload.
        try:
            curdoc = pn.state.curdoc
        except Exception:
            curdoc = None
        if curdoc is not None:
            _start()
        else:
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
            pn.pane.HoloViews(self.plot_view, sizing_mode="stretch_both"),
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

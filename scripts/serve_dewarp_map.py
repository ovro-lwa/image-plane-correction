#!/usr/bin/env python3
"""Serve the live-dewarp / beamformer tip-tilt vector-field Panel app.

Requires optional viz extras::

    pip install '.[viz]'

Discovers up to 16 per-subband tip-tilt maps under a warp directory (default
``/fast/claw/warps/warp_<N>MHz.npz``). The UI lets the operator select any
discovered subband. Launch with Panel (preferred; one app instance per browser
session)::

    PYTHONPATH=src panel serve scripts/serve_dewarp_map.py --show \\
        --args --warp-dir /fast/claw/warps

Or run this script directly (starts ``pn.serve``)::

    PYTHONPATH=src python scripts/serve_dewarp_map.py \\
        --warp-dir /fast/claw/warps --poll-seconds 3 --show

Optional ``--frequency-mhz`` only sets the initial subband selection when that
map is present. Pass ``--warp-path`` to lock the app to a single NPZ. Pass
``--warp-dir ''`` with no ``--warp-path`` to resolve maps via
``mnc.tiptilt.get_tiptilt_path`` for the selected catalog subband.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from image_plane_correction.tiptilt_map import (  # noqa: E402
    DEFAULT_WARP_DIR,
    MAX_WARP_SUBBANDS,
    discover_warp_subbands_mhz,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Serve live dewarp / beamformer tip-tilt correction "
            f"(discover up to {MAX_WARP_SUBBANDS} subbands)"
        )
    )
    parser.add_argument(
        "--warp-dir",
        type=str,
        default=str(DEFAULT_WARP_DIR),
        help=(
            "Directory of per-subband tip-tilt NPZ files "
            f"(default: {DEFAULT_WARP_DIR}; empty string → etcd catalog)"
        ),
    )
    parser.add_argument(
        "--warp-path",
        type=Path,
        default=None,
        help="Lock to one tip-tilt NPZ (disables multi-subband discovery)",
    )
    parser.add_argument(
        "--frequency-mhz",
        type=int,
        default=None,
        help=(
            "Optional initial subband selection in MHz when that map is "
            "discovered (default: first discovered subband)"
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=3.0,
        help="How often to rediscover maps / reload (default: 3)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Equal-area sample spacing multiplier (default: 3 → 7.5°)",
    )
    parser.add_argument(
        "--central-ra-deg",
        type=float,
        default=0.0,
        help="Mollweide central RA in deg (default: 0 → RA=0 at centre)",
    )
    parser.add_argument(
        "--central-az-deg",
        type=float,
        default=None,
        help=argparse.SUPPRESS,  # deprecated ZEA control
    )
    parser.add_argument(
        "--demo-pointings",
        type=Path,
        default=None,
        help="JSON fixture of /mon/beamformer/{N}/pointing payloads (skip live etcd)",
    )
    parser.add_argument(
        "--max-pointing-age-sec",
        type=float,
        default=120.0,
        help="Mark pointing stale when older than this (default: 120)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5007,
        help="Port when running as __main__ with pn.serve (default: 5007)",
    )
    parser.add_argument(
        "--address",
        default="127.0.0.1",
        help="Bind address for pn.serve (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Open the app URL once in a browser (do not also click the "
            "printed URL, or you get two tabs)"
        ),
    )
    return parser.parse_known_args(argv)[0]


def _resolved_warp_dir(args: argparse.Namespace) -> Optional[Path]:
    raw = "" if args.warp_dir is None else str(args.warp_dir).strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _build_app(args: argparse.Namespace):
    from image_plane_correction.viz.dewarp_map_app import create_app

    freq = args.frequency_mhz
    if freq is not None and int(freq) <= 0:
        raise SystemExit("--frequency-mhz must be a positive int")

    return create_app(
        warp_path=args.warp_path,
        warp_dir=_resolved_warp_dir(args),
        frequency_mhz=freq,
        poll_seconds=args.poll_seconds,
        stride=args.stride,
        central_ra_deg=args.central_ra_deg,
        central_az_deg=args.central_az_deg,
        max_pointing_age_sec=args.max_pointing_age_sec,
        demo_pointings_path=args.demo_pointings,
    )


def _session_layout(args: argparse.Namespace):
    """Build a fresh app+layout for one browser session (poll callback attaches here)."""
    return _build_app(args).panel()


_args = _parse_args()

# ``panel serve`` imports this module per browser session — servable app attaches
# its periodic callback via ``pn.state.onload``.
if __name__ != "__main__":
    _build_app(_args).servable(title="Live dewarp")

if __name__ == "__main__":
    import threading
    import time

    import panel as pn

    _warp_dir = _resolved_warp_dir(_args)
    if _args.warp_path is not None:
        _warp_label = str(_args.warp_path)
        _bands_label = "locked"
    elif _warp_dir is not None:
        _bands = discover_warp_subbands_mhz(
            _warp_dir, max_subbands=MAX_WARP_SUBBANDS
        )
        _warp_label = str(_warp_dir)
        _bands_label = (
            f"{len(_bands)}/{MAX_WARP_SUBBANDS} ["
            + ",".join(str(b) for b in _bands)
            + "]"
            if _bands
            else f"0/{MAX_WARP_SUBBANDS} (waiting)"
        )
    else:
        _warp_label = "etcd"
        _bands_label = f"catalog≤{MAX_WARP_SUBBANDS}"

    url = f"http://{_args.address}:{_args.port}/"
    print(
        f"Serving live dewarp at {url} "
        f"(warp={_warp_label}, subbands={_bands_label}, "
        f"poll={_args.poll_seconds}s)",
        file=sys.stderr,
    )
    # Always show=False. Panel's show=True calls server.show('/') and opens a
    # tab; clicking the printed URL (or an IDE localhost preview) then opens a
    # second tab. --show opens exactly once ourselves after the server is up.
    if _args.show:
        import webbrowser

        def _open_once() -> None:
            time.sleep(0.8)
            webbrowser.open(url, new=0)
            print(
                f"Opened {url} once (--show). "
                "Do not click the printed URL again or you get a second tab.",
                file=sys.stderr,
            )

        threading.Thread(target=_open_once, daemon=True).start()

    pn.serve(
        lambda: _session_layout(_args),
        port=int(_args.port),
        address=_args.address,
        show=False,
        title="Live dewarp",
        websocket_origin=[f"{_args.address}:{_args.port}", f"localhost:{_args.port}"],
    )

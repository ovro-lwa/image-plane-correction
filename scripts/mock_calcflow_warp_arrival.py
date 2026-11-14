#!/usr/bin/env python3
"""
Mock the watch_calcflow_warp service by walking a sorted list of ``*.fits.fz``
files and processing one every ``--interval-seconds`` (default 10).

Uses the same funpack → calcflow → fixed warp filename path as
``watch_calcflow_warp.py``, so you can exercise the pipeline without waiting
on live pipeline drops.

Example::

    PYTHONPATH=src python scripts/mock_calcflow_warp_arrival.py \\
        --source-dir /lustre/pipeline/.../55MHz \\
        --output-dir /lustre/claw/warps \\
        --warp-filename warp_55MHz.npz \\
        --cleaned --target-size 1024 --interval-seconds 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import watch_calcflow_warp as wcw  # noqa: E402

LOG = logging.getLogger("mock_calcflow_warp_arrival")


def _collect_fz(source_dir: Optional[Path], files: list[str]) -> list[Path]:
    paths: list[Path] = []
    if source_dir is not None:
        if not source_dir.is_dir():
            raise SystemExit(f"--source-dir is not a directory: {source_dir}")
        paths.extend(source_dir.glob("*.fits.fz"))
    for item in files:
        p = Path(item)
        if p.is_dir():
            paths.extend(p.glob("*.fits.fz"))
        elif p.is_file():
            paths.append(p)
        else:
            raise SystemExit(f"Not a file or directory: {p}")
    # Unique by resolved path, then chronological by embedded timestamp.
    uniq = {str(p.resolve()): p for p in paths}
    return sorted(uniq.values(), key=wcw._name_sort_key)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    warp_filename = args.warp_filename
    if Path(warp_filename).name != warp_filename:
        raise SystemExit(
            f"--warp-filename must be a bare filename, not a path: {warp_filename}"
        )
    if not warp_filename.lower().endswith(".npz"):
        raise SystemExit(
            f"--warp-filename must end with .npz (got {warp_filename!r})"
        )

    fz_list = _collect_fz(
        Path(args.source_dir) if args.source_dir else None,
        list(args.files or []),
    )
    if not fz_list:
        raise SystemExit("No *.fits.fz files found to process")

    reference_cache = (
        None
        if args.no_cache_reference_sky
        else wcw.ReferenceSkyCache()
    )
    LOG.info(
        "Mocking %d arrivals every %.1fs -> output-dir=%s warp=%s cache_ref=%s",
        len(fz_list),
        args.interval_seconds,
        output_dir,
        warp_filename,
        reference_cache is not None,
    )
    for i, fz in enumerate(fz_list, start=1):
        LOG.info("[%d/%d] Appearing: %s", i, len(fz_list), fz.name)
        try:
            wcw.process_fz(
                fz,
                output_dir=output_dir,
                warp_filename=warp_filename,
                funpack_bin=args.funpack,
                keep_fits=args.keep_fits,
                cleaned=args.cleaned,
                psf_fn=args.psf,
                qa=args.qa,
                target_size=args.target_size,
                catalog=args.catalog,
                catalog_path=args.catalog_path,
                alpha=args.alpha,
                gamma=args.gamma,
                scale_factor=args.scale_factor,
                reference_cache=reference_cache,
                publish_tiptilt=args.publish_tiptilt,
                ttl_sec=args.ttl_sec,
                frequency_mhz=args.frequency_mhz,
                az_step_deg=args.az_step_deg,
                el_step_deg=args.el_step_deg,
                el_min_deg=args.el_min_deg,
            )
        except Exception:
            LOG.exception("Failed processing %s", fz)
            if args.fail_fast:
                raise

        if i < len(fz_list):
            LOG.info("Sleeping %.1fs before next mock arrival", args.interval_seconds)
            time.sleep(args.interval_seconds)

    LOG.info("Done; processed %d file(s)", len(fz_list))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Iterate *.fits.fz in timestamp order, processing one every N seconds "
            "(mock new-file arrivals for calcflow warp)."
        )
    )
    p.add_argument(
        "--source-dir",
        default=None,
        help="Directory of *.fits.fz to walk (sorted by name timestamp)",
    )
    p.add_argument(
        "files",
        nargs="*",
        help="Optional explicit *.fits.fz paths (and/or extra directories)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for unpacked .fits and the warp map",
    )
    p.add_argument(
        "--warp-filename",
        required=True,
        help="Warp map filename inside --output-dir (e.g. warp_55MHz.npz)",
    )
    p.add_argument(
        "--interval-seconds",
        type=float,
        default=10.0,
        help="Seconds between mock arrivals (default: 10)",
    )
    p.add_argument(
        "--funpack",
        default="funpack",
        help="funpack executable (default: funpack on PATH)",
    )
    p.add_argument(
        "--remove-unpacked",
        dest="keep_fits",
        action="store_false",
        help="Delete the unpacked .fits from --output-dir after writing the warp",
    )
    p.set_defaults(keep_fits=True)
    p.add_argument(
        "--cleaned",
        action="store_true",
        help="Treat inputs as cleaned images (PSF from BMAJ/BMIN in header)",
    )
    p.add_argument(
        "--psf",
        default=None,
        help="Optional PSF FITS path passed to calcflow",
    )
    p.add_argument(
        "--qa",
        action="store_true",
        help="Enable calcflow residual QA (skip warp write if QA fails)",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Optional square resample size for calcflow",
    )
    p.add_argument("--catalog", default="VLSSR")
    p.add_argument(
        "--catalog-path",
        default="/home/claw/vlssr_radecpeak_unresolved.txt",
    )
    p.add_argument("--alpha", type=float, default=1.3)
    p.add_argument("--gamma", type=float, default=150.0)
    p.add_argument("--scale-factor", type=float, default=0.7)
    p.add_argument(
        "--no-cache-reference-sky",
        action="store_true",
        help="Rebuild theoretical reference_sky on every frame (default: reuse when WCS, target_size, and DATE-OBS/MJD-OBS match)",
    )
    p.add_argument(
        "--publish-tiptilt",
        dest="publish_tiptilt",
        action="store_true",
        default=True,
        help="After writing the NPZ, call mnc.tiptilt.set_tiptilt (default: on)",
    )
    p.add_argument(
        "--no-publish-tiptilt",
        dest="publish_tiptilt",
        action="store_false",
        help="Write the NPZ only; do not publish to etcd",
    )
    p.add_argument(
        "--ttl-sec",
        type=float,
        default=600.0,
        help="Tip-tilt publication TTL in seconds (default: 600)",
    )
    p.add_argument(
        "--frequency-mhz",
        type=int,
        default=None,
        help="Subband MHz for etcd publish (default: parse _<N>MHz from --warp-filename)",
    )
    p.add_argument(
        "--az-step-deg",
        type=float,
        default=5.0,
        help="Azimuth grid step for tip-tilt NPZ (default: 5)",
    )
    p.add_argument(
        "--el-step-deg",
        type=float,
        default=2.5,
        help="Elevation grid step for tip-tilt NPZ (default: 2.5)",
    )
    p.add_argument(
        "--el-min-deg",
        type=float,
        default=0.0,
        help="Minimum elevation for tip-tilt NPZ grid (default: 0)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first processing error (default: log and continue)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Keep watcher helpers' logger visible at the same level.
    logging.getLogger("watch_calcflow_warp").setLevel(getattr(logging, args.log_level))
    if not args.source_dir and not args.files:
        raise SystemExit("Provide --source-dir and/or explicit *.fits.fz paths")
    run(args)


if __name__ == "__main__":
    main()

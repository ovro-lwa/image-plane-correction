#!/usr/bin/env python3
"""
Watch a directory of UT day subdirs (``YYYY-MM-DD``) for ``*.fits.fz`` files,
use only the latest day, pick the newest file by the ``YYYYMMDD_HHMMSS`` token
in the filename, funpack it into ``--output-dir``, run ``calcflow``, and write
a tip-tilt named-array warp ``.npz`` as ``--warp-filename`` in that same
directory. Optionally publish the absolute path to etcd via
:func:`mnc.tiptilt.set_tiptilt`.

When a newer ``YYYY-MM-DD`` subdirectory appears under ``--watch-dir``, the
watcher switches to that day automatically.

Designed as a long-running process for user ``claw`` (e.g. systemd --user).

Example::

    PYTHONPATH=src python scripts/watch_calcflow_warp.py \\
        --watch-dir /lustre/pipeline/exopipe/.../55MHz \\
        --output-dir /lustre/claw/warps \\
        --warp-filename warp_55MHz.npz \\
        --cleaned --target-size 1024

See also ``scripts/watch_calcflow_warp.service`` for a systemd user unit.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from astropy.io import fits
from astropy.wcs import WCS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from image_plane_correction.flow import calcflow  # noqa: E402
from image_plane_correction.reference_cache import ReferenceSkyCache  # noqa: E402
from image_plane_correction.tiptilt_map import (  # noqa: E402
    obstime_unix_from_header,
    pixel_flow_to_tiptilt_npz,
    resolve_publish_frequencies_mhz,
)

LOG = logging.getLogger("watch_calcflow_warp")


def _funpack(fz_path: Path, fits_path: Path, funpack_bin: str) -> None:
    """Decompress ``fz_path`` to ``fits_path`` with CFITSIO funpack."""
    if fits_path.exists():
        LOG.info("FITS already exists, skipping funpack: %s", fits_path)
        return
    # -O sets output name; default would strip .fz in the same directory.
    cmd = [funpack_bin, "-O", str(fits_path), str(fz_path)]
    LOG.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _wait_until_stable(
    path: Path,
    *,
    settle_s: float,
    poll_s: float,
    timeout_s: float,
) -> bool:
    """
    Return True once ``path`` size is unchanged for ``settle_s`` seconds.

    Avoids processing a ``.fits.fz`` that is still being written.
    """
    deadline = time.monotonic() + timeout_s
    last_size: Optional[int] = None
    stable_since: Optional[float] = None
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        now = time.monotonic()
        if size == last_size and size > 0:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= settle_s:
                return True
        else:
            last_size = size
            stable_since = None
        time.sleep(poll_s)
    LOG.warning("Timed out waiting for stable size: %s", path)
    return False


def _write_warp_npz(
    flow,
    warp_output: Path,
    *,
    imwcs: WCS,
    obstime_unix: float,
    az_step_deg: float,
    el_step_deg: float,
    el_min_deg: float,
) -> None:
    """Write tip-tilt named-array ``.npz`` from ``flow.offsets`` + active WCS."""
    pixel_flow_to_tiptilt_npz(
        warp_output,
        flow.offsets,
        imwcs,
        obstime_unix,
        az_step_deg=az_step_deg,
        el_step_deg=el_step_deg,
        el_min_deg=el_min_deg,
    )
    LOG.info("Wrote tip-tilt warp %s", warp_output)


def _publish_tiptilt(
    warp_output: Path,
    *,
    ttl_sec: float,
    frequencies_mhz: Sequence[int],
) -> None:
    """
    Publish ``warp_output`` absolute path to etcd for ``mnc.tiptilt`` consumers.

    ``frequencies_mhz`` may be one subband or many (same map under each key).
    Import/etcd failures are logged and swallowed so local testing can still write
    the NPZ when ``mnc``/``lwa_f`` or etcd are unavailable. Use
    ``--no-publish-tiptilt`` to skip this path entirely.
    """
    abs_path = str(warp_output.resolve())
    freqs = tuple(int(f) for f in frequencies_mhz)
    if not freqs:
        LOG.warning("Tip-tilt publish skipped (empty frequency list) for %s", abs_path)
        return
    try:
        from mnc.tiptilt import set_tiptilt
    except ImportError as exc:
        LOG.warning(
            "Tip-tilt publish skipped (mnc.tiptilt import failed: %s). "
            "NPZ was written to %s. Pass --no-publish-tiptilt to silence this.",
            exc,
            abs_path,
        )
        return
    ok: list[int] = []
    for frequency_mhz in freqs:
        try:
            set_tiptilt(abs_path, ttl_sec, frequency_mhz=frequency_mhz)
        except Exception:
            LOG.exception(
                "Tip-tilt publish failed for %s (frequency_mhz=%d); NPZ is on disk",
                abs_path,
                frequency_mhz,
            )
            continue
        ok.append(frequency_mhz)
    if not ok:
        return
    if len(ok) == 1:
        LOG.info(
            "Published tip-tilt etcd entry frequency_mhz=%d path=%s ttl_sec=%s",
            ok[0],
            abs_path,
            ttl_sec,
        )
    else:
        LOG.info(
            "Published tip-tilt etcd entries frequencies_mhz=%s path=%s ttl_sec=%s",
            ok,
            abs_path,
            ttl_sec,
        )


def _unpacked_fits_name(fz_path: Path) -> str:
    """``foo.fits.fz`` -> ``foo.fits``."""
    name = fz_path.name
    if name.endswith(".fits.fz"):
        return name[: -len(".fz")]
    return Path(name).with_suffix("").name


def process_fz(
    fz_path: Path,
    *,
    output_dir: Path,
    warp_filename: str,
    funpack_bin: str,
    keep_fits: bool,
    cleaned: bool,
    psf_fn: Optional[str],
    qa: bool,
    target_size: Optional[int],
    catalog: str,
    catalog_path: str,
    alpha: float,
    gamma: float,
    scale_factor: float,
    reference_cache: Optional[ReferenceSkyCache] = None,
    max_flux: float = 20.0,
    horizon_elevation_deg: Optional[float] = 10.0,
    publish_tiptilt: bool = True,
    ttl_sec: float = 600.0,
    frequency_mhz: Optional[int] = None,
    publish_all_frequencies: bool = False,
    az_step_deg: float = 5.0,
    el_step_deg: float = 2.5,
    el_min_deg: float = 0.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = output_dir / _unpacked_fits_name(fz_path)
    warp_output = output_dir / warp_filename
    if warp_output.suffix.lower() != ".npz":
        raise ValueError(
            f"--warp-filename must end with .npz (got {warp_filename!r})"
        )

    _funpack(fz_path, fits_path, funpack_bin)

    cached_ref = None
    cache_key = None
    header = fits.getheader(str(fits_path))
    if reference_cache is not None:
        cache_key = ReferenceSkyCache.make_key(
            header=header,
            target_size=target_size,
        )
        cached_ref = reference_cache.get(cache_key)
        if cached_ref is not None:
            LOG.info(
                "Reusing cached reference_sky (hits=%d misses=%d)",
                reference_cache.hits,
                reference_cache.misses,
            )
        else:
            LOG.info(
                "Building reference_sky (cache miss; hits=%d misses=%d)",
                reference_cache.hits,
                reference_cache.misses,
            )

    LOG.info("Running calcflow on %s", fits_path)
    # Warp-only: skip apply_broadcast unless QA needs the dewarped image.
    _image, ref_hdu, flow, _dewarped, qa_ok = calcflow(
        str(fits_path),
        psf_fn=psf_fn,
        reference_sky=cached_ref,
        cleaned=cleaned,
        qa=qa,
        write=False,
        catalog=catalog,
        catalog_path=catalog_path,
        max_flux=max_flux,
        alpha=alpha,
        gamma=gamma,
        scale_factor=scale_factor,
        horizon_elevation_deg=horizon_elevation_deg,
        target_size=target_size,
        catalog_qa=False,
        dewarp=False,
    )
    if reference_cache is not None and cache_key is not None and cached_ref is None:
        reference_cache.store(cache_key, ref_hdu)

    if qa and not qa_ok:
        LOG.warning("QA failed for %s; not writing warp", fits_path)
        return

    # Active calcflow grid WCS lives on the reference HDU header.
    imwcs = WCS(ref_hdu.header).celestial
    obstime_unix = obstime_unix_from_header(header)
    _write_warp_npz(
        flow,
        warp_output,
        imwcs=imwcs,
        obstime_unix=obstime_unix,
        az_step_deg=az_step_deg,
        el_step_deg=el_step_deg,
        el_min_deg=el_min_deg,
    )

    if publish_tiptilt:
        freqs = resolve_publish_frequencies_mhz(
            warp_path=warp_output,
            frequency_mhz=frequency_mhz,
            publish_all_frequencies=publish_all_frequencies,
        )
        _publish_tiptilt(warp_output, ttl_sec=ttl_sec, frequencies_mhz=freqs)

    if not keep_fits:
        try:
            fits_path.unlink()
            LOG.info("Removed unpacked FITS %s", fits_path)
        except OSError as exc:
            LOG.warning("Could not remove %s: %s", fits_path, exc)


_DAY_DIR_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _parse_day_dirname(name: str) -> Optional[date]:
    """
    Return a :class:`~datetime.date` if ``name`` is a valid UT day dir
    (``YYYY-MM-DD``), else None.
    """
    m = _DAY_DIR_RE.fullmatch(name)
    if m is None:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _latest_day_dir(watch_dir: Path) -> Optional[Path]:
    """
    Return the newest ``YYYY-MM-DD`` subdirectory under ``watch_dir``, or None.

    Only direct child directories whose names parse as calendar dates are
    considered (UT day folders from the pipeline). Lexicographic order of
    valid ISO dates matches chronological order.
    """
    latest: Optional[Path] = None
    latest_day: Optional[date] = None
    try:
        children = watch_dir.iterdir()
    except OSError as exc:
        LOG.warning("Could not list --watch-dir %s: %s", watch_dir, exc)
        return None
    for child in children:
        if not child.is_dir():
            continue
        day = _parse_day_dirname(child.name)
        if day is None:
            continue
        if latest_day is None or day > latest_day:
            latest_day = day
            latest = child
    return latest


def _name_sort_key(path: Path) -> str:
    """
    Sort key from the filename timestamp convention.

    Pipeline names embed ``YYYYMMDD_HHMMSS`` (e.g.
    ``55MHz-I-Taper-10min-20241218_030021-image.fits.fz``). Lexicographic
    order of that token matches chronological order; fall back to the full
    name when no timestamp is present.
    """
    m = re.search(r"(20\d{6}_\d{6})", path.name)
    return m.group(1) if m else path.name


def _latest_fz(day_dir: Path) -> Optional[Path]:
    """Return the newest ``*.fits.fz`` in ``day_dir`` by filename timestamp."""
    files = list(day_dir.glob("*.fits.fz"))
    if not files:
        return None
    return max(files, key=_name_sort_key)


def run_loop(args: argparse.Namespace) -> None:
    watch_dir = Path(args.watch_dir)
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
    if not watch_dir.is_dir():
        raise SystemExit(f"--watch-dir is not a directory: {watch_dir}")

    # Track last processed file as (day_dirname, filename) so a same-named
    # file on a new UT day is not skipped, and a new day can be adopted.
    last_processed: Optional[tuple[str, str]] = None
    active_day_dir: Optional[Path] = None
    reference_cache: Optional[ReferenceSkyCache] = (
        None if args.no_cache_reference_sky else ReferenceSkyCache()
    )

    def resolve_day_dir() -> Optional[Path]:
        nonlocal active_day_dir
        day_dir = _latest_day_dir(watch_dir)
        if day_dir is None:
            if active_day_dir is not None:
                LOG.warning("No YYYY-MM-DD day subdirectory under %s", watch_dir)
                active_day_dir = None
            return None
        if active_day_dir is None or day_dir.resolve() != active_day_dir.resolve():
            LOG.info("Using latest UT day directory: %s", day_dir)
            active_day_dir = day_dir
        return day_dir

    def maybe_process_latest() -> None:
        nonlocal last_processed
        day_dir = resolve_day_dir()
        if day_dir is None:
            return
        fz = _latest_fz(day_dir)
        if fz is None:
            return
        key = (day_dir.name, fz.name)
        if key == last_processed:
            return
        if not _wait_until_stable(
            fz,
            settle_s=args.settle_seconds,
            poll_s=args.poll_seconds,
            timeout_s=args.stable_timeout,
        ):
            return
        # After settle, re-resolve day + latest in case a newer day or file arrived.
        day_dir = resolve_day_dir()
        if day_dir is None:
            return
        fz = _latest_fz(day_dir)
        if fz is None:
            return
        key = (day_dir.name, fz.name)
        if key == last_processed:
            return
        try:
            LOG.info("Selected latest by name: %s/%s", day_dir.name, fz.name)
            process_fz(
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
                horizon_elevation_deg=args.horizon_elevation_deg,
                reference_cache=reference_cache,
                publish_tiptilt=args.publish_tiptilt,
                ttl_sec=args.ttl_sec,
                frequency_mhz=args.frequency_mhz,
                publish_all_frequencies=args.publish_all_frequencies,
                az_step_deg=args.az_step_deg,
                el_step_deg=args.el_step_deg,
                el_min_deg=args.el_min_deg,
            )
            last_processed = key
        except Exception:
            LOG.exception("Failed processing %s", fz)
            # Leave last_processed unchanged so the next poll can retry.

    if args.process_existing:
        maybe_process_latest()

    LOG.info(
        "Watching %s for latest YYYY-MM-DD/*.fits.fz (by name timestamp) every %.1fs; "
        "output-dir=%s warp=%s",
        watch_dir,
        args.poll_seconds,
        output_dir,
        warp_filename,
    )
    while True:
        maybe_process_latest()
        time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Watch the latest YYYY-MM-DD day under --watch-dir for *.fits.fz, "
            "funpack, calcflow, write tip-tilt warp NPZ, optionally publish via "
            "mnc.tiptilt.set_tiptilt."
        )
    )
    p.add_argument(
        "--watch-dir",
        required=True,
        help=(
            "Parent directory whose YYYY-MM-DD (UT) subdirectories contain "
            "*.fits.fz; only the latest day is polled"
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for unpacked .fits and the warp map",
    )
    p.add_argument(
        "--warp-filename",
        required=True,
        help='Warp map filename inside --output-dir (e.g. warp_55MHz.npz)',
    )
    p.add_argument(
        "--funpack",
        default="funpack",
        help="funpack executable (default: funpack on PATH)",
    )
    p.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Directory poll interval (default: 5)",
    )
    p.add_argument(
        "--settle-seconds",
        type=float,
        default=10.0,
        help="Require unchanged file size this long before processing (default: 10)",
    )
    p.add_argument(
        "--stable-timeout",
        type=float,
        default=3600.0,
        help="Max seconds to wait for a file to finish writing (default: 3600)",
    )
    p.add_argument(
        "--process-existing",
        action="store_true",
        help="Process *.fits.fz already present at startup",
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
        help="Optional square resample size for calcflow (faster GPU smoke tests)",
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
        "--horizon-elevation-deg",
        type=float,
        default=10.0,
        help=(
            "Elevation cut (deg) for calcflow horizon_mask via horizon_r_normalized "
            "(default: 10; raise to exclude more near-horizon signal from the flow solve)"
        ),
    )
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
        "--publish-all-frequencies",
        action="store_true",
        help=(
            "Publish the same NPZ path under every OVRO-LWA subband MHz key "
            "(BeamTracker tip-tilt catalog); overrides --frequency-mhz"
        ),
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
    run_loop(args)


if __name__ == "__main__":
    main()

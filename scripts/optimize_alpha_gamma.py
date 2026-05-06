#!/usr/bin/env python3
"""
Grid-evaluate :func:`image_plane_correction.flow.calcflow` and
:mod:`image_plane_correction.flow_metrics` over FITS images and ``(alpha, gamma)`` pairs.

Example::

    PYTHONPATH=src python scripts/optimize_alpha_gamma.py \\
        --images obs.fits --cleaned \\
        --alphas 1.3 --gammas 150 200 \\
        --output-json metrics.json --output-csv metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from image_plane_correction.data import fits_image  # noqa: E402
from image_plane_correction.flow import calcflow, horizon_r_normalized  # noqa: E402
from image_plane_correction import flow_metrics  # noqa: E402

SCHEMA_VERSION = "image_plane_correction.optimize_alpha_gamma.v1"


def _load_paths_from_file(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def _parse_float_csv(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(x) for x in parts]


def _disk_mask_weights(n: int, horizon_r: float) -> np.ndarray:
    """Match ``util.circular_mask``: radius ``N * r * 0.5`` in pixel units from center."""
    idx = np.indices((n, n)).transpose(1, 2, 0).astype(np.float64)
    idx = idx - n / 2.0
    dist = np.sqrt(idx[..., 0] ** 2 + idx[..., 1] ** 2)
    inside = dist < n * horizon_r * 0.5
    return inside.astype(np.float64)


def _shift_stats(flow: Any) -> dict[str, float]:
    offsets = np.nan_to_num(np.asarray(flow.offsets))
    mag = np.linalg.norm(offsets, axis=-1)
    return {
        "shift_mean": float(np.nanmean(mag)),
        "shift_p05": float(np.nanpercentile(mag, 5)),
        "shift_p50": float(np.nanpercentile(mag, 50)),
        "shift_p95": float(np.nanpercentile(mag, 95)),
    }


def _attach_metrics(row: dict[str, Any], smap: dict[str, Any]) -> None:
    """Flatten structure_score dict into row keys ``metrics_*``."""
    for k, v in smap.items():
        key = f"metrics_{k}"
        if isinstance(v, tuple):
            row[key] = [float(x) if isinstance(x, (float, np.floating)) else x for x in v]
        elif isinstance(v, (float, np.floating)):
            row[key] = float(v)
        elif isinstance(v, (np.integer, int)):
            row[key] = int(v)
        else:
            row[key] = v


@dataclass
class ImageSpec:
    image: str
    psf: str | None
    reference_sky_fn: str | None


def _parse_specs(args: argparse.Namespace) -> list[ImageSpec]:
    images = list(args.images)
    if args.image_list:
        images.extend(_load_paths_from_file(Path(args.image_list)))

    if not images:
        raise SystemExit("No images: pass --images and/or --image-list.")

    ref_fns: list[str | None]
    if args.reference_sky_fns:
        ref_fns = list(args.reference_sky_fns)
        if len(ref_fns) != len(images):
            raise SystemExit("--reference-sky-fns must match --images length.")
    elif args.reference_sky_fn:
        ref_fns = [args.reference_sky_fn] * len(images)
    else:
        ref_fns = [None] * len(images)

    psfs: list[str | None]
    if args.cleaned:
        psfs = [None] * len(images)
    else:
        if args.psfs:
            psfs = list(args.psfs)
            if len(psfs) != len(images):
                raise SystemExit("--psfs must have the same length as --images.")
        elif args.psf:
            psfs = [args.psf] * len(images)
        else:
            psfs = [None] * len(images)

    out_specs: list[ImageSpec] = []
    for im, pf, rf in zip(images, psfs, ref_fns):
        if not args.cleaned and pf is None and rf is None:
            raise SystemExit(
                "Each image needs --cleaned, or a PSF (--psf / --psfs), "
                "or a reference sky (--reference-sky-fn / --reference-sky-fns)."
            )
        out_specs.append(ImageSpec(image=im, psf=pf, reference_sky_fn=rf))
    return out_specs


def _alpha_gamma_grid(args: argparse.Namespace) -> list[tuple[float, float]]:
    alphas = _parse_float_csv(args.alphas) if args.alphas else []
    gammas = _parse_float_csv(args.gammas) if args.gammas else []
    if not alphas or not gammas:
        raise SystemExit("Provide non-empty --alphas and --gammas (comma-separated floats).")
    return [(a, g) for a in alphas for g in gammas]


def evaluate_one(
    spec: ImageSpec,
    alpha: float,
    gamma: float,
    *,
    cleaned: bool,
    band_deg: tuple[float, float],
    structure_mask: str,
    horizon_elevation_deg: float | None,
    catalog: str,
    catalog_path: str,
    preprocess_weight: float,
    scale_factor: float,
    use_best_pb_model: bool,
    bright_source_flux_qa: bool,
    bright_source_flux_qa_count: int,
    qa: bool,
    quiet: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image": spec.image,
        "psf": spec.psf,
        "reference_sky_fn": spec.reference_sky_fn,
        "alpha": alpha,
        "gamma": gamma,
        "cleaned": cleaned,
        "error": None,
    }

    kwargs = dict(
        image_fn=spec.image,
        cleaned=cleaned,
        qa=qa,
        write=False,
        catalog=catalog,
        catalog_path=catalog_path,
        preprocess_weight=preprocess_weight,
        horizon_elevation_deg=horizon_elevation_deg,
        alpha=alpha,
        gamma=gamma,
        scale_factor=scale_factor,
        use_best_pb_model=use_best_pb_model,
        bright_source_flux_qa=bright_source_flux_qa,
        bright_source_flux_qa_count=bright_source_flux_qa_count,
    )
    if spec.reference_sky_fn is not None:
        kwargs["reference_sky_fn"] = spec.reference_sky_fn
        kwargs["psf_fn"] = None
    else:
        kwargs["psf_fn"] = spec.psf

    buf = StringIO()
    try:
        if quiet:
            with redirect_stdout(buf):
                _image, _ref, flow, _dewarped, qa_passed = calcflow(**kwargs)
        else:
            _image, _ref, flow, _dewarped, qa_passed = calcflow(**kwargs)

        _, imwcs = fits_image(spec.image)
        n = int(_image.shape[0])
        horizon_r = horizon_r_normalized(imwcs, n=n, horizon_elevation_deg=horizon_elevation_deg)

        mask: np.ndarray | None = None
        if structure_mask == "disk":
            mask = _disk_mask_weights(n, horizon_r)
        elif structure_mask == "none":
            mask = None
        else:
            raise ValueError(f"Unknown structure_mask {structure_mask!r}")

        smap = flow_metrics.structure_score(flow, imwcs, band_deg=band_deg, mask=mask)
        _attach_metrics(row, smap)
        row.update(_shift_stats(flow))
        row["runqa_score"] = int(1 if qa_passed else 0)
        row["qa_passed"] = bool(qa_passed)
        row["horizon_r"] = float(horizon_r)
        row["calcflow_stdout"] = buf.getvalue() if quiet else ""
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["qa_passed"] = False
        row["runqa_score"] = 0

    return row


def _aggregate(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
    min_qa_rate: float,
) -> dict[str, Any]:
    """Summarize by (alpha, gamma): medians, QA rate, bootstrap CI on median structure_score."""
    pairs: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("error"):
            continue
        key = (float(r["alpha"]), float(r["gamma"]))
        pairs.setdefault(key, []).append(r)

    by_pair: list[dict[str, Any]] = []
    for (a, g), grp in sorted(pairs.items()):
        scores = np.array([float(x["metrics_structure_score"]) for x in grp], dtype=np.float64)
        qa_hits = np.array([int(x.get("runqa_score", 0)) for x in grp], dtype=np.int64)
        rec = {
            "alpha": a,
            "gamma": g,
            "n_images": len(grp),
            "median_structure_score": float(np.median(scores)) if scores.size else float("nan"),
            "mean_structure_score": float(np.mean(scores)) if scores.size else float("nan"),
            "qa_pass_rate": float(np.mean(qa_hits)) if qa_hits.size else 0.0,
            "median_shift_p50": float(np.median([float(x["shift_p50"]) for x in grp]))
            if grp
            else float("nan"),
        }

        if bootstrap_samples > 0 and len(grp) > 1:
            meds = []
            idx = np.arange(len(grp))
            for _ in range(bootstrap_samples):
                samp = rng.choice(idx, size=len(grp), replace=True)
                drawn = scores[samp]
                meds.append(float(np.median(drawn)))
            meds_arr = np.array(meds, dtype=np.float64)
            rec["bootstrap_median_structure_score_p05"] = float(np.nanpercentile(meds_arr, 5))
            rec["bootstrap_median_structure_score_p95"] = float(np.nanpercentile(meds_arr, 95))
        by_pair.append(rec)

    feasible = [x for x in by_pair if x["qa_pass_rate"] >= min_qa_rate - 1e-9]
    ranked = sorted(
        feasible,
        key=lambda x: (x["median_structure_score"], x["qa_pass_rate"]),
        reverse=True,
    )
    recommended = ranked[0] if ranked else None

    return {"by_pair": by_pair, "recommended": recommended}


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", nargs="*", default=[], help="FITS image paths")
    p.add_argument("--image-list", type=str, default=None, help="Text file with one image path per line")
    p.add_argument("--cleaned", action="store_true", help="Use cleaned/beam-from-header mode")
    p.add_argument("--psf", type=str, default=None, help="Single PSF FITS applied to all images")
    p.add_argument("--psfs", nargs="*", default=None, help="PSF path per image (same order as combined image list)")
    p.add_argument("--reference-sky-fn", type=str, default=None, help="Optional reference sky FITS for all images")
    p.add_argument(
        "--reference-sky-fns",
        nargs="*",
        default=None,
        help="Optional reference sky FITS per image",
    )
    p.add_argument("--alphas", type=str, required=True, help="Comma-separated alpha values")
    p.add_argument("--gammas", type=str, required=True, help="Comma-separated gamma values")
    p.add_argument(
        "--band-deg-min",
        type=float,
        default=20.0,
        help="Minimum wavelength (degrees) for structure_score band",
    )
    p.add_argument(
        "--band-deg-max",
        type=float,
        default=100.0,
        help="Maximum wavelength (degrees) for structure_score band",
    )
    p.add_argument(
        "--structure-mask",
        choices=("none", "disk"),
        default="disk",
        help="Mask for flow_metrics FFT (disk matches preprocess horizon disk footprint)",
    )
    p.add_argument("--horizon-elevation-deg", type=float, default=10.0)
    p.add_argument("--catalog", type=str, default="VLSSR")
    p.add_argument("--catalog-path", type=str, default="/home/claw/vlssr_radecpeak_unresolved.txt")
    p.add_argument("--preprocess-weight", type=float, default=1.5)
    p.add_argument("--scale-factor", type=float, default=0.7)
    p.add_argument("--use-best-pb-model", action="store_true")
    p.add_argument("--bright-source-flux-qa", action="store_true")
    p.add_argument("--bright-source-flux-qa-count", type=int, default=10)
    p.add_argument("--qa", action="store_true", default=True)
    p.add_argument("--no-qa", action="store_false", dest="qa", help="Disable residual QA inside calcflow")
    p.add_argument("--quiet", action="store_true", help="Suppress calcflow/processing stdout")
    p.add_argument("--bootstrap", type=int, default=400, help="Bootstrap resamples for median CI (0 to disable)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--min-qa-rate",
        type=float,
        default=0.0,
        help="Minimum qa_pass_rate across images when choosing recommended pair",
    )
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--output-csv", type=str, default=None)

    args = p.parse_args(list(argv) if argv is not None else None)

    specs = _parse_specs(args)
    grid = _alpha_gamma_grid(args)
    band_deg = (float(args.band_deg_min), float(args.band_deg_max))

    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for alpha, gamma in grid:
            row = evaluate_one(
                spec,
                alpha,
                gamma,
                cleaned=args.cleaned,
                band_deg=band_deg,
                structure_mask=args.structure_mask,
                horizon_elevation_deg=args.horizon_elevation_deg,
                catalog=args.catalog,
                catalog_path=args.catalog_path,
                preprocess_weight=args.preprocess_weight,
                scale_factor=args.scale_factor,
                use_best_pb_model=args.use_best_pb_model,
                bright_source_flux_qa=args.bright_source_flux_qa,
                bright_source_flux_qa_count=args.bright_source_flux_qa_count,
                qa=args.qa,
                quiet=args.quiet,
            )
            rows.append(row)

    aggregate = _aggregate(
        rows,
        bootstrap_samples=max(0, int(args.bootstrap)),
        rng=rng,
        min_qa_rate=float(args.min_qa_rate),
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "alphas": _parse_float_csv(args.alphas),
            "gammas": _parse_float_csv(args.gammas),
            "band_deg": list(band_deg),
            "structure_mask": args.structure_mask,
            "horizon_elevation_deg": args.horizon_elevation_deg,
            "cleaned": args.cleaned,
            "qa": args.qa,
            "catalog": args.catalog,
            "catalog_path": args.catalog_path,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "min_qa_rate": args.min_qa_rate,
        },
        "rows": rows,
        "aggregate": aggregate,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            raise SystemExit("No rows to write.")
        fieldnames = list(rows[0].keys())
        for r in rows[1:]:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                flat = {}
                for k, v in r.items():
                    if isinstance(v, (list, dict)):
                        flat[k] = json.dumps(v)
                    elif isinstance(v, bool):
                        flat[k] = int(v)
                    elif v is None:
                        flat[k] = ""
                    else:
                        flat[k] = v
                w.writerow(flat)

    print(f"Wrote {out_json}", file=sys.stderr)
    if args.output_csv:
        print(f"Wrote {args.output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

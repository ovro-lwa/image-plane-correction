#!/usr/bin/env python3
"""
Grid-evaluate :func:`image_plane_correction.flow.calcflow` and
:mod:`image_plane_correction.flow_metrics` over FITS images and ``(alpha, gamma)`` pairs.

Supports explicit grids or a coarse **log-space** search (optionally followed by
bounded refinement on ``log(alpha), log(gamma)`` via SciPy ``L-BFGS-B``).

Example::

    PYTHONPATH=src python scripts/optimize_alpha_gamma.py \\
        --images obs.fits --cleaned \\
        --alphas 1.3 --gammas 150 200 \\
        --output-json metrics.json --output-csv metrics.csv

Coarse search + composite objective (see ``--w-struct`` / ``--w-qa``)::

    PYTHONPATH=src python scripts/optimize_alpha_gamma.py --search \\
        --images obs.fits --cleaned --output-json metrics.json
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
from scipy.optimize import minimize

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from image_plane_correction.data import fits_image  # noqa: E402
from image_plane_correction.flow import calcflow, horizon_r_normalized  # noqa: E402
from image_plane_correction import flow_metrics  # noqa: E402

SCHEMA_VERSION = "image_plane_correction.optimize_alpha_gamma.v2"


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


def composite_objective_from_row(
    row: dict[str, Any],
    *,
    w_struct: float,
    w_qa: float,
    soft_qa: bool,
) -> float | None:
    """
    Per-image composite objective: ``w_struct * structure_score + w_qa * qa_scalar``.

    When QA fails and ``soft_qa`` is false, returns ``None`` (drop from median aggregate).
    With ``soft_qa``, ``qa_scalar`` is 0 so the row still contributes structure term only.
    """
    if row.get("error"):
        return None
    raw = row.get("metrics_structure_score")
    if raw is None:
        return None
    struct = float(raw)
    if not np.isfinite(struct):
        return None
    qa_ok = bool(row.get("qa_passed"))
    if not qa_ok and not soft_qa:
        return None
    qa_scalar = 1.0 if qa_ok else 0.0
    return float(w_struct * struct + w_qa * qa_scalar)


def _coarse_geom_grid(
    alpha_min: float,
    alpha_max: float,
    n_alpha: int,
    gamma_min: float,
    gamma_max: float,
    n_gamma: int,
) -> list[tuple[float, float]]:
    alphas = np.geomspace(float(alpha_min), float(alpha_max), int(n_alpha))
    gammas = np.geomspace(float(gamma_min), float(gamma_max), int(n_gamma))
    return [(float(a), float(g)) for a in alphas for g in gammas]


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


def _positive_bounds_from_grid(grid: Sequence[tuple[float, float]], *, dim: int) -> tuple[float, float]:
    """Minimum and maximum along ``dim`` for refinement; expand degenerate intervals slightly."""
    vals = sorted({float(g[dim]) for g in grid})
    lo, hi = float(vals[0]), float(vals[-1])
    if lo <= 0 or hi <= 0:
        raise ValueError("alpha and gamma must be positive for log-space refinement.")
    if hi <= lo or np.isclose(lo, hi):
        lo = lo * 0.9
        hi = hi * 1.1
        if lo <= 0:
            lo = float(vals[0]) * 0.99
    return lo, hi


def _resolve_parameter_grid(args: argparse.Namespace) -> list[tuple[float, float]]:
    if getattr(args, "search", False):
        return _coarse_geom_grid(
            args.alpha_min,
            args.alpha_max,
            args.alpha_steps,
            args.gamma_min,
            args.gamma_max,
            args.gamma_steps,
        )
    alphas = _parse_float_csv(args.alphas) if args.alphas else []
    gammas = _parse_float_csv(args.gammas) if args.gammas else []
    if not alphas or not gammas:
        raise SystemExit(
            "Provide non-empty --alphas and --gammas (comma-separated floats), or pass --search."
        )
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
    w_struct: float = 1.0,
    w_qa: float = 1.0,
    soft_qa: bool = False,
    phase: str = "grid",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image": spec.image,
        "psf": spec.psf,
        "reference_sky_fn": spec.reference_sky_fn,
        "alpha": alpha,
        "gamma": gamma,
        "cleaned": cleaned,
        "error": None,
        "phase": phase,
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
        row["composite_objective"] = composite_objective_from_row(
            row, w_struct=w_struct, w_qa=w_qa, soft_qa=soft_qa
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["qa_passed"] = False
        row["runqa_score"] = 0
        row["composite_objective"] = None

    return row


def _rows_for_aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer ``refine_best`` over ``grid`` when the same image was evaluated at the same (alpha, gamma)."""
    phase_rank = {"refine_best": 2, "grid": 1}
    best: dict[tuple[str, float, float], dict[str, Any]] = {}
    for r in rows:
        if r.get("error"):
            continue
        ke = (str(r["image"]), float(r["alpha"]), float(r["gamma"]))
        ph = str(r.get("phase", "grid"))
        pr = phase_rank.get(ph, 1)
        old = best.get(ke)
        if old is None or pr > phase_rank.get(str(old.get("phase", "grid")), 1):
            best[ke] = r
    return list(best.values())


def _aggregate(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
    min_qa_rate: float,
) -> dict[str, Any]:
    """
    Summarize by (alpha, gamma): structure metrics, composite objective, QA rate.

    ``recommended`` maximizes median composite objective when any pair has finite
    composites; otherwise falls back to median ``structure_score`` only.
    Tie-break: higher QA pass rate, then lower median in-band curl/div ratio.
    """
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
        comps = []
        curls = []
        for x in grp:
            co = x.get("composite_objective")
            if co is not None and np.isfinite(float(co)):
                comps.append(float(co))
            else:
                comps.append(float("nan"))
            cr = x.get("metrics_curl_to_div_ratio_band")
            if cr is not None and np.isfinite(float(cr)):
                curls.append(float(cr))
            else:
                curls.append(float("nan"))
        comps_arr = np.asarray(comps, dtype=np.float64)
        curls_arr = np.asarray(curls, dtype=np.float64)
        med_comp = float("nan")
        mean_comp = float("nan")
        med_curl = float("nan")
        if comps_arr.size:
            if np.any(np.isfinite(comps_arr)):
                med_comp = float(np.nanmedian(comps_arr))
            if np.any(np.isfinite(comps_arr)):
                mean_comp = float(np.nanmean(comps_arr[np.isfinite(comps_arr)]))
        if curls_arr.size and np.any(np.isfinite(curls_arr)):
            med_curl = float(np.nanmedian(curls_arr))
        rec: dict[str, Any] = {
            "alpha": a,
            "gamma": g,
            "n_images": len(grp),
            "median_structure_score": float(np.median(scores)) if scores.size else float("nan"),
            "mean_structure_score": float(np.mean(scores)) if scores.size else float("nan"),
            "median_composite_objective": med_comp,
            "mean_composite_objective": mean_comp,
            "median_curl_to_div_ratio_band": med_curl,
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
    has_composite = any(np.isfinite(x["median_composite_objective"]) for x in feasible)

    def rank_key(x: dict[str, Any]) -> tuple[float, float, float]:
        if has_composite:
            mc = x["median_composite_objective"]
            mc_part = float(mc) if np.isfinite(mc) else float("-inf")
        else:
            mc_part = float(x["median_structure_score"]) if np.isfinite(x["median_structure_score"]) else float("-inf")
        qa_part = float(x["qa_pass_rate"])
        curl = x["median_curl_to_div_ratio_band"]
        curl_penalty = float(curl) if np.isfinite(curl) else float("inf")
        return (mc_part, qa_part, -curl_penalty)

    ranked = sorted(feasible, key=rank_key, reverse=True)
    recommended = ranked[0] if ranked else None

    return {"by_pair": by_pair, "recommended": recommended, "ranked_by_composite": bool(has_composite)}


def _eval_kw(args: argparse.Namespace, band_deg: tuple[float, float]) -> dict[str, Any]:
    return {
        "cleaned": args.cleaned,
        "band_deg": band_deg,
        "structure_mask": args.structure_mask,
        "horizon_elevation_deg": args.horizon_elevation_deg,
        "catalog": args.catalog,
        "catalog_path": args.catalog_path,
        "preprocess_weight": args.preprocess_weight,
        "scale_factor": args.scale_factor,
        "use_best_pb_model": args.use_best_pb_model,
        "bright_source_flux_qa": args.bright_source_flux_qa,
        "bright_source_flux_qa_count": args.bright_source_flux_qa_count,
        "qa": args.qa,
        "quiet": args.quiet,
        "w_struct": float(args.w_struct),
        "w_qa": float(args.w_qa),
        "soft_qa": bool(args.soft_qa),
    }


def _median_composite_across_specs(
    specs: list[ImageSpec],
    alpha: float,
    gamma: float,
    *,
    phase: str,
    **eval_kw: Any,
) -> float:
    comps: list[float] = []
    for spec in specs:
        row = evaluate_one(spec, alpha, gamma, phase=phase, **eval_kw)
        co = row.get("composite_objective")
        if co is not None and np.isfinite(float(co)):
            comps.append(float(co))
        else:
            comps.append(float("nan"))
    return float(np.nanmedian(np.asarray(comps, dtype=np.float64)))


def _run_refine(
    specs: list[ImageSpec],
    *,
    best_alpha: float,
    best_gamma: float,
    alpha_bounds: tuple[float, float],
    gamma_bounds: tuple[float, float],
    maxiter: int,
    eval_kw: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    amin, amax = alpha_bounds
    gmin, gmax = gamma_bounds
    bounds_log = [(np.log(amin), np.log(amax)), (np.log(gmin), np.log(gmax))]
    x0 = np.clip(
        np.log(np.asarray([best_alpha, best_gamma], dtype=np.float64)),
        [bounds_log[0][0], bounds_log[1][0]],
        [bounds_log[0][1], bounds_log[1][1]],
    )

    n_ev = {"n": 0}

    def objective(log_xy: np.ndarray) -> float:
        n_ev["n"] += 1
        a = float(np.exp(log_xy[0]))
        g = float(np.exp(log_xy[1]))
        med = _median_composite_across_specs(specs, a, g, phase="refine", **eval_kw)
        if not np.isfinite(med):
            return 1e100
        return float(-med)

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds_log, options={"maxiter": int(maxiter)})
    fin_a = float(np.exp(res.x[0]))
    fin_g = float(np.exp(res.x[1]))
    final_med = _median_composite_across_specs(specs, fin_a, fin_g, phase="refine_verify", **eval_kw)

    refine_rows: list[dict[str, Any]] = []
    for spec in specs:
        refine_rows.append(evaluate_one(spec, fin_a, fin_g, phase="refine_best", **eval_kw))

    summary: dict[str, Any] = {
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", -1)),
        "nfev_optimizer": int(getattr(res, "nfev", -1)),
        "n_callback_evals": int(n_ev["n"]),
        "final_alpha": fin_a,
        "final_gamma": fin_g,
        "final_median_composite_objective": final_med,
        "scipy_fun": float(res.fun),
    }
    return summary, refine_rows


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
    p.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated alpha values (ignored if --search)",
    )
    p.add_argument(
        "--gammas",
        type=str,
        default=None,
        help="Comma-separated gamma values (ignored if --search)",
    )
    p.add_argument(
        "--search",
        action="store_true",
        help="Coarse log-space grid over alpha/gamma (use --alpha-min/max/steps, --gamma-min/max/steps)",
    )
    p.add_argument("--alpha-min", type=float, default=0.1, help="Coarse search: minimum alpha")
    p.add_argument("--alpha-max", type=float, default=10.0, help="Coarse search: maximum alpha")
    p.add_argument("--alpha-steps", type=int, default=5, help="Coarse search: number of alpha samples (geomspace)")
    p.add_argument("--gamma-min", type=float, default=1.0, help="Coarse search: minimum gamma")
    p.add_argument("--gamma-max", type=float, default=1000.0, help="Coarse search: maximum gamma")
    p.add_argument("--gamma-steps", type=int, default=5, help="Coarse search: number of gamma samples (geomspace)")
    p.add_argument(
        "--w-struct",
        type=float,
        default=1.0,
        help="Weight on metrics.structure_score in composite objective",
    )
    p.add_argument(
        "--w-qa",
        type=float,
        default=1.0,
        help="Weight on QA scalar (1 pass / 0 fail) in composite objective",
    )
    p.add_argument(
        "--soft-qa",
        action="store_true",
        help="When QA fails, still score structure term (qa scalar 0); default is hard exclude",
    )
    p.add_argument(
        "--refine",
        action="store_true",
        help="After coarse grid, run SciPy L-BFGS-B on log(alpha),log(gamma) inside coarse bounds",
    )
    p.add_argument("--refine-maxiter", type=int, default=40, help="Max iterations for L-BFGS-B refinement")
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

    if args.search and (args.alphas or args.gammas):
        raise SystemExit("Use either --search or explicit --alphas/--gammas, not both.")
    if not args.search:
        if not args.alphas or not args.gammas:
            raise SystemExit(
                "Provide --alphas and --gammas, or enable --search for a coarse log-space grid."
            )
    else:
        if args.alpha_min <= 0 or args.alpha_max <= 0 or args.gamma_min <= 0 or args.gamma_max <= 0:
            raise SystemExit("--alpha-min/max and --gamma-min/max must be positive for --search.")
        if args.alpha_min >= args.alpha_max or args.gamma_min >= args.gamma_max:
            raise SystemExit("--search requires min < max for both alpha and gamma ranges.")
        if args.alpha_steps < 1 or args.gamma_steps < 1:
            raise SystemExit("--alpha-steps and --gamma-steps must be >= 1.")

    specs = _parse_specs(args)
    grid = _resolve_parameter_grid(args)
    band_deg = (float(args.band_deg_min), float(args.band_deg_max))
    eval_kw = _eval_kw(args, band_deg)

    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for alpha, gamma in grid:
            row = evaluate_one(spec, alpha, gamma, phase="grid", **eval_kw)
            rows.append(row)

    aggregate = _aggregate(
        _rows_for_aggregate(rows),
        bootstrap_samples=max(0, int(args.bootstrap)),
        rng=rng,
        min_qa_rate=float(args.min_qa_rate),
    )

    refinement_summary: dict[str, Any] | None = None
    if args.refine:
        rec = aggregate.get("recommended")
        mc = rec.get("median_composite_objective") if rec else float("nan")
        if rec is None:
            print("Skipping --refine: no feasible coarse candidate.", file=sys.stderr)
        elif not np.isfinite(float(mc)):
            print(
                "Skipping --refine: coarse best has non-finite median composite objective.",
                file=sys.stderr,
            )
        else:
            alpha_bounds = _positive_bounds_from_grid(grid, dim=0)
            gamma_bounds = _positive_bounds_from_grid(grid, dim=1)
            refinement_summary, refine_rows = _run_refine(
                specs,
                best_alpha=float(rec["alpha"]),
                best_gamma=float(rec["gamma"]),
                alpha_bounds=alpha_bounds,
                gamma_bounds=gamma_bounds,
                maxiter=max(1, int(args.refine_maxiter)),
                eval_kw=eval_kw,
            )
            rows.extend(refine_rows)
            aggregate = _aggregate(
                _rows_for_aggregate(rows),
                bootstrap_samples=max(0, int(args.bootstrap)),
                rng=rng,
                min_qa_rate=float(args.min_qa_rate),
            )

    if args.search:
        grid_alphas = sorted({float(a) for a, _ in grid})
        grid_gammas = sorted({float(g) for _, g in grid})
        search_bounds = {
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "alpha_steps": args.alpha_steps,
            "gamma_min": args.gamma_min,
            "gamma_max": args.gamma_max,
            "gamma_steps": args.gamma_steps,
        }
    else:
        grid_alphas = _parse_float_csv(args.alphas) if args.alphas else []
        grid_gammas = _parse_float_csv(args.gammas) if args.gammas else []
        search_bounds = {
            "alpha_min": None,
            "alpha_max": None,
            "alpha_steps": None,
            "gamma_min": None,
            "gamma_max": None,
            "gamma_steps": None,
        }

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "search": bool(args.search),
            **search_bounds,
            "alphas": grid_alphas,
            "gammas": grid_gammas,
            "grid_size": len(grid),
            "band_deg": list(band_deg),
            "structure_mask": args.structure_mask,
            "horizon_elevation_deg": args.horizon_elevation_deg,
            "cleaned": args.cleaned,
            "qa": args.qa,
            "w_struct": args.w_struct,
            "w_qa": args.w_qa,
            "soft_qa": args.soft_qa,
            "refine": bool(args.refine),
            "refine_maxiter": args.refine_maxiter,
            "catalog": args.catalog,
            "catalog_path": args.catalog_path,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "min_qa_rate": args.min_qa_rate,
        },
        "rows": rows,
        "aggregate": aggregate,
        "refinement": refinement_summary,
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

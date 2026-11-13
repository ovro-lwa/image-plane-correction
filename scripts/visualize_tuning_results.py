#!/usr/bin/env python3
"""
Plot heatmaps from ``optimize_alpha_gamma`` JSON (schema v2).

Example::

    python scripts/visualize_tuning_results.py tuning_results.json -o tuning_plots.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _nearest_index(values: list[float], x: float) -> int:
    arr = np.asarray(values, dtype=np.float64)
    return int(np.argmin(np.abs(arr - float(x))))


def _build_matrices(data: dict) -> tuple:
    p = data["params"]
    alphas = [float(a) for a in p["alphas"]]
    gammas = [float(g) for g in p["gammas"]]
    agg = data.get("aggregate") or data.get("summary") or {}
    rows = agg.get("by_pair") or agg.get("by_alpha_gamma") or []

    def key(a: float, g: float) -> tuple[float, float]:
        return (float(a), float(g))

    lookup: dict[tuple[float, float], dict] = {}
    for r in rows:
        lookup[key(r["alpha"], r["gamma"])] = r

    na, ng = len(alphas), len(gammas)
    struct = np.full((na, ng), np.nan)
    curl_ratio = np.full((na, ng), np.nan)
    shift_p50 = np.full((na, ng), np.nan)
    qa_rate = np.full((na, ng), np.nan)
    composite = np.full((na, ng), np.nan)

    for i, a in enumerate(alphas):
        for j, g in enumerate(gammas):
            r = lookup.get(key(a, g))
            if not r:
                continue
            struct[i, j] = r.get("median_structure_score")
            curl_ratio[i, j] = r.get("median_curl_to_div_ratio_band")
            shift_p50[i, j] = r.get("median_shift_p50")
            qa_rate[i, j] = r.get("qa_pass_rate")
            mc = r.get("median_composite_objective")
            if mc is not None and isinstance(mc, (int, float)) and np.isfinite(mc):
                composite[i, j] = float(mc)

    rec = agg.get("recommended") or {}
    return alphas, gammas, struct, curl_ratio, shift_p50, qa_rate, composite, rec, agg


def _plot(
    path_in: Path,
    path_out: Path,
    *,
    annotate: bool,
) -> None:
    data = _load(path_in)
    alphas, gammas, struct, curl_ratio, shift_p50, qa_rate, composite, rec, agg = _build_matrices(data)

    ranked_comp = bool(agg.get("ranked_by_composite"))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    ia = _nearest_index(alphas, rec["alpha"]) if rec.get("alpha") is not None else None
    ig = _nearest_index(gammas, rec["gamma"]) if rec.get("gamma") is not None else None

    subtitle = (
        f"recommended α={rec.get('alpha')!s}, γ={rec.get('gamma')!s} "
        f"(median structure={rec.get('median_structure_score')!s})"
    )
    fig.suptitle(f"{path_in.name}\n{subtitle}\nranked_by_composite={ranked_comp}", fontsize=11)

    def heatmap(
        ax: plt.Axes,
        Z: np.ndarray,
        title: str,
        *,
        cmap: str = "viridis",
        norm: mcolors.Normalize | None = None,
    ) -> None:
        im = ax.imshow(Z.T, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.3g}" for a in alphas], rotation=45, ha="right")
        ax.set_yticks(range(len(gammas)))
        ax.set_yticklabels([f"{g:.3g}" for g in gammas])
        ax.set_xlabel("alpha")
        ax.set_ylabel("gamma")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
        if ia is not None and ig is not None:
            ax.scatter([ia], [ig], s=220, facecolors="none", edgecolors="red", linewidths=2.2)
        if annotate:
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    v = Z[i, j]
                    if np.isfinite(v):
                        ax.text(i, j, f"{v:.2g}", ha="center", va="center", color="white", fontsize=6)

    heatmap(axes[0, 0], struct, "Median structure score (↑ better)")
    heatmap(
        axes[0, 1],
        curl_ratio,
        "Median curl/div in band (↓ more gradient-like)",
        cmap="magma_r",
    )
    heatmap(axes[1, 0], shift_p50, "Median flow magnitude p50 (pixels)", cmap="cividis")

    comp_finite = np.isfinite(composite).any()
    if comp_finite:
        heatmap(
            axes[1, 1],
            composite,
            "Median composite objective (↑ better)",
            cmap="RdYlGn",
        )
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.55,
            "Composite objective undefined\n(all QA failed or null).\n"
            "Ranking used structure metrics\n(tie-break curl/div, shift).",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
        )
        qa_flat = qa_rate.ravel()
        if qa_flat.size and np.all(qa_flat == 0):
            axes[1, 1].text(
                0.5,
                0.15,
                "qa_pass_rate is 0 for every grid cell.",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=10,
                style="italic",
            )

    plt.tight_layout()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "json_path",
        nargs="?",
        default="tuning_results.json",
        type=Path,
        help="Output JSON from optimize_alpha_gamma.py",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <json stem>_plots.png)",
    )
    ap.add_argument("--annotate", action="store_true", help="Overlay numeric values on cells")
    args = ap.parse_args()
    path_in = args.json_path.resolve()
    path_out = (
        args.output.resolve()
        if args.output is not None
        else path_in.with_name(path_in.stem + "_plots.png")
    )
    _plot(path_in, path_out, annotate=args.annotate)
    print(path_out)


if __name__ == "__main__":
    main()

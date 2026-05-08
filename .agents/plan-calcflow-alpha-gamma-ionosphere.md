# Implementation Plan: Optimize `calcflow` (alpha, gamma) for ionospheric warp structure

---
**Date:** 2026-05-05  
**Author:** AI Assistant  
**Status:** Draft  
**Related Documents:** *(none)*

---

## Overview

`calcflow` solves a dense optical flow field between an observed LWA image and a reference sky using a Brox-style variational model. Two parameters, `alpha` and `gamma`, strongly control the recovered vector-field structure: `alpha` effectively scales the data-term weight (and thus the smoothness-vs-fit trade-off), while `gamma` controls the relative influence of gradient-constancy terms. For ionospheric refraction, we expect **large-scale**, **smooth**, predominantly **curl-free** warps, with characteristic spatial scales from **~20° to ~100°** across the field.

This plan defines a **structure-matching objective** (spectral band + curl/divergence constraints in angular units), builds a reproducible evaluation harness around `calcflow`, and implements a parameter search that chooses `(alpha, gamma)` to match the expected ionospheric warp structure while preserving (or improving) existing residual-based QA.

**Goal:** Choose robust default values (and an optional auto-tuning mode) for `alpha` and `gamma` so the estimated flow is consistent with ionospheric warps on ~20–100° scales and improves `calcflow` dewarping quality across representative datasets.

**Motivation:** Current `alpha=1.3, gamma=150` are fixed defaults (`src/image_plane_correction/flow.py:210-485`) without an explicit connection to ionospheric structure or to the image’s angular scale. This makes results sensitive to dataset differences and risks producing overly small-scale, noisy, or non-physical vector fields.

## Current State Analysis

**Existing Implementation:**
- `src/image_plane_correction/flow.py:210-485` — `calcflow` preprocesses the image/sky pair then computes Brox optical flow via `Flow.brox(..., alpha=alpha, gamma=gamma, ...)` and dewarps by `flow.apply(image)`.
- `src/image_plane_correction/flow.py:119-140` — `Flow.brox` calls `brox_optical_flow(img2, img1, alpha, gamma, ...)` and returns the resulting dense `(u,v)` field.
- `src/image_plane_correction/brox.py:202-280` — `brox_optical_flow` implements a coarse-to-fine Brox solver. The key parameter couplings are:
  - `psi_data = 0.5 * rsqrt(q0^2 + gamma*(q1^2+q2^2) + eps) / alpha` (`prepare_sor_stage_1`), meaning `alpha` scales down the data term, and `gamma` increases the influence of derivative-constancy residuals.
- `src/image_plane_correction/preprocessing.py:35-56` — preprocessing includes horizon masking, nonlinear reweighting of bright sources, blurring, and normalization, all of which affect flow structure.
- `src/image_plane_correction/util.py:868-907` — `runqa` checks (a) shift magnitude statistics and (b) whether percentile residuals vs `reference_sky` improve after dewarping. This is **not** directly tied to the expected ionospheric spatial structure.

**Current Behavior:**
- `alpha`/`gamma` are user-provided knobs but default to constant values in `calcflow` and in cascade wrappers (`flow_cascade73MHz`, `flow_cascade73MHz_phase2`).
- QA is primarily “residual improvement” and does not penalize non-physical flow patterns (e.g., small-scale noise, high curl).

**Current Limitations:**
- No objective that explicitly encodes “ionospheric-like” structure (20–100° scales, smooth, mostly curl-free).
- No automated way to choose or validate `alpha`/`gamma` across datasets with differing pixel scale / projection.
- Parameter choice is not reproducible (no standardized dataset list + metric + selection procedure).

## Desired End State

**New Behavior:**
- A reproducible tuning/evaluation pipeline that:
  - Computes flow for many `(alpha, gamma)` candidates.
  - Scores flows using both **existing dewarp QA** and **structure metrics** expressed in **degrees** (via WCS pixel scale).
  - Runs a **catalog-based astrometric quality check** by measuring apparent source locations in the (raw and dewarped) images and comparing them to a **reference source catalog** (configurable; e.g. `vlssr_radecpeak_unresolved`).
  - Selects a recommended default `(alpha, gamma)` for typical operation and optionally supports per-image “auto-tune” bounded search.

**Success Looks Like:**
- Flow fields show most of their power in angular scales corresponding to **20–100°** and are smoother / less noisy outside that band.
- The flow is **predominantly curl-free** (low vorticity energy vs divergence) at the target scales.
- Dewarping quality does not regress and ideally improves on representative inputs, including:
  - existing residual QA, and
  - **catalog-based positional agreement** against a reference source catalog (median/RMS angular offsets improve after dewarp).

## What We're NOT Doing

- [ ] Implementing a full physical ionospheric forward model (TEC screens, altitude geometry, frequency-dependent refraction) beyond what’s needed to define a robust “expected structure” metric.
- [ ] Changing the Brox solver itself (numerics, pyramid schedule, SOR scheme) in `src/image_plane_correction/brox.py`.
- [ ] Introducing heavy new dependencies (e.g. probabilistic programming frameworks); we will use existing `numpy/scipy/jax/astropy`.
- [ ] Building a UI for tuning (interactive dashboards).

**Rationale:** The goal is to tune and validate `alpha/gamma` against structure expectations with minimal disruption and maximum reproducibility.

## Implementation Approach

**Technical Strategy:**
- Define a **structure score** computed from the estimated flow:
  1. Convert flow field to **divergence** and **curl** maps (finite differences).
  2. Compute their 2D power spectra and radially average to a 1D spectrum vs spatial frequency.
  3. Convert spatial frequency (cycles/pixel) to angular frequency (cycles/degree) using WCS pixel scale (at image center) from `astropy.wcs.utils.proj_plane_pixel_scales`.
  4. Score how much power lies in the target **20–100°** band, penalize out-of-band power, and penalize curl energy relative to divergence within-band.
- Define a **catalog-based astrometric QC** computed from measured source locations in the image:
  1. Detect a set of compact sources in the image (preferably after the existing preprocessing used for flow; optionally on a lightly smoothed version of the raw/dewarped image).
  2. Measure each source’s sky position (RA/Dec) using WCS (peak pixel → world coords; optionally refine with centroid/2D Gaussian fit).
  3. Load a **reference source catalog** (configurable; ideally dominated by compact/unresolved sources) and select reference sources within the image footprint and above a configurable flux/quality cut.
  4. Cross-match measured sources to catalog sources using `astropy.coordinates.SkyCoord.match_to_catalog_sky`.
  5. Compute robust summary stats for the match set (e.g. median separation, RMS separation, 90th percentile separation), and a match yield metric (matched fraction / count).
  6. Define a scalar `catalog_qc_score` that rewards lower separations and penalizes low match yield; compute this **before and after dewarp** and use the improvement (or absolute post-dewarp score) in selection.
- Combine structure score with existing `runqa` output and **catalog-based astrometric QC** as a composite objective.
- Implement a two-stage optimizer:
  - **Stage A (global coarse search):** log-grid / Latin hypercube over `(alpha, gamma)` to find a good region robustly across multiple images.
  - **Stage B (local refinement):** bounded derivative-free refinement (e.g., Nelder–Mead on log-parameters using SciPy) on a smaller subset to stabilize the final recommended defaults.

**Key Architectural Decisions:**
1. **Decision:** Encode “ionospheric-like” structure using **spectral bandpower** + **curl/divergence ratio** metrics in angular units.
   - **Rationale:** Ionospheric refraction is expected to be smooth and largely potential (gradient) flow; the dominant spatial scales are a primary physical prior (20–100°).
   - **Trade-offs:** This is a proxy for physics rather than a full forward model; it may not capture anisotropic or localized events perfectly.
   - **Alternatives considered:** (a) only residual-based QA; (b) fit a parametric covariance kernel and match kernel hyperparameters; (c) use learned priors. These are either not structure-specific, harder to interpret, or require more infrastructure.

2. **Decision:** Use WCS-derived pixel scale at/near the image center as the angular conversion for structure metrics.
   - **Rationale:** It makes the metric portable across images with different pixel scales and avoids hard-coding “degrees per pixel.”
   - **Trade-offs:** Wide-field projections vary across the field; center-scale is an approximation. We mitigate by using broad bands (20–100°) and by masking horizon regions already used in preprocessing.
   - **Alternatives considered:** per-pixel Jacobian mapping (more accurate, much heavier).

3. **Decision:** Add a catalog-based astrometric QC against a configurable reference source catalog using measured source positions and cross-matching.
   - **Rationale:** Residual-based QA can improve even when astrometry drifts; a direct positional check anchors tuning to an externally meaningful reference and catches non-physical warps.
   - **Trade-offs:** Requires source detection robustness and careful catalog/footprint filtering; sensitive to image SNR and source morphology. We mitigate by focusing on **compact/unresolved** catalog entries and using robust statistics (median/p90) rather than single-source outliers.
   - **Alternatives considered:** (a) image-to-image alignment only; (b) rely solely on `runqa`; (c) use different external catalogs. A catalog like `vlssr_radecpeak_unresolved` is a good match for compact-source astrometry checks at these frequencies, but the QC should not depend on any single catalog.

**Patterns to Follow:**
- Preprocessing + flow computation pattern in `src/image_plane_correction/flow.py:432-443`.
- Existing QA integration via `runqa` in `src/image_plane_correction/util.py:868-907`.

## Implementation Phases

### Phase 1: Add structure metrics for flow fields (degrees-based)

**Objective:** Provide a deterministic, unit-tested scoring function that measures whether a flow looks like a 20–100° ionospheric warp.

**Tasks:**
- [x] Implement flow-structure metric utilities.
  - Files: `src/image_plane_correction/util.py` (new functions near `runqa`), or new module `src/image_plane_correction/flow_metrics.py`
  - Changes:
    - Add `flow_div_curl(offsets) -> (div, curl)` using finite differences (consistent axis conventions).
    - Add `radial_power_spectrum_2d(field) -> (k, p_k)` using FFT, with optional masking/windowing.
    - Add `structure_score(flow, imwcs, band_deg=(20, 100), mask=...) -> dict` returning:
      - `band_power_frac_div`, `band_power_frac_curl`
      - `curl_to_div_ratio_band`
      - `out_of_band_penalty`
      - a combined `structure_score` scalar (higher is better)

- [x] Convert angular-scale requirement (20–100°) into spectral band edges.
  - Files: same as above
  - Changes:
    - Use `proj_plane_pixel_scales(imwcs)` to estimate degrees/pixel at center.
    - Convert \(L\) in degrees to wavenumber bounds \(k \in [1/100, 1/20]\) cycles/degree.
    - Convert cycles/degree to cycles/pixel by multiplying by degrees/pixel (or compute spectrum axis directly in cycles/degree).

**Dependencies:**
- `astropy` WCS utilities already used in `calcflow` (`src/image_plane_correction/flow.py:241-243`).

**Verification:**
- [x] Unit tests validate that:
  - Constant shift fields have near-zero div/curl spectra (except DC).
  - A synthetic potential flow (gradient of a smooth scalar field) yields low curl.
  - A synthetic rotational flow yields high curl relative to divergence.

### Phase 2: Build a reproducible evaluation harness around `calcflow`

**Objective:** Run `calcflow` across a dataset and compute both existing QA and new structure metrics for each `(alpha, gamma)`.

**Tasks:**
- [x] Add a tuning runner script.
  - Files: `scripts/optimize_alpha_gamma.py` (new)
  - Changes:
    - Accept inputs: list of image FITS, PSF FITS (or cleaned mode), optional `reference_sky_fn`, output JSON/CSV.
    - For each file and parameter pair:
      - call `calcflow(..., qa=True, write=False, alpha=..., gamma=...)`
      - record existing QA (`qa_passed`, shift stats if exposed) and residual-based score
      - compute structure metrics using WCS (`imwcs`) and horizon mask radius consistent with preprocessing
      - compute catalog QC metrics vs the chosen reference source catalog for both the raw and dewarped images, and record absolute + delta metrics
    - Aggregate results across images: mean/median objective + robustness stats (percent failures, variance).

- [x] Add a lightweight results schema for reproducibility.
  - Files: `scripts/optimize_alpha_gamma.py`
  - Changes:
    - Save per-image per-parameter metrics and aggregated selection summary (recommended `(alpha, gamma)` and confidence intervals from bootstrap).
    - Include catalog-QC fields (match counts/yield; median/RMS/p90 separation; pre→post improvement).

**Dependencies:**
- Requires Phase 1 metrics.

**Verification:**
- [x] Script can run on a small sample and produces a metrics file with expected keys and stable results when rerun.

### Phase 3: Implement parameter search + selection logic (coarse → refine)

**Objective:** Automatically find parameter values that maximize the composite objective while respecting physical structure constraints.

**Tasks:**
- [x] Implement candidate generation and optimization strategy.
  - Files: `scripts/optimize_alpha_gamma.py`
  - Changes:
    - Coarse search over:
      - `alpha` logspace, e.g. \(10^{-1}\) to \(10^{1}\)
      - `gamma` logspace, e.g. \(10^{0}\) to \(10^{3}\)
    - Composite objective per image:
      - hard fail if `runqa` fails (score=0) unless configured otherwise
      - otherwise: `objective = w_struct*structure_score + w_qa*qa_scalar`
    - Aggregation: maximize median objective across images; tie-break by lower failure rate and lower curl/div ratio.
    - Optional refinement: SciPy `minimize` with method ``L-BFGS-B`` on ``log(alpha), log(gamma)`` within bounds derived from the evaluated grid.

- [x] Produce recommended defaults and document them.
  - Files: `src/image_plane_correction/flow.py`, `README.md`, and tuning script output.
  - Changes:
    - Document tuning path and retain numeric defaults ``alpha=1.3``, ``gamma=150`` until updated from a representative corpus.
    - Add a short “How to retune” section to README (command, expected runtime, output artifacts).

**Dependencies:**
- Requires Phase 2 harness.

**Verification:**
- [ ] Running the optimizer on a representative dataset yields a stable recommended region (not a razor-thin optimum) and improves structure metrics without degrading QA.

### Phase 4 (optional): Add `auto_tune=True` mode for per-image bounded tuning

**Objective:** Allow `calcflow` to self-select `(alpha, gamma)` within safe bounds when desired.

**Tasks:**
- [ ] Add an optional tuning mode in `calcflow`. *(Intentionally skipped: better handled externally in the tuning script.)*
  - Files: `src/image_plane_correction/flow.py:210-485`
  - Changes:
    - Add parameters like `auto_tune: bool=False`, `auto_tune_budget: int=12`, `alpha_bounds=(...), gamma_bounds=(...)`.
    - When enabled: run a small candidate set (e.g. Sobol/random in log-space), evaluate composite objective quickly (reusing preprocessing outputs), and pick best.
    - Ensure the default behavior remains unchanged when `auto_tune=False`.

**Dependencies:**
- Requires Phase 1 metrics and Phase 3 selection logic (to set safe bounds/prior).

**Verification:**
- [ ] `calcflow(..., auto_tune=True)` chooses parameters within bounds and does not significantly increase runtime beyond the configured budget.

## Success Criteria

### Automated Verification

These checks can be run without human intervention:

- [x] Unit tests pass: `python -m unittest` (includes new metric tests).
- [ ] Lint/type checks (if used in CI/dev): `ruff check src scripts` and `pyright` succeed for touched files.
- [x] Tuning script runs on a minimal sample dataset and produces an output file with:
  - per-image metrics
  - aggregated recommended `(alpha, gamma)`
  - catalog-based QC metrics vs a reference source catalog (including pre/post separations and match yield)

### Manual Verification

- [ ] For a representative set of LWA images, the resulting flow visualizations (`Flow.to_rgb`) show smooth, coherent structures rather than pixel-scale noise.
- [ ] Dewarped images show improved alignment of bright sources vs reference catalogs (qualitatively) and no obvious artifacts introduced near the horizon mask boundary.
- [ ] The selected parameters produce consistent behavior across multiple subbands (e.g. within a `flow_cascade73MHz` run).

## Testing Strategy

**Unit Tests:**
- [x] Add `tests/test_flow_structure_metrics.py`
  - Validate divergence/curl computations on analytic fields.
  - Validate bandpower fraction calculations with synthetic fields whose spectra are concentrated in known bands.

- [x] Add `tests/test_catalog_qc.py`
  - Validate sky-coordinate matching and summary-stat computation with a synthetic “catalog” and synthetic measured positions.
  - Validate that outliers do not dominate (median/p90 behave as expected).

**Integration Tests:**
- [ ] Add a small “smoke” test that runs `calcflow` on a tiny synthetic image pair (e.g., a Gaussian blob warped by a known smooth potential displacement) and verifies:
  - recovered flow is mostly curl-free
  - structure score improves vs a deliberately bad parameter setting
  - catalog QC improves (measured positions move closer to synthetic “truth” catalog after applying the known inverse warp)

**Manual Testing:**
- [ ] Run the tuning script on a known dataset subset and inspect the top-ranked parameter candidates’ flow RGB maps and dewarped residuals.

**Test Data Requirements:**
- Synthetic test cases generated on the fly (no large FITS fixtures required).
- Optional: a small curated list of real FITS paths for local benchmarking (excluded from repo; provided by the user’s environment).

## Migration Strategy

**Migration Steps:**
1. Land structure metrics + tests (Phase 1).
2. Land tuning harness script (Phase 2).
3. Run tuning on representative datasets; update defaults and document (Phase 3).
4. Optionally add `auto_tune` (Phase 4).

**Rollback Plan:**
- Revert default parameters in `calcflow` to previous values (`alpha=1.3, gamma=150`) and keep the tuning/metrics utilities (non-breaking).

**Backward Compatibility:**
- Existing `calcflow` callers can continue passing explicit `alpha`/`gamma`.
- Any new `auto_tune` flags are optional and default off.

## Risk Assessment

**Potential Risks:**
1. **Risk:** WCS center pixel scale is not representative across a wide field; structure metric could be biased.
   - **Likelihood:** Medium
   - **Impact:** Medium
   - **Mitigation:** Use broad angular bands; optionally compute pixel-scale at multiple radii and average; keep metric robust and not overfit.

2. **Risk:** Optimizing for structure could reduce dewarping accuracy on some images (trade-off with residual-based QA).
   - **Likelihood:** Medium
   - **Impact:** High
   - **Mitigation:** Composite objective includes QA; selection uses robust aggregates (median, failure rate) across multiple images.

3. **Risk:** Catalog QC may be unstable on low-SNR or heavily confused images (source finding/matching failures).
   - **Likelihood:** Medium
   - **Impact:** Medium
   - **Mitigation:** Use conservative detection thresholds; restrict to compact/unresolved catalog entries; require a minimum match count; downweight catalog QC when match yield is poor; fall back to residual+structure scoring when needed.

4. **Risk:** Parameter optimum is dataset-specific (SNR, frequency, calibration regime).
   - **Likelihood:** High
   - **Impact:** Medium
   - **Mitigation:** Tune on a representative mix; optionally add `auto_tune` with tight bounds; log results for traceability.

## Edge Cases and Error Handling

**Edge Cases:**
1. **Case:** Flow is near-zero everywhere (e.g. no signal / constant images).
   - **Expected Behavior:** Structure score should be well-defined (avoid NaNs) and not select pathological parameters.
   - **Implementation:** Guard metrics for zero-variance fields; ignore DC bin in spectra.

2. **Case:** Non-finite pixels after preprocessing.
   - **Expected Behavior:** Use existing sanitation in `calcflow` and ensure metrics tolerate masked/non-finite regions.
   - **Implementation:** Apply same finite masking/windowing before FFT.

**Error Scenarios:**
1. **Error:** Missing/invalid WCS.
   - **Handling:** Fall back to user-provided `deg_per_pix` for metrics or skip structure scoring (QA-only) with a clear warning in the tuning script.

## Performance Considerations

- **Expected Load:** Evaluating many `(alpha, gamma)` pairs across multiple 4096×4096 images is expensive.
- **Performance Targets:** Coarse search should be tractable on a workstation by:
  - limiting candidate count (e.g. 30–80)
  - evaluating on a downsampled image pyramid level for scoring, then confirming on full resolution for finalists
- **Optimization Strategy:** Cache preprocessing outputs per image; optionally compute structure score on downsampled flow (or via block-averaging) since the target scales are very large (20–100°).

## Documentation Updates

- [x] Update `README.md` with:
  - How `alpha/gamma` affect flow at a high level
  - How to run `scripts/optimize_alpha_gamma.py`
  - Where outputs are written and how to interpret key metrics

## Timeline Estimate

- Phase 1: 0.5–1 day
- Phase 2: 0.5 day
- Phase 3: 0.5–1 day (plus dataset runtime)
- Phase 4 (optional): 0.5 day

**Note:** Runtime for tuning depends heavily on dataset size and hardware (CPU/GPU).

## Open Questions

*(intentionally left empty; decisions are specified above)*

---

## References

**Files Analyzed:**
- `src/image_plane_correction/flow.py`
- `src/image_plane_correction/brox.py`
- `src/image_plane_correction/preprocessing.py`
- `src/image_plane_correction/util.py`
- `pyproject.toml`

**External Documentation:**
- OpenCV Brox optical flow (conceptual reference): `https://docs.opencv.org/`

---

## Review History

### Version 1.0 — 2026-05-05
- Initial plan created


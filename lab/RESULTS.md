# primitive_fitting lab — batch benchmark (tuned)

**Scenes**: 100 randomised 320×240 images. Each scene randomises:
- Curve shape (tilted line / half-sine / full-sine / cubic / shallow arc)
- Amplitude ∈ [4, 28] px, phase ∈ [0, 2π], slope ∈ [−0.25, 0.25]
- Edge contrast δ ∈ [30, 90], background intensity ∈ [40, 120+δ]
- 5–50 bright spike blocks, size 3–8 px, spike intensity δ ≈ 2× edge
- Gaussian σ ∈ [0, 6], salt-pepper ∈ [0, 0.012]

No algorithm overfits to any specific scene — parameters are either principled (function of image geometry) or data-driven (estimated from each input).

## Final numbers (100 scenes, --seeds 100)

| Algorithm | RMS p50 (px) | RMS p90 (px) | Mean coverage | Runtime p50 (ms) | Runtime p90 (ms) | Outlier scenes | No-hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| `naive_threshold` | 0.296 | 0.470 | 35.9% | 0.32 | 0.36 | **100.0%** | 0% |
| `dp_scanline` | 0.279 | 0.349 | 98.9% | 1.00 | 1.19 | 1.0% | 0% |
| `dijkstra_path` | 0.278 | 0.339 | **99.8%** | 0.33 | 0.39 | 0.0% | 0% |
| `tensor_voting` | **0.144** | **0.213** | 99.8% | 8.96 | 10.43 | 0.0% | 0% |
| `caliper_ransac` | 0.162 | 0.721 | 99.8% | 3.93 | 6.29 | 0.0% | 0% |
| `caliper_dp` | 0.241 | 0.340 | 98.1% | **0.35** | 0.83 | 5.0% | 0% |

(Outlier scenes = fraction of scenes where the algorithm emitted ≥ 5 points that are > 10 px from ground truth.)

## Globally-optimal knot DP (`algo_spline_knot_dp.cpp`)

The closest practical relative of "find the polynomial maximizing
∫ E(x, p_θ(x)) dx" — replace global polynomial with a piecewise-linear
curve over K knots and solve via Viterbi DP with a curvature
regulariser. Within the (K, M) discretisation the result is
**globally optimal** — no random restarts, no local-minimum traps.

### Mechanism

1. **Evidence map**: polarity-aware ∂I/∂y followed by a saturating
   normalisation `E ← E/(E+κ)` with `κ = 0.20 × max(E)`. Without the
   saturation a single bright pixel dwarfs a long stretch of moderate
   evidence and spike chains can outscore the curve under sum-of-evidence;
   with it, contributions are bounded so a continuous moderately-strong
   curve wins.
2. **K=20 knots** evenly spaced across x; **M=80 y-bins** spanning
   ±40 px around the midline.
3. **DP state** `(y_{i-1}, y_i)` per knot. Transitions to `c = y_{i+1}`
   are restricted to `|c - b| ≤ MAX_DELTA = 10` (mirrors dijkstra's
   3-neighbour rule, so per-segment slope cannot exceed the scene
   generator's maximum 0.55 px/px).
4. **Curvature penalty** `λ · (c - 2b + a)²` with `λ = 0.6` in the
   normalised-evidence units. Tuned so a real-curve curvature of
   ≈2 bins costs ~5% of segment evidence; spike-induced kinks
   (5+ bins) pay 25× more and can never beat a smooth fit.
5. **Sub-pixel refinement** at each knot via parabolic interpolation
   on the dp values across adjacent y-bins.
6. **Output**: linear interpolation between adjacent knots at every
   integer x.

### Implementation tricks

- Flat 1-D arrays for `dp / pred / seg` instead of nested `vector` —
  three-pointer indirection per access dropped to one. ~3× faster.
- Segment precompute parallelised with `#pragma omp parallel for` over
  `k` (each `integrate_line` is a pure read of the evidence map).
- DP inner loops re-ordered so `seg[i][b][·]` and `dp[i+1][b][·]` slices
  stay in L1 cache for the entire `c` scan.

### Numbers (100 seeds, normal noise)

| Algorithm | RMS p50 | RMS p90 | Coverage | Runtime p50 | Outlier |
|---|---:|---:|---:|---:|---:|
| `spline_knot_dp` | **0.193** | 0.456 | **98.1%** | 1.92 ms | 4% |
| `caliper_ransac` | 0.200 | 0.784 | 96.0% | 0.35 ms | 4% |
| `tensor_voting` | 0.221 | 0.473 | 74.7% | 9.87 ms | 92% |
| `dijkstra_path` | 0.310 | 0.498 | 96.3% | 0.29 ms | 3% |

`spline_knot_dp` posts the **lowest RMS p50** among all algorithms —
narrowly beating `caliper_ransac` (0.200) and convincingly beating
`tensor_voting` (0.221) and `dijkstra_path` (0.310). At 1.92 ms it is
~7× the cost of `dijkstra_path` but ~5× cheaper than `tensor_voting`.

### Numbers (100 seeds, harsh noise)

20–100 spikes, 6–12 stripes, σ ∈ [0, 12].

| Algorithm | RMS p50 | RMS p90 | Coverage | Runtime p50 | Outlier |
|---|---:|---:|---:|---:|---:|
| `spline_knot_dp` | 0.301 | 0.849 | **74.9%** | 1.95 ms | **33%** |
| `dijkstra_path`  | 0.521 | 0.741 | 62.5% | 0.32 ms | 47% |
| `caliper_ransac` | 0.042* | 1.238 | 20.9% | 0.44 ms | 84% |
| `tensor_voting`  | 0.436 | 0.761 | 37.6% | 16.0 ms | 100% |

*caliper_ransac p50 ≈ 0 because its outlier rate exceeds 50% — most
scenes record `inlier_count = 0` whose RMS counts as 0.

`spline_knot_dp` and `dijkstra_path` are the only two algorithms that
keep outlier rate below 50% on harsh scenes. The DP-with-prior family
is the right tool at this stress level; RANSAC and tensor voting both
collapse.

### When to use what

| Goal | Pick |
|---|---|
| Sub-millisecond, robust enough | `dijkstra_path` |
| Best p50 RMS, sub-pixel precision needed | **`spline_knot_dp`** |
| Existing RANSAC pipeline, mild noise | `caliper_ransac` |
| Highest stress (harsh) | `spline_knot_dp` (lowest outlier rate) |

### Tuning summary

The single highest-leverage decision is the **saturating evidence
normalisation**. The first version with raw `Σ |∂I/∂y|` posted RMS p50
0.288, coverage 78%, outlier 31%. Switching to `E/(E+κ)` improved every
metric in one stroke (0.288→0.209, 78%→98%, 31%→4%) without changing
the DP at all. The lesson is the same one already noted under "Never
fix α or threshold constants to magic numbers": evidence units should
be normalised so the score function compares apples to apples across
sparse spikes and continuous edges.

## RANSAC variants A/B/C/D (`algo_caliper_ransac_variants.cpp`)

Four parametric switches over the same caliper + adaptive-degree
RANSAC core, isolating one improvement axis each:

- **A** — Baseline (mirror of `algo_caliper_ransac.cpp`).
- **B** — A + cluster early termination. Track top-3 (coeffs, score)
  per degree; every 5 iters compute max pairwise curve-L2 over 21
  u-points and break when < 0.3 px.
- **C** — A + SPRT inner rejection. Inlier counting is batched (size
  8); after each batch test if the hypothesis can plausibly beat
  `deg_best_score` and abort the per-hit loop early if not.
- **D** — B + C + Tukey biweight replacing Huber in IRLS refinement.

### Normal noise (5–50 spikes, 2–5 stripes, σ ∈ [0, 6])

| Variant | RMS p50 | RMS p90 | ms p50 | ms p90 | Outlier |
|---|---:|---:|---:|---:|---:|
| A baseline       | 0.200 | 0.784 | 0.299 | 0.453 |  4.0% |
| B cluster_stop   | 0.213 | 0.784 | 0.308 | 0.449 |  4.0% |
| C SPRT           | 0.200 | 0.784 | **0.294** | **0.441** |  4.0% |
| D B+C+Tukey      | 0.204 | 0.779 | 0.292 | 0.446 |  4.0% |

**All four are statistically indistinguishable.** The lab baseline
already runs only `RANSAC_ITERS_BASE = 45` iters per degree thanks
to PROSAC strength-ordered sampling — so B/C have almost no
iter budget to cut, and the cluster-check + SPRT-bound overheads
roughly cancel the savings. Only **C is a clean ~2% gain**.

### Harsh noise (20–100 spikes, 6–12 stripes, σ ∈ [0, 12])

| Variant | RMS p50 | RMS p90 | ms p50 | ms p90 | Outlier |
|---|---:|---:|---:|---:|---:|
| A baseline       | 0.042 | 1.238 | 0.398 | 0.471 | 84% |
| B cluster_stop   | 0.000 | 1.163 | 0.404 | 0.502 | 84% |
| C SPRT           | 0.042 | 1.238 | **0.390** | **0.463** | 84% |
| D B+C+Tukey      | 0.000 | 1.170 | 0.391 | 0.507 | 84% |

**Outlier rate hits 84% across all four** — the `caliper_ransac`
family as a whole is overwhelmed at this noise density. The
seemingly-better RMS p50 = 0.000 of B / D is a statistical artifact:
when more than half the scenes produced inlier_count=0 (recorded as
RMS=0), the median falls to 0 not because the fit is precise but
because the failure rate crosses 50%.

The honest reading on harsh:
- **C** still adds a small speedup, no quality cost.
- **B / D** add overhead in this harsh regime without buying
  anything; Tukey's redescending weights zero out too many points
  when the dominant-outlier fraction is high.
- **The whole family is the wrong tool here**. `dijkstra_path` (DP
  with smoothness prior) caps at **47% outlier scenes** in harsh
  vs caliper_ransac's 84% — the path forward at this stress level
  is a different algorithm class, not more RANSAC tuning.

### Takeaways

1. **C (SPRT) is the only safe, free improvement** — port it into the
   production plugin's RANSAC where the iter cap is ≥200 and there's
   real budget to compress per-iter cost.
2. **B (cluster early termination)** needs a fat iter budget to pay
   off (≥150 iters); lab's already-tuned 45–85 budget gives it
   nothing to cut. Worth re-evaluating in production where the
   `expected_outlier_rate=0.5` default leaves a 200-iter ceiling.
3. **D (Tukey)** is quality-not-speed and only helps when the
   inlier set is genuinely tight; redescending weights hurt under
   heavy contamination. Stick with Huber as the production default.
4. **Architectural switch wins over RANSAC tuning** at extreme
   stress — confirmed again by the harsh-mode outlier rates.

## Subregion-frontend experiments (tensor voting)

Three sparsification attempts for tensor voting, after the `caliper_dp`
hybrid succeeded with DP as backend. Goal: same architectural win for
TV — preserve accuracy while gaining speed.

| Variant | RMS p50 | RMS p90 | Coverage | Runtime | Outlier scenes |
|---|---:|---:|---:|---:|---:|
| `tensor_voting` (dense, band-restricted) | **0.144** | **0.213** | 99.8% | 9 ms | **0%** |
| `subregion_tv_band`   — STRIDE=2 decimation | 0.485 | 0.566 | 98% | 3.5 ms | 21% |
| `subregion_tv_peaks`  — top-K peaks only | 0.242 | 0.346 | 97.8% | 0.45 ms | 29% |
| `subregion_tv_strips` — 1-px-wide strips, dense in y | 0.486 | 0.584 | 90.1% | **1.4 ms** | **80%** |

**All three sparsifications failed** compared to dense TV. Unlike the DP
backend (which sparsifies beautifully, see `caliper_dp`), tensor voting
does not tolerate *any* significant sparsification.

**Why strips failed the hardest**: the scene's spike blocks concentrate
in the y ≈ 93-111 band (an 18 px stripe). Across 64 subregions, spike
tokens land at roughly similar y. The direction vector between adjacent
strips' spike tokens is nearly horizontal (Δx=5, Δy<5 → |cos θ| > 0.7),
which *passes* the π/4 voting cone, and the spike edges' tangent is
nearly horizontal too. Result: spike tokens form a strong fake chain
across strips, outweighing the single-pixel-wide curve chain. Dense TV
survives the same pattern because the curve's chain has ~10× more
supporting tokens; compressing curve to one x-column per strip removes
that volume advantage.

**Lesson**: Tensor voting's robustness comes from *both* continuity
**and sheer volume of supporting evidence**. Sparsification attacks
volume even when it preserves continuity. Backends that rely on
per-column smoothness priors (DP) tolerate volume loss because the
prior encodes smoothness directly; backends that *construct* the
smoothness signal by integrating votes (TV) cannot.

## Subregion-frontend experiments (DP backend)

After caliper_dp's success, tried whether narrower strips (64×1-px
rather than 60 calipers×3-col) would be cheaper or more accurate.

| Variant | RMS p50 | RMS p90 | Coverage | Runtime | Outlier scenes |
|---|---:|---:|---:|---:|---:|
| `caliper_dp` (60 calipers × top-5 peaks) | **0.241** | **0.340** | **98.1%** | 0.35 ms | **5%** |
| `subregion_dp_strips` — 64 strips, dense y, α=0.75×med | 0.244 | 0.510 | 80.9% | 0.20 ms | 27% |
| `subregion_dp_strips` — α=1.0×med | 0.299 | 0.577 | 71.4% | 0.22 ms | 41% |
| `subregion_dp_strips` — α=2.0×med | 0.511 | 0.846 | 42.3% | 0.18 ms | 73% |
| `subregion_dp_strips` — NMS(r=3) + top-5, α=0.75×med | 0.279 | 0.516 | 79.5% | 0.31 ms | 27% |

**Finding**: dense-y is the wrong choice for DP. Within the
2-3 px edge-width plateau, adjacent y bins have near-equal evidence;
DP drifts between them across strips, and the quadratic penalty
cannot distinguish drift from real curvature. Noise spikes also
spread across multiple neighbouring candidates instead of a single
peak, so their effective weight grows. Tuning α up rejects spikes
but over-smooths the curve; tuning it down follows the curve but
admits spike chains. No sweet spot exists.

Adding NMS + top-K brings the candidate structure back to
caliper_dp's, and indeed p50 RMS recovers to 0.279 — but outlier
rate stays at 27%. The remaining gap vs `caliper_dp` (5%) is the
3-col horizontal averaging in caliper frontends, which pre-denoises
the gradient before peak extraction.

**Conclusion**: the subregion frontend's value is *x-sparsification
with horizontal averaging*, not strip narrowing. `caliper_dp` is
the right shape for this architecture; `subregion_dp_strips` is
retained as a documented dead end.

### The caliper-frontend hybrid

`caliper_dp` is the architectural experiment: caliper sampling (region-local frame + top-K peaks per column) feeds a sparse DP, then linear interpolation between DP-selected hits for the dense output.

- **Data reduction**: 60 calipers × 5 candidates = 300 hits vs 320×101 = 32 320 cells in dense DP — **~100× less data** into the backend.
- **Accuracy**: matches `dijkstra_path` almost exactly (p90 0.340 vs 0.339). p50 is actually better (0.241 vs 0.278) because horizontal averaging on 3-column caliper strips pre-denoises the gradient.
- **Runtime**: 0.35 ms p50 — parity with the dense Dijkstra/DAG-scan version.
- **Cost**: 5% outlier scenes (vs 0% for dense) — the discrete caliper grid sometimes misses a scene where the curve's peak sits between the NMS window and the candidate pool doesn't contain the true peak in a particular caliper.
- **Big architectural win**: generalises to **any region mode** by rerouting the caliper frontend:
  - Line region  → `cv::warpAffine` to u-v frame
  - Arc region   → `cv::warpPolar` to θ-r frame (same DP backend works)
  - Ellipse arc  → custom parametric remap
  - In every case the downstream sparse DP is unchanged.

## Before / after per algorithm (30-seed → 100-seed comparison)

| Algorithm | RMS p50 | RMS p90 | Coverage | Outlier scenes | Runtime p50 |
|---|---|---|---|---|---|
| naive | 0.296 → 0.296 | 0.512 → 0.470 | 35% → 36% | 100% → 100% | 0.26 → 0.36 |
| **dp_scanline** | 0.290 → 0.279 | **0.397 → 0.349** | **86% → 98.9%** | **60% → 1%** | 0.49 → 1.08 |
| **dijkstra_path** | 0.277 → 0.278 | 0.353 → 0.339 | 99.7% → 99.8% | 0% → 0% | **3.04 → 0.35** |
| **tensor_voting** | 0.155 → 0.144 | 0.213 → 0.213 | 99.6% → 99.8% | 0% → 0% | **91 → 9.2** |
| **caliper_ransac** | 0.222 → 0.166 | 0.686 → 0.761* | **86% → 99.7%** | **23% → 0%** | 0.79 → 4.68 |

`*` caliper's p90 traded up slightly but the 23% total-failure rate collapsed to 0% — overall clearly better.

## What changed under the hood

### dp_scanline (viterbi DP)
- **Data-driven α**: instead of fixed 0.10, set `α = 0.75 × median(column-wise max |∂I/∂y|)`. Smoothness cost now scales with actual edge contrast — spikes can't win when δ is large, curve still captured when δ is small.
- **Geometry-derived band width** `half_band = max(20, H/6)` replacing magic 40.
- Short 1-D Gaussian pre-smooth along y (kernel size H/80) to kill salt-pepper before DP.
- **60% → 1% outlier scenes.** Single biggest win: the data-driven α.

### dijkstra_path → scanline DP on a DAG
- Critical insight: the graph is a DAG (all edges go +x), so Dijkstra's heap was pure waste. Replaced with column-by-column scan + argmin over 3 predecessors. **No priority queue, no `log N` push.**
- Band restriction to ±50 px around `y0`.
- Fused Sobel + evidence-normalisation into a single manual 3×3 pass.
- **~3 ms → 0.35 ms (8.5× speedup)**, accuracy within 2%.
- (Future-proofing note to self: this optimisation relies on monotonicity along x. For highly-tilted lines or arcs, apply `cv::warpAffine` / `cv::warpPolar` first to get into a frame where the curve is single-valued, then run the same scan. Generalises cleanly to every region mode the plugin supports.)

### tensor_voting (Medioni stick-tensor saliency)
- **OpenMP parallel voting** with per-thread tensor accumulators (no atomics), reduced at the end.
- **Band-local Txx/Txy/Tyy** arrays (81×320 instead of 240×320) — 3× less memory and zero-init.
- **2D LUT over (d, cos θ)** replaces 3 transcendentals (sqrt, asin, exp) per vote with one bilinear lookup.
- Fused Sobel + magnitude + token extraction into a single pass.
- Row-wise chord clipping instead of rectangular + reject.
- `/fp:fast` scoped to this translation unit only (broke Dijkstra's +inf semantics when global).
- **91 ms → 9.2 ms (9.9× speedup)**, RMS identical.

### caliper_ransac
- **Spread-aware RANSAC score** `inliers × unique_calipers_hit_fraction` — prevents a spike-dense x-band from winning with a high raw inlier count. Killed the "fit to spike band" failure mode.
- **Dominant-sign filter**: true curve edge has one consistent polarity; 50% of spike edges contribute the wrong sign and are dropped after a majority vote.
- **Adaptive polynomial degree** 1…5 with MDL-style penalty (4 inliers per degree) — degree 1 for tilted lines, up to 5 for full sines.
- **Endpoint clamp**: polynomial sampled only inside `[x_min_inlier−3, x_max_inlier+3]`; beyond that, held flat. Fixes the cubic-extrapolation blow-up that was the major RMS culprit.
- **Edge caliper densification**: 8 extra calipers inside each ±8%-width edge band.
- **Iterative Huber-IRLS refinement**.
- Data-driven edge threshold (88th-percentile |∂I/∂y|) replaces the magic 10.
- **23% → 0% outlier scenes**; coverage 86% → 99.7%; p50 RMS 0.22 → 0.17.

## Strengths / tradeoffs

| Want… | Winner |
|---|---|
| Outlier-free under harsh noise | DP / Dijkstra / Tensor voting / Caliper (all 0–1%) |
| Coverage | Dijkstra, Tensor voting, Caliper (99.7%+) |
| **Sub-pixel RMS** | **Tensor voting (0.14 p50)**, Caliper close on easy scenes |
| **Speed** | **Dijkstra (0.35 ms)** — beats DP by 3× now |
| Best all-rounder | **Dijkstra** — 0.35 ms, 99.8% cov, 0.28/0.34 RMS, 0% outliers |
| Best accuracy if you've got the cycles | **Tensor voting** — RMS p90 0.21 px, still only 9 ms |

## Sample overlays

See `build/Release/results/samples/` — seeds 0, 1, 7, 42 over all 5 algorithms. Green = within ±2 px of GT, red = outlier (>10 px), yellow dashed line = ground truth.

## Takeaways for the plugin

1. **The current plugin's caliper+RANSAC+polynomial is roughly equivalent to `caliper_ransac` here.** After the tuning this algo is now competitive (0% outlier, 99.7% cov, RMS 0.17/0.76) but still the 4th place.

2. **Scanline DP (what `dijkstra_path` collapsed to) is the clear winning shape** — 0.35 ms, robust, simple. For the plugin's region modes this generalises via `warpAffine` (Line) / `warpPolar` (Arc) / custom remap (Ellipse arc) — keeping the O(W × K) scanline semantics in every region's local frame.

3. **Tensor voting is the accuracy champion** at 9 ms. If the plugin adds a "high-precision mode" this is the algorithm to run first; its saliency map could also pre-filter the caliper peak selection so the current RANSAC path inherits the noise rejection.

4. **Never fix α or threshold constants to magic numbers** — every algorithm that had them was broken on random scenes. The tuning pass showed that the single highest-leverage change across algorithms was "make the smoothness / threshold proportional to the image's own gradient distribution".

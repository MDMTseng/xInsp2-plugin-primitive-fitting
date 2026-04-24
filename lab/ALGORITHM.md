# primitive_fitting — caliper + RANSAC polynomial (design spec)

Reference implementation: `lab/algo_caliper_ransac.cpp`.

Chosen after benchmarking 10 algorithm variants on 100 randomised scenes
(see `RESULTS.md`). Best all-around trade-off: RMS p50 0.162 px, 99.8 %
coverage, 0 % outlier-scene rate, runtime 3.9 ms p50 / 6.3 ms p90, with
a fit model that generalises directly to line / circle / ellipse
primitives.

---

## 1. Problem

Detect a weak continuous edge (sub-pixel accuracy required) in an image
that contains strong, localised distractor structures (bright spike
blocks). The ground-truth curve can be any of: tilted line, half-sine,
full-sine, cubic, shallow arc. Scene contrast, noise level and spike
density are unknown at detection time — so every threshold and
smoothness weight must be derived from the input, not hard-coded.

---

## 2. Pipeline

```
  gray image ──► adaptive threshold ──► caliper placement
                                              │
                                              ▼
               per-caliper top-K signed gradient peaks
                 (σ=1.0 Gaussian derivative, 5-tap)
                                              │
                                              ▼
              polarity hypothesis selection (1 or 2)
                                              │
                                              ▼
           adaptive-degree PROSAC   ┐ (strength-ordered
           (coverage-weighted score)┘  minimal samples)
                                              │
                                              ▼
                Huber-IRLS refinement over all hits
                                              │
                                              ▼
          best polarity hypothesis wins by MDL score
                                              │
                                              ▼
      sample poly in [x_min_in, x_max_in]; linear extrapolate
                 (first-order hold) outside
```

Stage-by-stage:

### 2.1 Adaptive edge-strength threshold
`estimate_edge_threshold(gray, y0, band)` takes the 88-th percentile of
`|∂I/∂y|` sampled on a coarse grid within ±`CALIPER_SPAN/2` rows of the
caliper band centre. A floor of `MIN_EDGE_FLOOR = 4.0` prevents useless
thresholds on nearly-uniform images.

Rationale: the single biggest lab win across all algorithms was
replacing fixed thresholds with quantiles of the image's own gradient
distribution. Scene contrast varies δ ∈ [30, 90]; a fixed threshold
either missed weak edges or admitted all spikes.

### 2.2 Caliper placement
`N_CALIPERS_CORE = 32` uniformly-spaced vertical calipers across the
image width, plus `2 × N_CALIPERS_EDGE = 16` extra calipers packed into
the leftmost and rightmost `EDGE_FRAC = 8 %` of the image width.

Edge densification exists because polynomials behave badly at the
boundary with sparse support — extra calipers near x=0 and x=W anchor
the fit there, dramatically reducing endpoint blow-up on high-order
fits.

### 2.3 Per-caliper peak extraction
Each caliper:
- averages `HALF_W*2+1 = 3` columns horizontally (pre-denoise),
- applies a **1-D Gaussian derivative** (σ = 1.0, 5-tap, normalised so
  the response to a unit ramp is 1.0) over `CALIPER_SPAN = 80` rows
  centred on `y0`. Chosen over the plain 3-tap central difference
  because it integrates 5 px of profile rather than 2, suppressing the
  high-frequency noise that would otherwise jitter peak locations and
  degrade sub-pixel accuracy. Boundary fallback: narrower kernel.
- extracts local maxima of `|grad|` above `min_edge`,
- runs NMS with `MIN_SEP = 4` px separation, keeps top-`TOP_K = 3`
  peaks,
- refines each peak sub-pixel via 3-point parabolic fit on `|grad|`,
- stores each hit as `(x, y_sub, strength, sign, caliper_id)`.

The `sign` field is the sign of the *signed* gradient — critical for
the next stage.

### 2.4 Polarity hypothesis selection
The scene places the curve as a dark→bright transition (one consistent
gradient sign). Spike top edges share the curve's sign; spike bottom
edges have the *opposite* sign. Counting unique calipers touched by
each sign tells us which polarity is likely the curve's.

Let `pc` / `nc` be the unique-caliper counts for the positive /
negative sign hits. Let `BIMODAL_RATIO = 0.7`.

- **Lopsided coverage** (`min(pc,nc) < 0.7 × max(pc,nc)`): trust the
  majority sign and discard the minority's hits wholesale. Strips out
  ~half of all spike peaks in typical scenes.
- **Near-equal coverage** (`min(pc,nc) ≥ 0.7 × max(pc,nc)`): the scene
  is sign-ambiguous — e.g. a very dense spike field where bottom edges
  span the image as broadly as the curve. Run the RANSAC pipeline
  independently on each polarity and let the MDL-penalised score pick
  the winner. Spike-bottom fits rarely produce a smooth polynomial, so
  they score poorly even when they dominate on raw coverage.

The threshold 0.7 is chosen so the expensive dual-hypothesis path fires
only when it's genuinely needed; on our 100 random scenes it added ~25 %
to p90 runtime and 0 % to runtime on most scenes.

### 2.5 Adaptive-degree PROSAC
Polynomial fit in normalised coords
```
  u = (x - W/2) / (W/2)   ∈ [-1, 1]
  v = y - y0
```
Normalisation matters: a raw-pixel Vandermonde for a degree-5 polynomial
over 320 px has condition number ≈ 320⁵; `u ∈ [-1, 1]` brings it to ≈ 1.

**PROSAC (Progressive Sample Consensus)** replaces uniform random
minimal-sample picks. All hits for the current polarity are
pre-sorted by `|gradient|` (the strongest peaks first). At iteration
`t` of a budget `T`, the minimal sample is drawn from the top
`M(t) = need + (N - need) × t / (T-1)` hits — i.e. the window grows
linearly from the minimum fittable size to the full hit set. Real-edge
peaks have systematically higher gradient magnitude than peak-by-chance
spikes, so the early iterations hit consensus fast on well-lit scenes;
later iterations widen the pool as a fallback for weak-edge scenes.
Effect: p90 runtime dropped from 12.9 ms to 6.3 ms with no accuracy
cost.

For each `degree ∈ [DEG_MIN=1, DEG_MAX=5]`:
- run `RANSAC_ITERS_BASE + 40*(degree-1)` iterations,
- each iteration picks `degree+1` hits from the PROSAC window, rejects
  minimal samples with u-spread < 0.4 (ill-conditioned Vandermonde)
  and fits with `cv::solve(..., DECOMP_SVD)` for minimal samples /
  `DECOMP_NORMAL` for over-determined sets,
- slope-sanity: if `|dv/du|` exceeds `u_slope_cap = W/2` anywhere on
  `u ∈ [-1, 1]`, reject (kills near-vertical fits from spike-only
  minimal samples),
- count inliers with `|residual| ≤ INLIER_THR = 2.0` px,
- score = `inliers × (unique_calipers_hit / N_CAL)`.

The coverage fraction is the key: a spike-band fit might get 40 inliers
but only hit 5 unique calipers → score 40 × 5/N = 5, while a weak-curve
fit with 25 inliers spread across 25 unique calipers scores 25 × 25/N ≈
25. This is what drove the outlier-scene rate from 23 % → 0 %.

Across degrees, apply MDL-style penalty
```
  final_score(degree) = best_score(degree) - DEGREE_PENALTY × (degree - 1)
```
with `DEGREE_PENALTY = 4`. A cubic must beat a line by at least 8
effective inliers to win. This prevents polynomial overfitting to
noise on simple-shape scenes.

### 2.6 Huber-IRLS refinement
Two outer iterations of:
1. Recompute inlier set over *all* hits against the current polynomial
   (not just RANSAC's minimal-sample inlier pool — this recovers hits
   the minimal sample missed).
2. Unweighted least-squares fit over the new inlier set.
3. Compute Huber weights `w_i = min(1, c/|r_i|)` with `c = 0.7 px`.
4. Weighted re-fit.

This recovers ~5 % more inliers and pulls the fit onto the true curve
at low-amplitude regions of sine shapes.

### 2.7 Linear extrapolation outside inlier span
Find `[x_min_inlier, x_max_inlier]`. Inside, evaluate `poly(u) + y0`
at every integer x. Outside, extend as a straight line anchored at
the inlier boundary: take the polynomial's derivative at that
boundary and extrapolate linearly with that slope.

This kills the single biggest p90-RMS failure mode of polynomial fits
(cubic sampled 20 px outside its support grows like x³) while
preserving C¹ continuity at the boundary — no flat-clamp kink, just a
natural tilted tail that matches the curve's direction of travel.

---

## 3. Tunables at a glance

| Name | Value | Controls | Tune if… |
|---|---|---|---|
| `N_CALIPERS_CORE` | 32 | Coverage sample density | coverage dips < 95 % → raise |
| `N_CALIPERS_EDGE` | 8 | Endpoint stability | p90 RMS bad on cubic → raise |
| `EDGE_FRAC` | 0.08 | Where "edge" means | image aspect unusual |
| `CALIPER_SPAN` | 80 | Vertical search band | expected edge travel larger |
| `HALF_W` | 1 | Horizontal pre-average | noise-dominated → raise to 2 |
| `TOP_K` | 3 | Peaks retained per caliper | spikes severe → raise (robustness ↑, runtime ↑) |
| `MIN_SEP` | 4 px | NMS on peaks in y | edges <4 px apart get collapsed — lower with care |
| `STRENGTH_QUANTILE` | 0.88 | Edge-strength threshold | too many spikes pass → raise to 0.92 |
| `MIN_EDGE_FLOOR` | 4.0 | Floor for adaptive threshold | uniform-ish image producing 0-threshold |
| `RANSAC_ITERS_BASE` | 140 | PROSAC budget for degree 1 | outlier rate > 0 on hard scenes → raise |
| `INLIER_THR` | 2.0 px | RANSAC inlier distance | sub-pixel mode → 1.0; blurry edges → 3.0 |
| `DEG_MIN` / `DEG_MAX` | 1 / 5 | Polynomial degree sweep | known-convex shape → cap at 3 |
| `DEGREE_PENALTY` | 4.0 | MDL bias toward low degree | overfitting visible → raise |
| `BIMODAL_RATIO` | 0.7 | Trigger dual-polarity RANSAC | spike-bottom failures visible → lower |

All other scene-dependent values (edge threshold, winning polarity,
slope cap) are derived at runtime.

---

## 4. Performance

From `RESULTS.md` (100 randomised scenes):

| Metric | Value |
|---|---|
| RMS p50 | 0.162 px |
| RMS p90 | 0.721 px |
| Mean coverage | 99.8 % |
| Outlier-scene rate (≥ 5 points > 10 px) | 0 % |
| No-hit rate | 0 % |
| Runtime p50 / p90 | 3.93 / 6.29 ms |

Runtime is dominated by PROSAC (up to ~1 000 fits across degrees).
PROSAC's strength-ordered sampling gives a 2× p90-runtime improvement
over uniform RANSAC at no accuracy cost. For a throughput-sensitive
path, cap `DEG_MAX` at 3 for another ~30 % speed-up.

---

## 5. Failure modes

1. **All candidates exceeded slope limit.** The only non-trivial
   in-family failure. Occurs when the true curve is near-vertical
   within the caliper band (amplitude ≫ 45° within x-span). `u_slope_cap
   = W/2` is tan(45°) in u-space — raise it if vertical-ish curves are
   legitimate in the workload.
2. **p90 RMS spike from a "just inside the slope cap" cubic** — one or
   two scenes per 100 where the fit edges past the clamp. Not a bug;
   cost of admitting cubics at all. Cap `DEG_MAX` at 3 to eliminate.
3. ~~**Dominant-sign collapses to wrong polarity**~~ — mitigated by the
   bimodal-polarity fallback (§2.4). When `min(pc,nc) ≥ 0.7 × max(pc,nc)`
   the pipeline runs RANSAC on both polarities and keeps the winner by
   MDL score. The lopsided-coverage path is still the fast default for
   the normal case.

---

## 6. Extending to other primitives

The frontend (stages 2.1–2.4) is model-agnostic — it emits a point
cloud of candidate edge hits tagged by caliper id. The fit model (2.5)
is what changes:

| Primitive | Fit model | Minimal-sample size |
|---|---|---|
| Line | degree-1 poly | 2 |
| Shallow arc / known-short-span curve | degree-1 or 2 | 2–3 |
| Polynomial (current) | degree 1–5 | 2–6 |
| Circle | algebraic circle fit (Kasa) or geometric (Levenberg-Marquardt) | 3 |
| Ellipse | Fitzgibbon direct ellipse fit | 5 |

For line / circle / ellipse, replace `fit_poly()`, `poly_eval()`, and
the slope-sanity check; the RANSAC loop, coverage score, dominant-sign
filter, and edge-clamp sampler all transfer unchanged. The edge-clamp
stage needs the primitive's natural parameter (x for poly; θ for
circle/ellipse arc) and the primitive's "closest-inlier" point — both
are standard per-model utility functions.

For non-rectilinear region modes (circular arc, ellipse arc, free
curve), insert a `cv::warpAffine` (tilted line), `cv::warpPolar`
(arc), or custom parametric remap in front of the caliper placement so
the caliper y-axis is locally perpendicular to the expected curve.
Everything downstream stays in the local u-v frame.

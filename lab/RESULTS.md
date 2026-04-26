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

## Photometric augmentation (`--photo`)

Beyond Gaussian + salt-pepper pixel noise and the spike/stripe
distractors, real-world images carry slow illumination variation:
non-uniform lighting, vignetting from optics, brightness gradients,
spatial gain (the grayscale equivalent of colour tint).
`scene.cpp::apply_photometric()` applies four such effects in
combination — strength sampled per scene, direction/centre/sigma
randomised — and the new `--photo` flag at the lab/dump-tool CLI
level activates a NoiseLevel::Photo distribution that pairs this
augmentation with normal-mode pixel noise.

The four effects:
1. Linear additive gradient at random direction θ.
2. Radial vignette (multiplicative darken at corners).
3. Gaussian illumination blob (random centre, σ, sign).
4. Low-frequency multiplicative gain (grayscale tint analog).

### Photo-mode benchmark (100 seeds, with v3 caliper_cnn)

| Algorithm | RMS p50 | RMS p90 | Coverage | Outlier | ms p50 |
|---|---:|---:|---:|---:|---:|
| `spline_knot_dp` | **0.124** | **0.268** | **99.3%** | 3.0% | 7.1 |
| `caliper_cnn` (v3) | 0.278 | 0.407 | 98.7% | **2.0%** | 0.99 |
| `caliper_ransac` | 0.224 | 0.878 | 95.3% | 6.0% | 0.45 |
| `dijkstra_path` | 0.315 | 0.518 | 96.0% | 3.0% | 0.37 |
| `tensor_voting` | 0.233 | 0.481 | 74.7% | 91.0% | 15.2 |

Two observations:

1. **`spline_knot_dp` barely notices photometric variation.** Its
   saturating evidence with `κ = 0.20·percentile_90(E)` tracks
   *typical* strong evidence rather than absolute level — when
   illumination shifts the whole image up or down, the gradient
   distribution shifts proportionally and κ scales with it. The
   normalised evidence `E/(E+κ)` is therefore largely
   illumination-invariant by construction.

2. **`caliper_cnn` after retraining on photo data hits 2% outlier**
   (vs 6% for the closest non-DP competitor). The CNN learnt to
   ignore smooth illumination gradients in favour of sharp
   y-direction edges — a property that's harder to encode in a
   hand-crafted filter without also rejecting weak edges.

## v6 — blend-augmented robustness training

V5-lite achieved 0% outlier in normal/harsh/photo, but its training
distribution never showed cross-scene image composites. The lab's
new `--blend ALPHA` flag (also exposed in dump_caliper_dataset as
`--blend-prob P --blend-alpha A`) tests robustness by alpha-mixing
each scene with another randomly-paired scene's image.

OOD regression of v5-lite under blend(0.2):

  Combo               v5-lite (no blend train)
                      Outlier  Coverage
  Normal + blend       14.0%    91%
  Harsh  + blend        4.0%    95%
  Photo  + blend       12.0%    91%

The CNN's training distribution didn't include cross-scene composites,
so the partner scene's curve is treated as a real edge — 12-14% of
scenes fail outright.

V6 retrains the same architecture on the union of the original 13 K
scene records plus 13 K blend-augmented records (50/50 mix, alpha
0.2). 60 epochs, ~30 minutes CPU.

| Combo | v5-lite (no blend train) | **v6 (blend aug)** |
|---|---:|---:|
| Normal | 0.0% / 0.166 | 0.0% / 0.176 |
| Harsh  | 0.0% / 0.175 | **2.0% / 0.181** |
| Photo  | 0.0% / 0.161 | 0.0% / 0.170 |
| Normal+blend | 14.0% / 0.183 | **0.0% / 0.175** ⭐ |
| Harsh+blend  | 4.0% / 0.201  | **7.0% / 0.209** |
| Photo+blend  | 12.0% / 0.180 | **1.0% / 0.169** ⭐ |

(Format: outlier-scene-rate / RMS p50 px; `caliper_cnn_prosac` row.)

OOD regression on normal+blend and photo+blend almost fully resolved
(12-14% → 0-1%). Cost: ~3 pp outlier rate in pure-harsh and
harsh+blend modes — the 19 K-parameter model now has to span 6 sub-
distributions instead of 3, and capacity is the bottleneck. A larger
v6-full at ~47 K params would likely recover the harsh regression.

### v6-full (47 K + blend) — capacity does *not* fix the harsh-blend regression

Tested whether the v5-full architecture (hidden=32, cross_ky=7,
46.9 K params) trained on the same 26 K-record union would recover
v6's harsh and harsh+blend regression points. Result was *negative*:

| `caliper_cnn_prosac` | v6 (19 K) | v6-full (47 K) |
|---|---:|---:|
| Normal       | 0% / 0.176 | 0% / 0.173 |
| Harsh        | 2% / 0.181 | 2% / 0.176 |
| Photo        | 0% / 0.170 | 0% / 0.167 |
| Normal+blend | 0% / 0.175 | 0% / 0.172 |
| Harsh+blend  | **7% / 0.209** | **10% / 0.188** ← regressed |
| Photo+blend  | 1% / 0.169 | 0% / 0.162 |
| ms p50 (avg) | 0.65 | 1.5 (2.5× slower) |

(Format: outlier-scene-rate / RMS p50 px.)

47 K reduces RMS p50 by ~3-7% across the board (more capacity =
slightly tighter sub-pixel fit), but the worst-case outlier rate
*does not improve* and harsh+blend gets *worse* (7% → 10%). The
runtime cost is 2.5× — for no benefit.

Plausible explanation: 6 sub-distributions × 26 K records ≈ 4.3 K
samples per sub-distribution. At fixed 60-epoch / cosine-LR
budget the 47 K-param model overfits its per-sub-distribution
neighbourhood instead of learning the cross-distribution structure,
while 19 K is at the sweet-spot capacity for this dataset size.

**Recommendation: v6 (19 K) is the production-leaning model.** To
push harsh+blend below 7% the right next moves are weighted-loss
oversampling of harsh+blend or a larger blend-augmented dataset
(say 50 K records), not naively scaling up the model.

Crucially, **v6 now strictly dominates spline_knot_dp on worst-case
outlier rate across all six combinations**:

| Combo | spline_knot_dp Outlier | **v6 caliper_cnn_prosac** |
|---|---:|---:|
| Normal       | 3.0%  | **0.0%** |
| Harsh        | 8.0%  | **2.0%** |
| Photo        | 3.0%  | **0.0%** |
| Normal+blend | 3.0%  | **0.0%** |
| Harsh+blend  | 16.0% | **7.0%** |
| Photo+blend  | 3.0%  | **1.0%** |

Saturating-evidence p90-κ keeps spline_knot_dp competitive on RMS p50
(0.118-0.217 vs CNN's 0.169-0.209), but the trained CNN family is
the worst-case-error winner across the OOD-extended benchmark.

## Final lab summary (CNN family, all variants)

Cross-regime numbers (100 seeds × normal / harsh / photo):

| Algorithm | RMS p50 (n / h / p) | RMS p90 | Coverage | Outlier | ms p50 |
|---|---:|---:|---:|---:|---:|
| `caliper_cnn_prosac` ⭐ | **0.166 / 0.175 / 0.161** | 0.36 / 0.46 / 0.37 | ≥99.3% | **0.0%** | **0.65** |
| `caliper_cnn_ort` (v5-lite) | 0.195 / 0.227 / 0.193 | 0.31 / 0.45 / 0.31 | ≥99.2% | 0.0% | 0.78 |
| `caliper_cnn_spline` | 0.239 / 0.273 / 0.238 | 0.32 / 0.45 / 0.32 | ≥98.9% | 0.0% | 0.64 |
| `caliper_cnn_cross` (cv::dnn) | 0.195 / 0.227 / 0.193 | 0.31 / 0.45 / 0.31 | ≥99.2% | 0.0% | 1.97 |
| `spline_knot_dp` (ref) | 0.118 / 0.163 / 0.124 | 0.26 / 0.44 / 0.27 | ≥93.2% | 1.7-8.3% | 4.5-5.3 |

The CNN family of algorithms — `caliper_cnn_prosac` in particular —
occupies a previously empty spot on the lab's quality/speed Pareto
frontier: **sub-millisecond, sub-pixel, zero-outlier in all noise
regimes including harsh**. spline_knot_dp still has a tighter median
RMS (0.118 vs 0.166) but trips on 8% of harsh scenes vs caliper_cnn's
0%.

Recipe stack (reproducible from this repo):

  1. `dump_caliper_dataset --scene-records` → 16-caliper batches
  2. `train.py` (60 epochs CPU, ~10 min) → CrossCaliperEdgeNet
  3. `export_onnx.py` → ONNX (auto-detects in_ch from checkpoint)
  4. ONNX Runtime backend (XINSP_ENABLE_ORT) → 0.7 ms/scene
  5. CNN+DP+adaptive-poly with safety net → caliper_cnn_prosac

## CNN + polynomial post-fit (`caliper_cnn_prosac`)

After v5-lite proved cross-caliper inference reliable (0% outlier
in all regimes), the next question was whether to keep emitting
piecewise-linear output between the 16 DP picks or fit a smooth
parametric curve through them. The lab tested three post-fit
strategies on the same CNN+DP front-end:

| Post-fit | RMS p50 (avg 3 regimes) | RMS p90 (avg) | Outlier | ms |
|---|---:|---:|---:|---:|
| Linear interp (cnn_ort)            | 0.205 | 0.36 | 0% | 0.78 |
| Adaptive poly LSQ + safety net     | **0.167** | 0.40 | **0%** | 0.65 |
| Natural cubic spline + safety net  | 0.250 | 0.36 | 0% | 0.64 |

Polynomial post-fit wins on RMS p50 by 19% over plain linear interp,
and beats the natural cubic spline on the median by an even larger
margin. **The spline is counter-intuitively worse on p50** because
natural-cubic interpolation through 16 noisy DP picks (CNN sub-pixel
σ ≈ 0.1 px) overshoots between knots — a smoothing spline (P-spline
with λ regulariser) would do better but adds complexity.

### Safety net design

Without a guard, polynomial deg-3 LSQ overshoots in ~5% of scenes
(boundary cubic blow-up despite C¹ linear extrapolation outside the
inlier x-span). The safeguard runs after fit:

```cpp
constexpr double DEV_THR_PX = 3.0;
bool safe = true;
for (int x = ceil(x_min_in); x <= floor(x_max_in); x += 4) {
    double dev = std::abs(poly_eval(x) - linear_interp(x));
    if (dev > DEV_THR_PX) { safe = false; break; }
}
if (!safe) return linear_interp_output;
return poly_output;
```

I.e. compare polynomial output to a linear-interp baseline through
the DP picks; if they disagree by more than 3 px anywhere, the
polynomial is rejected. Catches every overshoot scene (5% → 0%
outlier in all three regimes).

### Why PROSAC didn't help here (negative result)

Initial attempt: PROSAC sampling over the 48 raw NMS hits
(16 × top-3 per caliper). Idea: pick polynomial fit from highest-
probability minimal samples, score by Σ_inlier prob × coverage.

Outcome: 17% outlier scenes — far worse than DP+linear's 0%. PROSAC
fundamentally cannot reject false-stripe co-occurrences as well as
sparse DP because it operates on points, not on per-caliper smoothness.
With ~80% of harsh-mode calipers seeing a stripe peak, PROSAC's
minimal samples sometimes draw 4 stripe peaks that *all* pass the
score function (high CNN prob, decent caliper coverage), but produce
a polynomial that traces the stripe band rather than the curve.

Switching to "DP first, then poly LSQ over the 16 DP picks" fixed
this — DP enforces per-caliper smoothness which PROSAC structurally
lacks. The "caliper_cnn_prosac" name is now a misnomer (no random
sampling phase), kept for benchmark continuity.

## ONNX Runtime backend (`caliper_cnn_ort`)

The same CrossCaliperEdgeNet ONNX runs through three inference
backends in this lab:

| Backend | ms p50 (avg 3 regimes) | Notes |
|---|---:|---|
| Python PyTorch (CPU) | ~1.8 | Reference; not in lab |
| `cv::dnn` (OpenCV built-in) | 1.97 | `caliper_cnn_cross` |
| **ONNX Runtime** | **0.78** | `caliper_cnn_ort` ⭐ |

ONNX Runtime is **2.5× faster than cv::dnn** on the same model.
The lab's CMakeLists auto-detects ORT in the user's NuGet cache
(or via `ORT_ROOT`); when found, defines `XINSP_ENABLE_ORT` and
post-build copies `onnxruntime.dll` next to the lab executable.
Without ORT the algo cleanly stubs out (returns empty / 100% no-hit
in the benchmark table).

Implementation gotchas captured in commit history:
- `Ort::Env` cannot be statically initialised on Windows — DLL
  loader race causes a segfault before main(). Lazy-construct via
  `std::unique_ptr` inside the first `get_ort()` call.
- `ORTCHAR_T = wchar_t` on Windows; ONNX path needs widening.
- Single-thread (`SetIntraOpNumThreads(1)`) is fastest at our
  47-K-param model size; thread-pool overhead would dominate.

## Trained-CNN v5 — cross-caliper feature exchange

The biggest single quality jump in any lab algorithm to date.
v4 already established that wide-context per-caliper (CAL_W=15)
gets to 0.0% outlier on normal+photo. v5 fixes the remaining 1%
on harsh by letting calipers exchange features mid-network.

### Mechanism

The earlier per-caliper CNN sees only one caliper at a time. False
stripes that span 3-5 calipers fool each one independently — the
downstream sparse DP can correct some failures via smoothness, but
not all.  v5 inserts a 2-D Conv2d mixing layer between the
per-caliper feature extractor and the final logit head:

```
Input [B_scene, K=16 calipers, C=15 cols, H=80 y]
  ↓
Per-caliper Conv1d×2 (15→16→32)            # local features in y
  ↓ reshape to [B, 32, K, H]
Cross-caliper Conv2d×2 (kernel 3×7)        # mix neighbour calipers
  ↓ reshape back to [B·K, 32, H]
Per-caliper Conv1d 1×1 → logits [B, K, 80]
```

The kernel `(3 across calipers, 7 in y)` lets each caliper see its
±1 neighbour's features directly. Two stages → effective receptive
field ±2 calipers (~40 px in image x).

Total parameters: 47 K (vs 11 K for v4). 1.8 ms PyTorch forward per
scene; 2.6 ms in cv::dnn (slightly heavier than v4's 0.9 ms because
of the 2-D conv + larger feature volume).

### Numbers (100 seeds × 3 noise regimes)

| Regime | Algorithm | RMS p50 | RMS p90 | Coverage | Outlier | ms p50 |
|---|---|---:|---:|---:|---:|---:|
| normal | `caliper_cnn_cross` (v5) | 0.196 | 0.293 | **99.8%** | **0.0%** | 2.6 |
|        | `caliper_cnn` (v4)        | 0.224 | 0.390 | 99.6% | 0.0% | 0.9 |
|        | `spline_knot_dp`          | 0.118 | 0.259 | 99.3% | 3.0% | 4.0 |
| harsh  | `caliper_cnn_cross` (v5) | 0.228 | 0.396 | **99.5%** | **0.0%** | 2.7 |
|        | `caliper_cnn` (v4)        | 0.304 | 0.524 | 98.7% | 1.0% | 1.2 |
|        | `spline_knot_dp`          | 0.163 | 0.438 | 93.2% | 8.0% | 4.8 |
| photo  | `caliper_cnn_cross` (v5) | 0.197 | 0.302 | **99.8%** | **0.0%** | 2.5 |
|        | `caliper_cnn` (v4)        | 0.227 | 0.342 | 99.6% | 0.0% | 1.0 |
|        | `spline_knot_dp`          | 0.124 | 0.268 | 99.3% | 3.0% | 5.4 |

**v5 is the first lab algorithm to reach 0.0% outlier in *all
three* noise regimes**, and 99.5%+ coverage in all three. The
remaining sub-pixel-precision gap to spline_knot_dp on the
cleanest scenes (0.196 vs 0.118 RMS p50) is the dense-DP advantage;
v5 closes the harsh-mode worst-case where DP also struggles.

### Why cross-caliper helps on harsh specifically

In harsh mode the lab generator emits 6–12 false stripes per scene
spanning up to 30% of W. Roughly 79% of calipers see a stripe in
their search band (vs 45% in normal). With per-caliper inference
those calipers must each independently distinguish stripe vs curve
from a 15-col context — sometimes the stripe edge texture is
ambiguous within a single caliper window.

With cross-caliper feature exchange the model learns:
- "Caliper i sees a peak at y=110, and so does i±1 at the same y,
  but neighbours i±2 do not" → likely a 60-80 px stripe whose ends
  fall outside the cluster. Suppress.
- "Caliper i sees a peak at y=110 and so does *every* other
  caliper at a smoothly varying y" → real curve. Keep.

This is a pattern hand-crafted DP can't represent at the cost level
because it operates on points, not features.

### Validation training curves

V5 trained 60 epochs on normal+harsh+photo union (13,000 scene
records, 207 K caliper-equivalents). Final val_y_L1 = **0.093 px**:

| Model         | val_y_L1 | params  | Δ vs v3 |
|---|---:|---:|---:|
| v3 (3-ch, per-caliper) | 0.350 px |  10 K | — |
| v4 (15-ch, per-caliper) | 0.156 px |  11 K | 2.2× tighter |
| **v5 (15-ch, cross-caliper)** | **0.093 px** |  47 K | **3.8× tighter** |

### Reproducing v5

```sh
# Wide-context scene-record dump
./build/Release/dump_caliper_dataset.exe lab/cnn/data/normal_sc.bin \
    --scenes 5000 --calipers-per-scene 16 --scene-records
./build/Release/dump_caliper_dataset.exe lab/cnn/data/harsh_sc.bin \
    --scenes 3000 --harsh --scene-records
./build/Release/dump_caliper_dataset.exe lab/cnn/data/photo_sc.bin \
    --scenes 5000 --photo --scene-records

cd lab/cnn
python train.py data/normal_sc.bin data/harsh_sc.bin data/photo_sc.bin \
    --epochs 60 --bs 32 --patience 10 --out caliper_edge_v5.pt
python plot_metrics.py caliper_edge_v5.pt.metrics.csv
python export_onnx.py caliper_edge_v5.pt --out caliper_edge_v5.onnx --K 16

cd ../..
XICAL_ONNX_CROSS=$(pwd)/lab/cnn/caliper_edge_v5.onnx \
    ./lab/build/Release/lab.exe --seeds 100
```

### Re-tuning when caliper geometry changes

Architectural constants baked into the model:

| Parameter | Hardcoded in | Retrain on change? |
|---|---|---|
| `CAL_W` (column count, =15) | First Conv1d weight shape | **Yes** |
| `CAL_H` (height, =80) | Soft-argmax range, RF scale | **Yes** (or sub-pixel quality degrades) |
| `N_CAL` (caliper count, =16, **cross only**) | Cross Conv2d trained-with K | No retrain, but re-export ONNX with dynamic K-axis |

For the per-caliper v4 model, N_CAL is a free batch dim — any K
works at inference without retraining.

## Trained-CNN v4 — wide-context input (CAL_W = 15)

The biggest single quality jump in the CNN line of work. Per-caliper
input shape goes from 3×80 to **15×80** — each caliper sees ±7 px of
horizontal context instead of ±1. This matters because the lab's
false stripes span up to 30 % of W (96 px); with a 3-px window the
network only ever sees the *interior* of a stripe (indistinguishable
from a real edge), but with a 15-px window it routinely catches
stripe edges (sharp transitions at *both* y = top *and* y = bottom)
which are visually distinct from the true curve (sharp y-transition
with extended dark→bright region).

Architecture unchanged otherwise (1-D CNN, 3 conv stages, 11 K
parameters — only ~1 K more than CAL_W = 3). Training data is the
same (normal + harsh + photo, 207 K records); the model trained for
~12 minutes (auto early-stop after ~42 epochs with 10-epoch
patience), val_y_L1 dropped to **0.156 px** (vs v3's 0.350 px —
2.2× tighter sub-pixel localisation).

### Numbers (100 seeds × 3 noise regimes)

| Regime | Algorithm | RMS p50 | RMS p90 | Coverage | Outlier | ms p50 |
|---|---|---:|---:|---:|---:|---:|
| normal | `caliper_cnn` (v4) | 0.224 | 0.390 | **99.6%** | **0.0%** | 0.92 |
|        | `spline_knot_dp`   | 0.118 | 0.259 | 99.3% | 3.0% | 4.0 |
|        | `caliper_ransac`   | 0.200 | 0.784 | 96.0% | 4.0% | 0.33 |
|        | `dijkstra_path`    | 0.310 | 0.498 | 96.3% | 3.0% | 0.30 |
| harsh  | `caliper_cnn` (v4) | 0.304 | 0.524 | **98.7%** | **1.0%** | 1.18 |
|        | `spline_knot_dp`   | 0.163 | 0.438 | 93.2% | 8.0% | 4.8 |
|        | `dijkstra_path`    | 0.499 | 0.814 | 62.4% | 48.0% | 0.39 |
| photo  | `caliper_cnn` (v4) | 0.227 | 0.342 | **99.6%** | **0.0%** | 1.00 |
|        | `spline_knot_dp`   | 0.124 | 0.268 | 99.3% | 3.0% | 5.4 |
|        | `caliper_ransac`   | 0.224 | 0.878 | 95.3% | 6.0% | 0.44 |

**`caliper_cnn` v4 is the first lab algorithm to reach 0.0%
outlier-scene rate on both normal and photo, and ≤ 1.0% in all
three regimes.** Coverage ≥ 98.7% across the board.

The remaining sub-pixel-precision gap to `spline_knot_dp` (0.224 vs
0.118 RMS p50 on normal) is the dense-DP advantage: spline_knot_dp
runs an exhaustive K=20 × M=160 Viterbi over saturated evidence
that CNN inference does not match. For applications where bounded
worst-case error matters more than the lowest mean, v4 is now the
strict winner; for the lowest absolute RMS, spline_knot_dp's 4 ms
DP is still in the lead.

### What wide-context buys

A 15-column window per caliper is just barely wide enough to span
half of the smallest false stripe (32 px) and roughly 16 % of the
widest (96 px). The network exploits this in two ways the original
3-column model could not:

1. **Stripe top/bottom co-detection.** A real curve has a single
   sharp y-transition (dark→bright). A stripe has two transitions
   2–3 px apart (top dark→bright, bottom bright→dark). With the
   15-col view the network sees both transitions as a *paired*
   pattern and learns to suppress the corresponding peaks.

2. **Spatial coherence within the caliper.** A spike block 5–8 px
   wide produces evidence in only ~half the caliper window's
   columns; a real curve fills all 15. The network can use this
   density signal as a discriminator the 3-col model never saw.

### Reproducing v4

```sh
# Wide-context dataset dump (CAL_W = 15)
./build/Release/dump_caliper_dataset.exe lab/cnn/data/normal15.bin --scenes 5000
./build/Release/dump_caliper_dataset.exe lab/cnn/data/harsh15.bin  --scenes 3000 --harsh
./build/Release/dump_caliper_dataset.exe lab/cnn/data/photo15.bin  --scenes 5000 --photo

cd lab/cnn
python train.py data/normal15.bin data/harsh15.bin data/photo15.bin \
    --epochs 60 --bs 256 --patience 10 --out caliper_edge_v4.pt
python plot_metrics.py caliper_edge_v4.pt.metrics.csv   # health check
python export_onnx.py caliper_edge_v4.pt --out caliper_edge_v4.onnx
```

## Trained-CNN (`algo_caliper_cnn.cpp`) v3 — multi-noise

The v1 model was trained only on normal-noise scenes; v2 added harsh;
v3 also adds photo. 60 epochs over the union (207 K records) keeps
each scene visible to the optimiser as many times as v1's normal-only
30-epoch run.

### v3 numbers (100 seeds × 3 noise regimes)

| Regime | Algorithm | RMS p50 | RMS p90 | Coverage | Outlier | ms p50 |
|---|---|---:|---:|---:|---:|---:|
| normal | `caliper_cnn` (v3) | 0.274 | 0.407 | 99.2% | **1.0%** | 0.89 |
|        | `spline_knot_dp`   | 0.118 | 0.259 | 99.3% |  3.0%  | 4.0 |
|        | `caliper_ransac`   | 0.200 | 0.784 | 96.0% |  4.0%  | 0.33 |
|        | `dijkstra_path`    | 0.310 | 0.498 | 96.3% |  3.0%  | 0.28 |
| harsh  | `caliper_cnn` (v3) | 0.371 | 0.613 | **97.4%** | **3.0%** | 1.06 |
|        | `spline_knot_dp`   | 0.197 | 0.475 | 94.0% |  7.0%  | 5.1 |
|        | `dijkstra_path`    | 0.521 | 0.741 | 62.5% | 47.0%  | 0.34 |
| photo  | `caliper_cnn` (v3) | 0.278 | 0.407 | 98.7% | **2.0%** | 0.99 |
|        | `spline_knot_dp`   | 0.124 | 0.268 | 99.3% |  3.0%  | 7.1 |
|        | `caliper_ransac`   | 0.224 | 0.878 | 95.3% |  6.0%  | 0.45 |
|        | `dijkstra_path`    | 0.315 | 0.518 | 96.0% |  3.0%  | 0.37 |

`caliper_cnn` (v3) is the **only lab algorithm with outlier rate
≤ 3% in all three noise regimes**. RMS p50 trails `spline_knot_dp`
by ~0.15 px on the cleanest scenes — the dense knot-DP retains the
sub-pixel-precision crown — but caliper_cnn covers the
speed/robustness Pareto frontier at ~5× the throughput.

Reproducing v3:
```sh
./build/Release/dump_caliper_dataset.exe lab/cnn/data/normal.bin --scenes 5000
./build/Release/dump_caliper_dataset.exe lab/cnn/data/harsh.bin  --scenes 3000 --harsh
./build/Release/dump_caliper_dataset.exe lab/cnn/data/photo.bin  --scenes 5000 --photo
cd lab/cnn
python train.py data/normal.bin data/harsh.bin data/photo.bin --epochs 60
python export_onnx.py caliper_edge.pt --out caliper_edge.onnx
```

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

## Learned per-caliper edge filter (`algo_caliper_cnn.cpp`)

A 30K-parameter 1-D CNN replaces the hand-crafted gradient + NMS
peak detector inside each caliper. Inputs the 3 columns of the
caliper ROI as 3 channels, height H=80; outputs an 80-element
edge-probability array; NMS top-3 sub-pixel peaks; sparse Viterbi
DP across calipers with α·|Δy| smoothness.

Training pipeline lives in `lab/cnn/` (PyTorch). Dataset built by
`dump_caliper_dataset.exe`: 5000 normal scenes × 16 calipers each,
keeping only those whose GT y falls inside the ROI. ~80 K records.
30 epochs, batch 256, Adam lr=1e-3, ≈4 minutes on CPU. Loss = MSE
on Gaussian-smeared heatmap + soft-argmax L1 on sub-pixel y.
Validation L1 converges to **0.196 px** mean abs sub-pixel error.

### Numbers (100 seeds, normal noise) — model trained only on this regime

| Algorithm | RMS p50 | RMS p90 | Coverage | Outlier | ms p50 |
|---|---:|---:|---:|---:|---:|
| `caliper_cnn`     | 0.236 | **0.298** | **99.9%** | **0.0%** | **2.3** |
| `spline_knot_dp`  | **0.118** | 0.259 | 99.3% | 3.0% | 4.3 |
| `caliper_ransac`  | 0.200 | 0.784 | 96.0% | 4.0% | 0.33 |
| `dijkstra_path`   | 0.310 | 0.498 | 96.3% | 3.0% | 0.27 |
| `tensor_voting`   | 0.221 | 0.473 | 74.7% | 92.0% | 9.4 |

**0.0% outlier and 99.9% coverage** — first algorithm in the lab
that fails on no scene at all. Sub-pixel RMS is dominated by
spline_knot_dp's heavy DP (0.118 vs 0.236), but the CNN's outputs
are **never wildly wrong**. For any application where bounded
worst-case error matters more than the lowest mean, caliper_cnn
is the new winner. And it does this in **half the runtime** of
spline_knot_dp (2.3 ms vs 4.3 ms).

### Numbers (100 seeds, harsh noise) — same normal-only-trained model

| Algorithm | RMS p50 | RMS p90 | Coverage | Outlier | ms p50 |
|---|---:|---:|---:|---:|---:|
| `caliper_cnn`     | 0.295 | 0.513 | **96.7%** | 8.0% | **2.0** |
| `spline_knot_dp`  | 0.197 | 0.475 | 94.0% | **7.0%** | 4.2 |
| `dijkstra_path`   | 0.521 | 0.741 | 62.5% | 47.0% | 0.31 |

Trained only on normal scenes, the CNN still posts **best-in-class
coverage** and outlier rate within 1 pp of the heavy hand-crafted
DP, at 2× the speed. Training a separate harsh-mode model (or one
on the union) is expected to push outliers further down, since the
model has never seen 20-100-spike scenes during training.

### Why this works

The classical caliper peak extractor uses a 3-tap or 5-tap
Gaussian-derivative filter — completely identical for "bright
spike row" and "real edge row". Both look like high `|∂I/∂y|`.
The downstream RANSAC / DP / coverage tricks compensate by
exploiting *cross-caliper coherence*.

A learned 1-D filter sees the entire 80-px caliper at once and can
exploit *local* features the hand-crafted filter cannot:
- Spike top edges have a **matching opposite-polarity bottom edge**
  4–8 px below them; the curve does not.
- Spike-vs-curve texture differs even within a single column.

The CNN therefore filters most spike-induced false peaks at the
peak-extraction stage, leaving the downstream sparse DP a much
cleaner candidate pool. The result is the first lab algorithm that
**fails on no normal scene at all**.

### Reproducing

```sh
# 1. Build the dump tool (already part of lab CMake)
cmake --build lab/build --config Release

# 2. Generate dataset
./lab/build/Release/dump_caliper_dataset.exe lab/cnn/data/normal.bin \
    --scenes 5000 --calipers-per-scene 16

# 3. Train (Python venv)
cd lab/cnn
python -m venv .venv
.venv/Scripts/pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install numpy onnx
.venv/Scripts/python.exe train.py data/normal.bin --epochs 30 --bs 256 --out caliper_edge.pt
.venv/Scripts/python.exe export_onnx.py caliper_edge.pt --out caliper_edge.onnx

# 4. Run lab benchmark with the model
cd ../..
XICAL_ONNX="$(pwd)/lab/cnn/caliper_edge.onnx" ./lab/build/Release/lab.exe --seeds 100
```

Without the env var, `caliper_cnn` skips itself (100% no-hit) and
the rest of the benchmark runs unchanged.

## Constrained-polynomial search variants (`algo_constrained_*.cpp`)

A user-driven follow-up: instead of free-form spline knots, fit a
**deg-3 polynomial with explicit derivative box constraints** —
the user wants to bound max |p'|, |p''|, |p'''| in pixel-space, so
the curve has user-specified maximum complexity. Three search
methods over the same constrained problem:

  * **`cpoly_grid`** — coarse-to-fine grid search in (a₁, a₂, a₃)
    with inner 1-D search for a₀. Within the discretisation it is
    globally optimal — the "ground truth" for what the constrained
    polynomial can do.
  * **`cpoly_ransac`** — random samples of 4 strong-evidence pixels,
    fit the unique cubic through them, reject constraint violators,
    score by integrated saturated evidence. PROSAC-flavoured.
  * **`cpoly_knot_dp`** — same DP backbone as `spline_knot_dp` but
    with the soft `λ·curv²` penalty replaced by *hard* box
    constraints derived from `slope_max`, `curv_max`. Curve is
    still piecewise-linear over knots, not a polynomial — but the
    constraint structure is the same as the polynomial fit.

All three share `constrained_poly_common.hpp` (saturated evidence,
`PolyConstraints` struct, `score_poly`, `satisfies` constraint
check). PolyConstraints defaults: slope ≤ 1.5 px/px, curv ≤ 0.05
px/px², jerk ≤ 0.005 px/px³ — loose enough to admit every curve
the lab generator produces.

### Numbers (100 seeds, normal noise)

| Algorithm | RMS p50 | RMS p90 | Coverage | Runtime p50 | Outlier |
|---|---:|---:|---:|---:|---:|
| `spline_knot_dp` (reference) | 0.193 | 0.456 | 98.1% | 1.80 ms | 4% |
| `cpoly_knot_dp`              | 0.209 | 0.419 | 93.7% | 3.80 ms | 12% |
| `cpoly_grid`                 | 0.707 | 0.849 | 81.6% | 39.7 ms | 25% |
| `cpoly_ransac`               | 0.761 | 1.168 | 49.0% | 0.35 ms | 62% |

### Numbers (100 seeds, harsh noise)

| Algorithm | RMS p50 | RMS p90 | Coverage | Runtime p50 | Outlier |
|---|---:|---:|---:|---:|---:|
| `spline_knot_dp` (reference) | 0.301 | 0.849 | 74.9% | 1.95 ms | 33% |
| `cpoly_knot_dp`              | 0.384 | 0.850 | 52.3% | 4.06 ms | 64% |
| `cpoly_grid`                 | 0.719 | 1.147 | 55.4% | 40.2 ms | 54% |
| `cpoly_ransac`               | 1.104 | 1.203 |  8.2% | 0.38 ms | 100% |

### What we learned

1. **`cpoly_knot_dp` ≈ `spline_knot_dp`.** Replacing the soft
   curvature *penalty* with a *hard box* loses some flexibility
   (12% vs 4% outlier on normal) but produces nearly the same
   RMS. The DP search-method is the right tool either way.

2. **Grid search over polynomial coefficients is the wrong tool.**
   Even at 9 bins per axis × 25 inner-a₀ bins, `cpoly_grid` posts
   RMS p50 0.707, outlier 25%, runtime 40 ms. Bumping to 15³ × 33
   (verified, not committed) lifts to 179 ms but only drops RMS to
   0.576 and outlier to 18%. The coefficient distribution of real
   scene curves is *dense*, not sparse — the grid never lands close
   enough to the true peak, and refinement around the wrong coarse
   peak doesn't recover.

3. **`cpoly_ransac` (4-point exact-fit cubic) fails completely** on
   harsh (100% outlier). Random 4-point cubics through edge-pixel
   candidates are too noisy: a single sub-pixel jitter on any of
   the 4 points throws the whole cubic off by orders more than the
   tolerance. The classical caliper-ransac pipeline avoids this by
   doing LSQ over many inliers (not 4-point exact fit) plus IRLS,
   and by sampling from per-caliper top-K peaks rather than the
   raw evidence map.

4. **Expressed via control-points / knots, the constrained problem
   becomes well-conditioned for DP.** The `cpoly_knot_dp` baseline
   demonstrates this — the same PolyConstraints box (slope/curv
   bounds) is *easier* to enforce on adjacent y-bin differences
   than on monomial coefficients, because the differences correspond
   directly to local derivatives. This is the variation-diminishing
   property of B-spline / Bernstein bases, encoded as a discrete
   bin-difference inequality.

### Recommendation

For production "constrained polynomial" use the `cpoly_knot_dp`
shape — output piecewise-linear knots (or, if true C² output is
needed, post-fit a cubic spline through the K knots after DP).
The grid-search version is a reference baseline only; in practice
its 40 ms cost buys lower quality than spline-DP at 2 ms.

The single biggest finding is that **monomial-coefficient search
spaces are ill-suited to dense data** — the obvious-looking
"discretise the polynomial" approach is not a viable replacement
for either RANSAC or knot DP.

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

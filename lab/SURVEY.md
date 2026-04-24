# Survey: Robust Detection of Weak Continuous Curved Edges Amidst Strong Isolated Noise

_Independent survey produced by an Opus research agent. The agent was **not** told about the current caliper + RANSAC + polynomial approach — the prompt described the problem abstractly and asked for a free survey._

The core tension in the problem is that **local gradient magnitude is the wrong signal**: the noise wins on that metric. Every viable approach exploits the one property that separates signal from noise — **spatial coherence / continuity** of the true edge over an extended path, versus the isolated nature of noise features. Below are 14 approaches organised by family, with mechanisms, noise-rejection logic, failure modes, cost, and sub-pixel capability.

---

## 1. Classical / Filter-Based Approaches

### 1.1 Canny with Hysteresis + Connected-Component Length Filtering

**Mechanism.** Standard Canny: Gaussian blur (sigma tuned so the weak edge survives smoothing but isolated spikes are attenuated), compute gradient magnitude/direction, non-maximum suppression along gradient normal, then *double-thresholded hysteresis*. The key extension for your problem is setting the **low hysteresis threshold below the weak-edge gradient** and the **high threshold above it**, then post-filtering connected edge chains by length and curvature variance. A chain of N connected pixels surviving the low threshold is kept only if it touches a seed above the high threshold *and* has length >> typical noise blob perimeter.

**Noise resistance.** Isolated noise spikes form short, closed, or star-shaped components — they fail the length filter. True edge is a long 1-D string.

**Failure modes.** If the weak edge has gaps larger than hysteresis can bridge (Canny only bridges 8-connected neighbours), it fragments. Strong noise adjacent to the curve can fuse into the chain. Sensitive to `sigma` choice.

**Cost.** Real-time trivially (O(N) on image pixels).

**Sub-pixel.** Yes — Canny's NMS can be augmented with parabolic interpolation across the gradient-normal triplet to get ~0.1-pixel localisation.

**Keywords.** Canny 1986; "edge linking"; "hysteresis thresholding"; Devernay sub-pixel edge detector (1995) — specifically designed to extract sub-pixel chains.

---

### 1.2 Steerable / Oriented Matched Filters

**Mechanism.** Convolve the image with a bank of elongated oriented filters (second-derivative-of-Gaussian, Gabor, or anisotropic kernels) at many orientations and a few scales. Each filter is *long* along its preferred orientation (e.g. 15–25 px) and *narrow* across it (1–3 px). Take the per-pixel maximum response across orientations; this yields an "oriented edge energy" map. Freeman & Adelson's steerable filters let you synthesise arbitrary orientations from a small basis, making this cheap.

**Noise resistance.** An isolated blob, being radially symmetric, produces only moderate response across all orientations — no directional peak. The weak but *extended* edge aligns with one filter orientation over many pixels, accumulating response along its length. The elongated support integrates the weak gradient over many pixels, boosting SNR by ~sqrt(filter length).

**Failure modes.** Sharp curvature (below filter length's radius of curvature) is blurred/missed. Two parallel nearby edges merge.

**Cost.** Real-time with separable or steerable bases; O(N · K) for K orientations.

**Sub-pixel.** Yes — peak response gives orientation and sub-pixel offset via quadratic interpolation across the filter-response tensor.

**Keywords.** Freeman & Adelson 1991 "steerable filters"; Perona "steerable-scalable"; Jacob & Unser "design of steerable filters"; "oriented energy".

---

### 1.3 Phase Congruency

**Mechanism.** Rather than gradient magnitude, measure the *local phase coherence* of the image's Fourier components. At a true edge, Fourier components across scales are phase-aligned (congruent); the measure is dimensionless and contrast-invariant. Kovesi's algorithm uses log-Gabor wavelets at multiple scales and orientations and computes the phase-congruency scalar (plus orientation).

**Noise resistance.** A weak edge that is a true step has the same phase-congruent structure as a strong one — it survives even with tiny amplitude. An isolated blob has different phase structure (more peak-like than step-like) and partially destructively interferes across scales. Noise amplitude is downweighted by the normalisation.

**Failure modes.** Computationally heavy; sensitive to the noise-floor estimator; can give spurious response at texture junctions.

**Cost.** Offline or high-end real-time (many wavelet convolutions).

**Sub-pixel.** Yes, with phase-based interpolation.

**Keywords.** Kovesi 1999 "Image features from phase congruency"; Morrone & Owens; log-Gabor.

---

## 2. Variational / Energy-Minimisation Methods

### 2.1 Active Contours (Snakes) — Kass, Witkin, Terzopoulos

**Mechanism.** Parameterise a curve v(s) = (x(s), y(s)) and minimise
E = ∫ (α|v'|² + β|v''|²) ds  −  λ ∫ |∇I(v(s))|² ds.
The internal energy (first two terms) penalises stretching and bending, enforcing smoothness. The external energy pulls the curve toward high-gradient locations. Solve by gradient descent on control points (finite-difference or FEM).

**Noise resistance.** The bending/stretching prior prevents the curve from darting off to latch onto an isolated noise spike — the energy cost of the detour exceeds the gradient reward, provided β is tuned so the curve's stiffness scales with noise spike size. Integration of external energy along the contour averages out small-scale noise.

**Failure modes.** Requires reasonable initialisation near the true edge. Gets stuck in local minima. Tends to shrink (classic snake shrink-bias).

**Cost.** Real-time for a single curve with modest control points.

**Sub-pixel.** Yes, directly — the curve is continuous in R².

**Keywords.** Kass, Witkin, Terzopoulos 1988; GVF snakes (Xu & Prince) for extended capture range; B-spline snakes.

---

### 2.2 Geodesic Active Contours / Level-Set Methods

**Mechanism.** Reformulate the snake as finding a curve of minimal weighted length in a Riemannian metric where distance is short where edges are strong: minimise ∫ g(|∇I|) ds, with g a decreasing edge indicator. Embed the curve as the zero level-set of a function φ(x,y,t) and evolve ∂φ/∂t = g(|∇I|)(κ + c)|∇φ| + ∇g · ∇φ.

**Noise resistance.** The curvature term κ smooths small wiggles; the metric weighting means the curve naturally avoids short, isolated high-gradient features because the length-weighted cost favours long coherent structures. Topologically flexible.

**Failure modes.** Similar initialisation issues as snakes; slow; parameter-sensitive.

**Cost.** Offline or near-real-time with narrow-band methods.

**Sub-pixel.** Yes — zero level-set is interpolated continuously.

**Keywords.** Caselles, Kimmel, Sapiro 1997; Chan–Vese (region-based alternative); Osher–Sethian level sets.

---

### 2.3 Mumford–Shah Segmentation

**Mechanism.** Minimise E(u, K) = ∫_{Ω\K} |∇u|² dx + μ · length(K) + λ ∫ (u − I)² dx over a piecewise-smooth approximation u and discontinuity set K. The recovered K contains the edges.

**Noise resistance.** The length penalty discourages short isolated boundaries; the region-smoothness term prefers coherent regions separated by *long* discontinuities. Isolated noise spikes are absorbed into u's smooth part rather than promoted to K if their contribution is less than μ · (perimeter).

**Failure modes.** NP-hard in general; solved approximately via Ambrosio–Tortorelli phase-field, graph cuts, or convex relaxations. Parameter tuning is delicate.

**Cost.** Offline.

**Sub-pixel.** The edge set K can be extracted with sub-pixel accuracy via phase-field zero-crossings.

**Keywords.** Mumford–Shah 1989; Ambrosio–Tortorelli; Chan–Vese.

---

## 3. Graph-Based & Dynamic-Programming Methods

### 3.1 Minimal-Path / Fast-Marching Extraction

**Mechanism.** Define a cost image c(x,y) that is *low* where edge evidence is high (e.g. c = 1/(ε + |∇I|) or phase-congruency inverse). Given two endpoints (user-clicked or detected), compute the minimal-cost path using Dijkstra or fast-marching (which solves the eikonal equation |∇U| = c, giving a continuous arrival-time map; backtracking from endpoint gives the geodesic).

**Noise resistance.** The globally optimal path integrates cost over its length. A detour through an isolated strong-gradient blob adds path length (high cost in surrounding low-gradient pixels) that outweighs the local gain. Coherent weak edges offer a consistently lower-cost corridor.

**Failure modes.** Needs endpoints or a seed region. With bad cost design, shortcuts through background may win.

**Cost.** Real-time (Dijkstra O(N log N); fast-marching O(N log N)).

**Sub-pixel.** Fast-marching provides continuous arrival-time — gradient descent on U from the endpoint gives a sub-pixel path.

**Keywords.** Cohen & Kimmel 1997 "Global minimum for active contour models"; Sethian fast-marching; "Intelligent Scissors" (Mortensen & Barrett); Livewire.

---

### 3.2 Dynamic Programming along a Scan Direction

**Mechanism.** If the curve is known to be a function y = f(x) (or ρ = f(θ) in polar), treat it as a shortest-path problem on a grid: node (x, y) has cost −edge-evidence(x, y); transitions (x, y) → (x+1, y+dy) with dy ∈ {−K, …, K} have a smoothness cost proportional to |dy − dy_prev| or (dy)². Solve via Viterbi-style DP in O(W · H · K).

**Noise resistance.** The smoothness transition cost prevents jumping to an isolated blob that is off the main curve. The global optimum integrates evidence across the full image width.

**Failure modes.** Requires a monotone parameterisation direction (no self-occlusions or vertical tangents in the chosen axis). Bias toward the scan direction.

**Cost.** Real-time — classic retina-layer segmentation and lane-detection algorithm.

**Sub-pixel.** Yes with parabolic interpolation around the optimal y at each x.

**Keywords.** Chiu 2010 "Automatic segmentation of seven retinal layers"; Amir 1990 DP-snakes; Viterbi; "shortest-path edge".

---

### 3.3 Graph Cuts with Elongation Priors

**Mechanism.** Binary-label the image (edge vs. non-edge) by min-cut on a graph with unary costs from edge evidence and pairwise costs encoding a ribbon/contour prior. Advanced variants use higher-order potentials or curvature regularisation (Schoenemann et al.) that directly penalise curvature of the edge set rather than length — important because pure length penalties shrink curves.

**Noise resistance.** Curvature regularisation strongly penalises the sharp turns required to visit isolated blobs. Global optimisation avoids greedy mistakes.

**Failure modes.** Curvature regularisation requires lifted graphs (edge-pair nodes), ballooning memory.

**Cost.** Offline for curvature-regularised versions; near-real-time for simple length-regularised.

**Sub-pixel.** Only on a sub-pixel lifted grid.

**Keywords.** Boykov–Kolmogorov; Schoenemann, Kahl, Cremers 2009 "curvature regularity".

---

## 4. Statistical / Probabilistic Approaches

### 4.1 A Contrario / Helmholtz-Principle Detection (Desolneux, Moisan, Morel)

**Mechanism.** Under a null hypothesis that gradient orientations are i.i.d. uniform, compute for every candidate curve (chain of aligned pixels) the *number of false alarms* (NFA): the expected number of such chains that would appear by chance in a pure-noise image. NFA = N_tests · P(chain of length k with orientation coherence ≥ observed). Detections are curves with NFA < ε (e.g. ε = 1 means "less than one expected false alarm per image").

**Noise resistance.** Parameter-free: the threshold emerges from image dimensions. Isolated noise blobs produce short chains with low statistical significance; a long weak coherent edge accumulates improbability multiplicatively across its length, yielding tiny NFA.

**Failure modes.** Requires good local orientation estimates; null model (uniform orientations) can be violated in textured scenes.

**Cost.** Real-time for the LSD implementation (line-segment detector); a bit more for curve extensions.

**Sub-pixel.** Yes — LSD reports sub-pixel line endpoints.

**Keywords.** Desolneux, Moisan, Morel "From Gestalt Theory to Image Analysis"; LSD (von Gioi et al. 2010); "meaningful alignments".

---

### 4.2 Markov Random Field / Conditional Random Field with Contour Potentials

**Mechanism.** Each pixel has a binary edge label with unary potentials from local edge evidence. Pairwise potentials along compatible orientations reward co-linear/co-circular neighbours (a "good continuation" prior from Gestalt psychology — Parent & Zucker's co-circularity). Inference via belief propagation, TRW, or mean-field.

**Noise resistance.** A strong isolated spike has high unary but no co-circular neighbours, so posterior marginal is pulled down. Weak edge pixels reinforce each other through the pairwise potentials, climbing above threshold.

**Failure modes.** Approximate inference; sensitive to potential design.

**Cost.** Near-real-time with efficient solvers.

**Sub-pixel.** Only on sub-pixel graphs.

**Keywords.** Parent & Zucker 1989 "trace inference and curvature consistency"; Geman & Geman; "tensor voting" (Medioni, Lee, Tang) — arguably the purest instantiation of this idea, described next.

---

### 4.3 Tensor Voting

**Mechanism.** Each image token (edge pixel candidate) casts votes into its neighbourhood encoded as a second-order symmetric tensor (a "stick" tensor oriented along the token's estimated tangent). The vote field at a receiving point is a weighted sum; the decomposition of the accumulated tensor into stick/plate/ball components gives saliency for curve / surface / junction structure. The vote field is shaped so votes decay with distance and with angular deviation from a co-circular trajectory.

**Noise resistance.** An isolated token receives few supporting votes (its neighbours don't lie on a smooth continuation). Weak but aligned edge tokens reinforce each other, producing large stick saliency precisely along the true curve. The algorithm is non-iterative and has no thresholds until the final saliency step.

**Failure modes.** Initial token extraction still needs a (possibly low) threshold. 3-D extensions are costly.

**Cost.** Offline for large fields; real-time on GPU for modest sizes.

**Sub-pixel.** Saliency field is continuous; curve extraction via stick-saliency ridges gives sub-pixel contours.

**Keywords.** Medioni, Lee, Tang "A Computational Framework for Segmentation and Grouping"; "Gestalt grouping"; "co-circularity".

---

### 4.4 Particle Filter / Sequential Monte Carlo Contour Tracking

**Mechanism.** Represent the contour as a parametric curve (B-spline, polynomial) and maintain a posterior over parameters via a particle set. At each step, propose perturbations, weight by a likelihood that integrates image evidence along the curve (sum of log-gradient perpendicular to curve tangent at sample points). Resample.

**Noise resistance.** The likelihood integrates over the whole curve — an isolated noise spike contributes at most one sample's worth, while the coherent weak edge contributes at every sample. Outlier-robust likelihoods (Huber, truncated) further downweight spikes.

**Failure modes.** Proposal design is critical; high-dimensional curve spaces are hard.

**Cost.** Real-time if parameterisation is low-dimensional (< ~20 DoF).

**Sub-pixel.** Yes — posterior mean curve is continuous.

**Keywords.** Isard & Blake CONDENSATION 1998; MCMC curve sampling.

---

## 5. Learning-Based Methods

### 5.1 CNN Edge Detectors — HED, RCF, BDCN, PiDiNet

**Mechanism.** Encoder–decoder CNNs trained on BSDS500 or similar edge-annotated datasets. HED (Holistically-Nested) takes side outputs from multiple VGG stages, each supervised with the ground-truth edge map, and fuses them. Deeply-supervised multi-scale lets the network learn that coherent structure at *multiple* scales is the edge signal.

**Noise resistance.** Training data implicitly teaches the network to ignore isolated high-contrast distractors (common in natural-image training sets) in favour of coherent boundaries. Receptive fields of hundreds of pixels integrate long-range context.

**Failure modes.** Domain gap — if your image is industrial/scientific and training was natural images, spurious responses occur. Can hallucinate edges where none exist.

**Cost.** Real-time on GPU.

**Sub-pixel.** Output is a dense edge-probability map; post-processing (NMS + parabolic interp) gives sub-pixel.

**Keywords.** Xie & Tu HED 2015; RCF (Liu et al. 2017); BDCN; PiDiNet (lightweight); EDTER (transformer variant).

---

### 5.2 Transformer-Based Contour / Curve Regression (DETR-style, polyline heads)

**Mechanism.** A set-prediction transformer takes image features (CNN or ViT) and directly outputs a fixed-size set of parametric curves (Bezier/spline control points or polyline vertices). Hungarian matching to ground-truth during training. Examples: LETR for line segments, various lane-detection transformers (LSTR, CLRNet).

**Noise resistance.** Global self-attention integrates the full image; learned priors on curve shape reject distractors. Outputs are geometric entities, not pixel maps, so isolated-noise activations never reach the output.

**Failure modes.** Requires large labelled datasets of parametric curves; fixed output cardinality.

**Cost.** Real-time on GPU.

**Sub-pixel.** Native — outputs continuous coordinates.

**Keywords.** LETR (Xu et al. 2021); CLRNet lane detection; deformable DETR.

---

### 5.3 Diffusion / Score-Based Generative Priors for Edge Restoration

**Mechanism.** Treat edge extraction as inverse problem: observation y = A(x) + n where x is a clean edge map. Train (or use a pretrained) diffusion model as the prior p(x), then sample from p(x | y) via guided reverse diffusion (DPS, DDRM, ΠGDM). The learned prior encodes what plausible continuous edges look like.

**Noise resistance.** The prior assigns near-zero density to isolated-blob patterns when trained on curve-like data; sampling effectively inpaints gaps and suppresses incoherent activations. Excellent at bridging occlusions.

**Failure modes.** Heavy compute; prior may impose distributional bias; uncertainty quantification is non-trivial.

**Cost.** Offline (tens of NFEs per image), though distilled variants approach interactive speed.

**Sub-pixel.** Sample the posterior at higher resolution, or use a continuous-coordinate parameterisation.

**Keywords.** Chung DPS 2023; Kawar DDRM; "diffusion posterior sampling"; "deep image prior" (Ulyanov) as a simpler untrained cousin.

---

## 6. Quick Comparison & Recommendation Framing

| Method | Robust to isolated noise | Sub-pixel | Speed | Needs training |
|---|---|---|---|---|
| Canny + length filter | Medium | Yes (Devernay) | Real-time | No |
| Steerable / oriented filters | High | Yes | Real-time | No |
| Phase congruency | High | Yes | Slow | No |
| Snakes / GVF | High (with init) | Yes | Real-time | No |
| Geodesic active contours | High | Yes | Medium | No |
| Mumford–Shah | Very high | Yes | Offline | No |
| Minimal-path / fast-marching | Very high | Yes | Real-time | No |
| DP along scan axis | Very high | Yes | Real-time | No |
| Graph cuts w/ curvature | Very high | Partial | Offline | No |
| A contrario / LSD | Very high (parameter-free) | Yes | Real-time | No |
| MRF / CRF with co-circularity | High | Partial | Medium | Optional |
| Tensor voting | Very high | Yes | Medium | No |
| Particle filter contour | High | Yes | Real-time | No |
| CNN edge (HED/RCF/PiDiNet) | Domain-dependent | Yes (post-proc) | Real-time GPU | Yes |
| Transformer curve regression | High (in-domain) | Native | Real-time GPU | Yes |
| Diffusion prior restoration | Very high | Yes | Offline | Yes |

**Strategic framing.** For the problem (weak coherent curve vs. strong isolated spikes), the algorithmic "sweet spots" cluster around methods that **integrate evidence globally along a hypothesised path**: minimal-path (3.1), DP (3.2), a-contrario (4.1), tensor voting (4.3), and — with a good initialisation — geodesic active contours (2.2). If you have no training data and need classical robustness with sub-pixel output, **tensor voting followed by minimal-path extraction on the saliency field** is a particularly strong combined pipeline. If real-time and no training data, **steerable-filter oriented energy + LSD-style a-contrario curve grouping + Devernay sub-pixel refinement** is the most battle-tested trio. If you can afford training data, a **transformer curve-regression head** gives the cleanest geometric output.

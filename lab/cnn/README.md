# CaliperEdgeNet — learned per-caliper edge-probability filter

Tiny 1D CNN that replaces the hand-crafted `|∂I/∂y|` peak detector
inside each caliper. Trained on synthetic scenes from `lab/scene.cpp`,
exported to ONNX, then loaded by `lab/algo_caliper_cnn.cpp` at
inference time.

## Pipeline

```
+-------------------------+      +------------------------+
|  lab.exe scene gen      |  →   |  caliper_dataset.bin   |
|  (dump_caliper_dataset) |      |  uint8 ROIs + GT y     |
+-------------------------+      +------------------------+
                                            ↓
                              +--------------------------+
                              |  python train.py         |
                              |  (PyTorch, ~30 K params) |
                              +--------------------------+
                                            ↓
                              +--------------------------+
                              |  caliper_edge.pt         |
                              +--------------------------+
                                            ↓
                              +--------------------------+
                              |  python export_onnx.py   |
                              +--------------------------+
                                            ↓
                              +--------------------------+
                              |  caliper_edge.onnx       |
                              +--------------------------+
                                            ↓
+-------------------------------------+
|  lab.exe (algo_caliper_cnn)         |
|  cv::dnn::readNetFromONNX → infer   |
|  per-caliper, NMS, feed DP backend  |
+-------------------------------------+
```

## Step 1 — generate dataset

After building lab (`cmake --build build --config Release`):

```sh
./build/Release/dump_caliper_dataset.exe \
    cnn/data/normal.bin --scenes 5000 --calipers-per-scene 16
./build/Release/dump_caliper_dataset.exe \
    cnn/data/harsh.bin  --scenes 1000 --harsh
```

Resulting binary holds `(uint8 ROI[H][W], float gt_y, int polarity)`
tuples; format documented in the cpp source.

## Step 2 — train

```sh
cd cnn
pip install -r requirements.txt
python train.py data/normal.bin --epochs 50 --bs 256 --out caliper_edge.pt
```

Training time:
- CPU: ~30 minutes for 50 K records × 50 epochs
- GPU: ~3 minutes

Validation `val_y_L1` (mean abs error of soft-argmax y) should bottom
out around **0.3 px** on normal noise after 30 epochs.

## Step 3 — export ONNX

```sh
python export_onnx.py caliper_edge.pt --out caliper_edge.onnx --H 80
```

## Step 4 — run lab benchmark with the model

```sh
set XICAL_ONNX=cnn/caliper_edge.onnx
./build/Release/lab.exe --seeds 100
```

`algo_caliper_cnn` looks for the ONNX path in the `XICAL_ONNX` env
var; if absent, it skips that algorithm in the benchmark table.

## Architecture

```
input: [B, 3, H=80]    (3 columns of caliper ROI as channels, y as spatial axis)

  Conv1d 3 → 32, k=5, padding=2  →  BN  →  ReLU
  Conv1d 32 → 64, k=5, padding=2 →  BN  →  ReLU
  Conv1d 64 → 64, k=7, padding=3 →  BN  →  ReLU
  Conv1d 64 → 1, k=1

output: [B, H=80] logits → sigmoid → edge probability per y
```

~30 K parameters total. Receptive field ≈ 15 px in y, integrating
across 5–7 px neighbourhoods at three scales — enough to distinguish
"isolated bright spike row" from "the genuine extended curve edge".

## Loss

Two terms summed:

* **Heatmap MSE** between sigmoid(logits) and a Gaussian-smeared
  centerline at `gt_y` (σ=1.5 px). Encourages the right peak shape.
* **Soft-argmax L1** between `Σ p_y · y` and `gt_y`. Direct sub-pixel
  supervision — without this term the heatmap loss can settle for an
  off-by-one peak position.

Default weights `1.0 : 0.5`. The argmax loss dominates fine-tuning;
the heatmap loss prevents collapse.

## Why this works

The peak-extraction step in classic caliper pipelines uses a hand-
designed 1D filter (Sobel / Gaussian-derivative) followed by NMS. This
filter is identical for "spike pixels" and "real edge pixels" — both
look like high `|∂I/∂y|`. The downstream RANSAC/DP/coverage tricks
compensate by exploiting *cross-caliper* coherence.

A learned 1D filter sees the whole 80-px caliper at once and can
exploit features the hand-crafted version misses:

- Spike top edges have a *matching opposite-polarity bottom edge*
  4–8 px below them; the curve doesn't.
- Spikes are often isolated within the caliper; the curve is the only
  feature that's consistently present at the same y across many calipers
  (this part the cross-caliper backend already exploits, but the CNN
  can also learn the local "spike vs curve" texture difference within
  a single caliper).

The CNN therefore filters out most spike-induced false peaks at the
peak-extraction stage, leaving the downstream RANSAC/DP backend a
much cleaner candidate pool to work with.

"""Tiny 1D CNN for caliper edge-probability prediction.

Input  shape: [B, 3, H]    (3 channels = the 3 columns of the caliper ROI)
Output shape: [B, H]       (per-y logit; sigmoid → edge probability)

Total parameters ≈ 30 K, all in 1D conv. CPU inference per caliper
≈ 30 µs on a modern x86; for ~20 calipers per image, total ≈ 0.6 ms.

Architecture choices:
- Three Conv1d stages with kernel sizes 5/5/7 and 32/64/64 channels —
  enough receptive field (5+5+7 ≈ 15 px) to localise an edge in the
  presence of a nearby spike, but not so deep that it overfits the
  synthetic distribution.
- BatchNorm between stages stabilises training given the wide range of
  contrasts (δ ∈ [30, 90]) the lab generator produces.
- Output is a single 1×1 conv → 1 channel of logits, no activation.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CaliperEdgeNet(nn.Module):
    """Tiny 1-D CNN for caliper edge probability.

    Default config (10 K params, ~0.5 ms forward on CPU per batch of 30):
        in 3 → 16 (k=5)  →  ReLU
              16 → 32 (k=5)  →  ReLU
              32 → 32 (k=7)  →  ReLU
              32 → 1  (k=1)
    BatchNorm dropped — at ~10 K params the network is small enough that
    BN's overhead and ONNX-export quirks (constant-fold needed) outweigh
    its training stability benefit.

    To restore the original 40 K-param config, pass `hidden=64`.
    """

    def __init__(self, in_ch: int = 15, hidden: int = 32):
        super().__init__()
        h1 = hidden // 2  # narrower first stage
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, h1, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(h1, hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H] → logits [B, H]
        return self.net(x).squeeze(1)


class CrossCaliperEdgeNet(nn.Module):
    """Edge-prob CNN that exchanges features across calipers.

    Input shape:  [B_scene, K_caliper, in_ch, H]
                   (e.g. [scenes, 16, 15, 80])
    Output shape: [B_scene, K_caliper, H]

    Three stages:
        1. Per-caliper Conv1d feature extractor (same as CaliperEdgeNet
           up through the second conv).
        2. Cross-caliper Conv2d mixing layer: features reshaped to
           [B, hidden, K, H] and a (3 × 7) kernel slides across both
           the caliper axis and y. Lets a caliper *see* its ±1
           neighbours' features and detect false-stripe co-occurrences
           the per-caliper model is blind to.
        3. Per-caliper 1×1 head producing the H-dim logit per caliper.

    Total params depend on hidden; the default (h1=16, hidden=32) gives
    ≈ 25 K parameters — about 2× CaliperEdgeNet but still well below
    a typical UNet's million-param budget.

    Inference can be done either as `model(x)` with x.shape =
    [B, K, in_ch, H], or via `forward_flat(x_flat)` where
    x_flat.shape = [K, in_ch, H] (single image, no scene-batch dim).
    """

    def __init__(self, in_ch: int = 15, hidden: int = 32):
        super().__init__()
        h1 = hidden // 2
        # Stage 1: per-caliper local feature extractor (1-D in y).
        self.local = nn.Sequential(
            nn.Conv1d(in_ch, h1, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(h1, hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        # Stage 2: cross-caliper 2-D mix (caliper × y).
        # Kernel (3 across calipers) × (7 in y).
        self.cross = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=(3, 7), padding=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 7), padding=(1, 3)),
            nn.ReLU(inplace=True),
        )
        # Stage 3: per-caliper 1×1 logit head.
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)
        self.in_ch = in_ch
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, C, H]  → logits [B, K, H]
        B, K, C, H = x.shape
        # Per-caliper local features.
        z = self.local(x.reshape(B * K, C, H))      # [B·K, hidden, H]
        # Cross-caliper mix in 2D.
        z = z.reshape(B, K, self.hidden, H).permute(0, 2, 1, 3)  # [B, hidden, K, H]
        z = self.cross(z)
        # Back to per-caliper.
        z = z.permute(0, 2, 1, 3).reshape(B * K, self.hidden, H)
        z = self.head(z).squeeze(1)                 # [B·K, H]
        return z.reshape(B, K, H)

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        # x: [K, C, H]  → logits [K, H]   (single-image inference path)
        return self.forward(x.unsqueeze(0)).squeeze(0)


def soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample sub-pixel argmax via softmax-weighted index."""
    H = logits.shape[-1]
    p = torch.softmax(logits, dim=-1)
    idx = torch.arange(H, device=logits.device, dtype=logits.dtype)
    return (p * idx).sum(dim=-1)

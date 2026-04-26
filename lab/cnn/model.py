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

    def __init__(self, in_ch: int = 3, hidden: int = 32):
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


def soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample sub-pixel argmax via softmax-weighted index."""
    H = logits.shape[-1]
    p = torch.softmax(logits, dim=-1)
    idx = torch.arange(H, device=logits.device, dtype=logits.dtype)
    return (p * idx).sum(dim=-1)

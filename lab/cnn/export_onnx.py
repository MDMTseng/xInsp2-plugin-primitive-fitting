"""Export trained CaliperEdgeNet to ONNX.

Usage:
    python export_onnx.py <checkpoint.pt> [--out caliper_edge.onnx]
                                          [--W 15] [--H 80] [--opset 13]

The exported model takes input shape [B, W, H] (float32 in [0, 1]) and
returns logits of shape [B, H]. The C++ side runs sigmoid + parabolic
sub-pixel refinement after the network. W is auto-detected from the
first conv layer's input-channel count if not given.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch

from model import CaliperEdgeNet, CrossCaliperEdgeNet


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--out", type=str, default="caliper_edge.onnx")
    ap.add_argument("--W", type=int, default=0,
                    help="input channels (caliper width). 0 = auto-detect "
                         "from checkpoint's first conv weight.")
    ap.add_argument("--H", type=int, default=80)
    ap.add_argument("--K", type=int, default=16,
                    help="K calipers per scene (cross-caliper models only)")
    ap.add_argument("--opset", type=int, default=13)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    H = args.H if args.H else int(ckpt.get("H", 80))
    state = ckpt["state_dict"]
    # Detect model type from state-dict key prefixes.
    is_cross = any(k.startswith("local.") or k.startswith("cross.") for k in state)
    # Detect input-channel count from the first Conv1d weight (shape
    # [out_channels, in_channels, kernel]).
    first_conv_key = next(k for k in state if k.endswith(".weight") and state[k].ndim == 3)
    W = args.W if args.W else int(state[first_conv_key].shape[1])

    out_path = Path(args.out)
    if is_cross:
        # Detect hidden width from head.weight = [1, hidden, 1] and
        # cross_ky from cross.0.weight = [hidden, hidden, 3, ky].
        hidden = int(state["head.weight"].shape[1])
        cross_ky = int(state["cross.0.weight"].shape[3])
        model = CrossCaliperEdgeNet(in_ch=W, hidden=hidden, cross_ky=cross_ky)
        model.load_state_dict(state)
        model.eval()
        K = args.K
        dummy = torch.zeros(1, K, W, H, dtype=torch.float32)
        torch.onnx.export(
            model, dummy, str(out_path),
            opset_version=args.opset,
            input_names=["calipers"],
            output_names=["logits"],
            dynamic_axes={
                "calipers": {0: "batch"},
                "logits":   {0: "batch"},
            },
        )
        print(f"exported {out_path}  (CrossCaliper input [B,{K},{W},{H}] -> output [B,{K},{H}])")
    else:
        model = CaliperEdgeNet(in_ch=W)
        model.load_state_dict(state)
        model.eval()
        dummy = torch.zeros(1, W, H, dtype=torch.float32)
        torch.onnx.export(
            model, dummy, str(out_path),
            opset_version=args.opset,
            input_names=["caliper"],
            output_names=["logits"],
            dynamic_axes={
                "caliper": {0: "batch"},
                "logits":  {0: "batch"},
            },
        )
        print(f"exported {out_path}  (input [B,{W},{H}] -> output [B,{H}])")


if __name__ == "__main__":
    main()

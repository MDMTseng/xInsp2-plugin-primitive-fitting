"""Export trained CaliperEdgeNet to ONNX.

Usage:
    python export_onnx.py <checkpoint.pt> [--out caliper_edge.onnx]
                                          [--H 80] [--opset 13]

The exported model takes input shape [B, 3, H] (float32 in [0, 1]) and
returns logits of shape [B, H]. The C++ side runs sigmoid + parabolic
sub-pixel refinement after the network.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch

from model import CaliperEdgeNet


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--out", type=str, default="caliper_edge.onnx")
    ap.add_argument("--H", type=int, default=80)
    ap.add_argument("--opset", type=int, default=13)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    H = args.H if args.H else int(ckpt.get("H", 80))
    model = CaliperEdgeNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.zeros(1, 3, H, dtype=torch.float32)
    out_path = Path(args.out)
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
    print(f"exported {out_path}  (input [B,3,{H}] → output [B,{H}])")


if __name__ == "__main__":
    main()

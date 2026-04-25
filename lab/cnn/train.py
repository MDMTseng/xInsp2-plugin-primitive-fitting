"""Train CaliperEdgeNet on a dump_caliper_dataset.exe binary.

Usage:
    python train.py <dataset.bin> [--out model.pt] [--epochs 50] [--bs 256]
                                  [--lr 1e-3] [--val-frac 0.1]
"""
from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import CaliperDataset
from model import CaliperEdgeNet, soft_argmax


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("--out", type=str, default="caliper_edge.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--w-heatmap", type=float, default=1.0,
                    help="weight on Gaussian-heatmap MSE loss")
    ap.add_argument("--w-argmax", type=float, default=0.5,
                    help="weight on soft-argmax L1 loss (sub-pixel supervision)")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"device: {args.device}")
    ds = CaliperDataset(args.dataset)
    print(f"loaded {len(ds)} records, ROI = {ds.W}×{ds.H}")

    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                          num_workers=0)

    model = CaliperEdgeNet().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    out_path = Path(args.out)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_loss = 0.0
        n = 0
        for batch in train_dl:
            x = batch["x"].to(args.device)
            y = batch["target"].to(args.device)
            gt_y = batch["gt_y"].to(args.device)
            logits = model(x)
            # Heatmap MSE: encourages spatial peak shape.
            loss_hm = F.mse_loss(torch.sigmoid(logits), y)
            # Soft-argmax L1: direct sub-pixel supervision.
            yhat = soft_argmax(logits)
            loss_arg = F.l1_loss(yhat, gt_y)
            loss = args.w_heatmap * loss_hm + args.w_argmax * loss_arg
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        train_loss = sum_loss / max(1, n)

        # Validation.
        model.eval()
        v_loss = 0.0
        v_n = 0
        v_l1 = 0.0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["x"].to(args.device)
                y = batch["target"].to(args.device)
                gt_y = batch["gt_y"].to(args.device)
                logits = model(x)
                loss_hm = F.mse_loss(torch.sigmoid(logits), y)
                yhat = soft_argmax(logits)
                l1 = F.l1_loss(yhat, gt_y)
                loss = args.w_heatmap * loss_hm + args.w_argmax * l1
                v_loss += loss.item() * x.size(0)
                v_l1   += l1.item()   * x.size(0)
                v_n += x.size(0)
        v_loss /= max(1, v_n)
        v_l1   /= max(1, v_n)
        dt = time.time() - t0
        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={v_loss:.5f}  "
              f"val_y_L1={v_l1:.4f} px  ({dt:.1f}s)")
        if v_loss < best_val:
            best_val = v_loss
            torch.save({"state_dict": model.state_dict(),
                        "W": ds.W, "H": ds.H,
                        "epoch": epoch + 1, "val_loss": v_loss},
                       out_path)
            print(f"  ✓ saved {out_path}  (val_loss={v_loss:.5f})")

    print(f"\nbest val_loss={best_val:.5f} → {out_path}")


if __name__ == "__main__":
    main()

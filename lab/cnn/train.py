"""Train CaliperEdgeNet on a dump_caliper_dataset.exe binary.

Usage:
    python train.py <dataset.bin> [--out model.pt] [--epochs 50] [--bs 256]
                                  [--lr 1e-3] [--val-frac 0.1]
                                  [--patience N] [--metrics-csv path]

Real-time monitoring:
    * stdout is flushed each epoch so a `tail -f` on the redirected log
      shows per-epoch progress immediately, even when run as a
      background task.
    * `--metrics-csv` (default: <out>.metrics.csv) writes one row per
      epoch with epoch / train_loss / val_loss / val_y_L1 / time_s,
      so external tools can plot or check for plateau.
    * `--patience N` enables early-stop: after N consecutive epochs
      without a val_loss improvement, training halts and the best
      checkpoint is kept.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split

from dataset import CaliperDataset, SceneDataset, detect_format
from model import CaliperEdgeNet, CrossCaliperEdgeNet, soft_argmax


# All status prints flush immediately so background-task log files
# stay readable in real time.
def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str, nargs="+",
                    help="one or more *.bin dataset files; concatenated")
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
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N consecutive epochs without "
                         "val_loss improvement; 0 disables")
    ap.add_argument("--metrics-csv", type=str, default="",
                    help="per-epoch metrics CSV (default: <out>.metrics.csv)")
    ap.add_argument("--hidden", type=int, default=0,
                    help="model hidden width (0 = use class default)")
    ap.add_argument("--cross-ky", type=int, default=0,
                    help="cross-caliper Conv2d kernel y-size (cross models only; "
                         "0 = use class default)")
    return ap.parse_args()


def main():
    args = parse_args()
    log(f"device: {args.device}")
    fmts = [detect_format(p) for p in args.dataset]
    if len(set(fmts)) != 1:
        raise ValueError(f"mix of dataset formats: {fmts}")
    fmt = fmts[0]
    is_scene = (fmt == "XICAS")
    log(f"dataset format: {fmt}{' (cross-caliper)' if is_scene else ''}")
    cls = SceneDataset if is_scene else CaliperDataset
    parts = [cls(p) for p in args.dataset]
    if len(parts) == 1:
        ds = parts[0]
    else:
        # All sub-datasets must share W/H so model output dim is constant.
        for p in parts[1:]:
            if (p.W, p.H) != (parts[0].W, parts[0].H):
                raise ValueError(f"dataset shape mismatch: {p.W}x{p.H} vs {parts[0].W}x{parts[0].H}")
        ds = ConcatDataset(parts)
        # Expose .W/.H for downstream prints.
        ds.W, ds.H = parts[0].W, parts[0].H
    sizes = [len(p) for p in parts]
    log(f"loaded {len(ds)} records ({' + '.join(map(str, sizes))}), "
        f"ROI = {ds.W}x{ds.H}")

    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                          num_workers=0)

    model_cls = CrossCaliperEdgeNet if is_scene else CaliperEdgeNet
    model_kwargs = {}
    if args.hidden > 0:
        model_kwargs["hidden"] = args.hidden
    if is_scene and args.cross_ky > 0:
        model_kwargs["cross_ky"] = args.cross_ky
    model = model_cls(**model_kwargs).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: {model_cls.__name__}, parameters: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_path = Path(args.out)
    csv_path = Path(args.metrics_csv) if args.metrics_csv else \
               out_path.with_suffix(out_path.suffix + ".metrics.csv")
    csv_f = open(csv_path, "w", newline="", buffering=1)  # line-buffered
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["epoch", "train_loss", "val_loss", "val_y_L1_px",
                    "time_s", "saved"])
    log(f"metrics CSV → {csv_path}")

    def step_losses(batch, training: bool):
        # Returns (loss, l1_for_logging, weight_count)
        x    = batch["x"].to(args.device)
        y    = batch["target"].to(args.device)
        gt_y = batch["gt_y"].to(args.device)
        if is_scene:
            valid = batch["valid"].to(args.device)             # [B, K]
            logits = model(x)                                  # [B, K, H]
            # Heatmap MSE — masked by valid (zero where invalid).
            sig  = torch.sigmoid(logits)
            err  = (sig - y).pow(2).mean(dim=-1)               # [B, K]
            mask_sum = valid.sum().clamp(min=1.0)
            loss_hm = (err * valid).sum() / mask_sum
            # Soft-argmax L1 — also masked.
            B, K, H = logits.shape
            p = torch.softmax(logits, dim=-1)
            idx = torch.arange(H, device=logits.device, dtype=logits.dtype)
            yhat = (p * idx).sum(dim=-1)                       # [B, K]
            l1_per = (yhat - gt_y).abs()                       # [B, K]
            loss_arg = (l1_per * valid).sum() / mask_sum
            wn = int(mask_sum.item())
            l1_avg = (l1_per * valid).sum().item() / max(1, wn)
        else:
            logits = model(x)
            loss_hm = F.mse_loss(torch.sigmoid(logits), y)
            yhat = soft_argmax(logits)
            l1_avg_t = F.l1_loss(yhat, gt_y)
            loss_arg = l1_avg_t
            wn = int(x.size(0))
            l1_avg = l1_avg_t.item()
        loss = args.w_heatmap * loss_hm + args.w_argmax * loss_arg
        return loss, l1_avg, wn

    best_val = float("inf")
    epochs_since_improve = 0
    stop_reason = "completed all epochs"
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_loss = 0.0
        n = 0
        for batch in train_dl:
            loss, _, wn = step_losses(batch, training=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_loss += loss.item() * wn
            n += wn
        sched.step()
        train_loss = sum_loss / max(1, n)

        # Validation.
        model.eval()
        v_loss = 0.0
        v_n = 0
        v_l1 = 0.0
        with torch.no_grad():
            for batch in val_dl:
                loss, l1_avg, wn = step_losses(batch, training=False)
                v_loss += loss.item() * wn
                v_l1   += l1_avg     * wn
                v_n    += wn
        v_loss /= max(1, v_n)
        v_l1   /= max(1, v_n)
        dt = time.time() - t0
        improved = v_loss < best_val
        log(f"epoch {epoch+1:3d}/{args.epochs}  "
            f"train={train_loss:.5f}  val={v_loss:.5f}  "
            f"val_y_L1={v_l1:.4f} px  ({dt:.1f}s)"
            + ("  [save]" if improved else ""))
        csv_w.writerow([epoch + 1, f"{train_loss:.6f}", f"{v_loss:.6f}",
                        f"{v_l1:.4f}", f"{dt:.2f}", int(improved)])
        if improved:
            best_val = v_loss
            epochs_since_improve = 0
            torch.save({"state_dict": model.state_dict(),
                        "W": ds.W, "H": ds.H,
                        "epoch": epoch + 1, "val_loss": v_loss},
                       out_path)
        else:
            epochs_since_improve += 1
            if args.patience > 0 and epochs_since_improve >= args.patience:
                stop_reason = (f"early-stop: {args.patience} epochs without "
                               f"val_loss improvement")
                log(f"  {stop_reason}")
                break

    csv_f.close()
    log(f"\n{stop_reason}; best val_loss={best_val:.5f} -> {out_path}")


if __name__ == "__main__":
    main()

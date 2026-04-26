"""Plot training metrics from a train.py-emitted CSV.

Usage:
    python plot_metrics.py <metrics.csv> [--out plot.png]

Renders three panels:
  1. train_loss + val_loss vs epoch
  2. val_y_L1_px (sub-pixel L1) vs epoch — the eyeball metric
  3. epoch wall-time vs epoch — sanity check for slowdowns

Saves PNG (default 1200×800). When matplotlib is unavailable,
falls back to a plain-text summary so headless environments
still get the gist.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str)
    ap.add_argument("--out", type=str, default="")
    return ap.parse_args()


def load(path: Path):
    epochs, train, val, l1, dt, saved = [], [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
            l1.append(float(row["val_y_L1_px"]))
            dt.append(float(row["time_s"]))
            saved.append(int(row["saved"]))
    return epochs, train, val, l1, dt, saved


def text_summary(epochs, train, val, l1, dt, saved):
    n = len(epochs)
    if n == 0:
        return "(empty CSV)"
    best_idx = min(range(n), key=lambda i: val[i])
    return (
        f"epochs:        {n}\n"
        f"latest train:  {train[-1]:.5f}\n"
        f"latest val:    {val[-1]:.5f}\n"
        f"latest y_L1:   {l1[-1]:.4f} px\n"
        f"best val:      {val[best_idx]:.5f} (epoch {epochs[best_idx]}, "
        f"y_L1 {l1[best_idx]:.4f} px)\n"
        f"checkpoints:   {sum(saved)} saves\n"
        f"epoch time:    {dt[-1]:.1f}s (last); "
        f"{sum(dt)/max(1,n):.1f}s avg\n"
        f"total time:    {sum(dt):.1f}s = {sum(dt)/60:.1f} min\n"
    )


def main():
    args = parse_args()
    epochs, train, val, l1, dt, saved = load(Path(args.csv))
    print(text_summary(epochs, train, val, l1, dt, saved), end="")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skip PNG)")
        return
    if not args.out:
        args.out = str(Path(args.csv).with_suffix(".png"))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax1.plot(epochs, train, label="train", color="#1f77b4")
    ax1.plot(epochs, val,   label="val",   color="#d62728")
    save_x = [e for e, s in zip(epochs, saved) if s]
    save_y = [v for v, s in zip(val, saved)    if s]
    ax1.scatter(save_x, save_y, color="#2ca02c", s=20, zorder=5, label="best")
    ax1.set_ylabel("loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, l1, color="#9467bd")
    ax2.set_ylabel("val_y_L1 (px)"); ax2.grid(alpha=0.3)
    ax2.axhline(y=0.5, color="#888", linestyle="--", linewidth=0.8)
    ax3.plot(epochs, dt, color="#7f7f7f")
    ax3.set_ylabel("epoch time (s)"); ax3.set_xlabel("epoch")
    ax3.grid(alpha=0.3)
    fig.suptitle(args.csv)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"plot -> {args.out}")


if __name__ == "__main__":
    main()

"""Caliper-ROI dataset loader for the CNN edge-probability model.

Reads the binary file produced by `dump_caliper_dataset.exe`:

    8B   magic    "XICAL\0\0\0"
    4B   version  (1)
    4B   W_caliper
    4B   H_caliper
    4B   N_records
    repeat N times:
        H * W bytes  uint8 grayscale ROI (row-major)
        4B           float32 gt_y (sub-pixel y, ROI-local frame)
        4B           int32   polarity (+1 / -1)

Output `__getitem__` returns:
    image     — torch.float32 [3, H, 1] (channels_first; 3 is the W=3 stripe)
                                          ── note we treat W=3 as the channel dim
                                          since spatial resolution is 1D in y.
                Actually we keep the 3 columns as 3 input channels:
                    shape [3, H]  (1D conv along H)
    target    — torch.float32 [H]   Gaussian-smeared GT centerline
    gt_y      — float32         exact sub-pixel GT y (for soft-argmax loss)
"""
from __future__ import annotations
import io
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def gaussian_target(H: int, gt_y: float, sigma: float = 1.5) -> np.ndarray:
    ys = np.arange(H, dtype=np.float32)
    return np.exp(-((ys - gt_y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


class CaliperDataset(Dataset):
    def __init__(self, path: str | os.PathLike, sigma: float = 1.5):
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic[:5] != b"XICAL":
                raise ValueError(f"bad magic in {path}: {magic!r}")
            (self.version,) = struct.unpack("<I", f.read(4))
            (W, H, N) = struct.unpack("<III", f.read(12))
            self.W, self.H, self.N = int(W), int(H), int(N)
            record_size = self.W * self.H + 4 + 4
            payload = f.read(self.N * record_size)
        if len(payload) != self.N * record_size:
            raise ValueError(
                f"truncated dataset: expected {self.N * record_size} bytes, "
                f"got {len(payload)}"
            )
        self._buf = np.frombuffer(payload, dtype=np.uint8).reshape(
            self.N, record_size
        )
        self.sigma = sigma

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        rec = self._buf[idx]
        img = rec[: self.W * self.H].reshape(self.H, self.W).astype(np.float32) / 255.0
        # gt_y / polarity packed as float32 + int32 little-endian.
        gt_y = float(np.frombuffer(rec[self.W * self.H:].tobytes(), dtype=np.float32, count=1)[0])
        polarity = int(np.frombuffer(rec[self.W * self.H + 4:].tobytes(), dtype=np.int32, count=1)[0])
        # Channels = the W=3 columns; spatial axis = H (the y direction).
        x = torch.from_numpy(img.T)            # [W, H] = [3, H]
        target = torch.from_numpy(gaussian_target(self.H, gt_y, self.sigma))
        return {
            "x":      x,                       # [3, H]
            "target": target,                  # [H]   Gaussian heatmap
            "gt_y":   torch.tensor(gt_y, dtype=torch.float32),
            "polarity": torch.tensor(polarity, dtype=torch.float32),
        }


class SceneDataset(Dataset):
    """Loader for the 'XICAS' scene-record format produced by
    `dump_caliper_dataset --scene-records`.

    Each record contains K consecutive calipers from one scene plus
    per-caliper validity flag. __getitem__ returns:

        x       : float32 [K, W, H]   — K calipers stacked, channels-first
                                        as before (W=15 columns become
                                        the channel dim, y is spatial).
        target  : float32 [K, H]      — Gaussian-smeared GT heatmap per
                                        caliper (zeros where invalid).
        gt_y    : float32 [K]         — sub-pixel y per caliper.
        valid   : float32 [K]         — 1.0 if caliper's gt_y is in-ROI;
                                        0.0 to mask in losses.
    """

    def __init__(self, path: str | os.PathLike, sigma: float = 1.5):
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic[:5] != b"XICAS":
                raise ValueError(f"bad magic in {path}: {magic!r} (expected XICAS)")
            (self.version,) = struct.unpack("<I", f.read(4))
            (K, W, H, N) = struct.unpack("<IIII", f.read(16))
            self.K, self.W, self.H, self.N = int(K), int(W), int(H), int(N)
            self.cal_bytes = self.W * self.H + 4 + 4 + 1  # roi + gt_y + polarity + valid
            self.scene_bytes = self.K * self.cal_bytes
            payload = f.read(self.N * self.scene_bytes)
        if len(payload) != self.N * self.scene_bytes:
            raise ValueError(
                f"truncated dataset: expected {self.N * self.scene_bytes} bytes, "
                f"got {len(payload)}"
            )
        self._buf = np.frombuffer(payload, dtype=np.uint8)
        self.sigma = sigma

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        offset = idx * self.scene_bytes
        scene_buf = self._buf[offset : offset + self.scene_bytes]
        # Per-caliper unpack.
        x_arr = np.empty((self.K, self.W, self.H), dtype=np.float32)
        target = np.empty((self.K, self.H), dtype=np.float32)
        gt_y = np.empty(self.K, dtype=np.float32)
        valid = np.empty(self.K, dtype=np.float32)
        roi_size = self.W * self.H
        for k in range(self.K):
            base = k * self.cal_bytes
            roi = scene_buf[base : base + roi_size].reshape(self.H, self.W).astype(np.float32) / 255.0
            x_arr[k] = roi.T  # [H, W] -> [W, H]
            tail = scene_buf[base + roi_size : base + self.cal_bytes].tobytes()
            (gy,)  = struct.unpack_from("<f", tail, 0)
            (_pol,) = struct.unpack_from("<i", tail, 4)
            v = tail[8]
            gt_y[k]  = gy
            valid[k] = 1.0 if v else 0.0
            if valid[k] > 0:
                ys = np.arange(self.H, dtype=np.float32)
                target[k] = np.exp(-((ys - gy) ** 2) / (2.0 * self.sigma * self.sigma))
            else:
                target[k] = 0.0
        return {
            "x":      torch.from_numpy(x_arr),
            "target": torch.from_numpy(target),
            "gt_y":   torch.from_numpy(gt_y),
            "valid":  torch.from_numpy(valid),
        }


def detect_format(path: str | os.PathLike) -> str:
    """Returns 'XICAL' or 'XICAS' based on file magic. Raises on unknown."""
    with open(path, "rb") as f:
        magic = f.read(8)
    if magic[:5] == b"XICAL": return "XICAL"
    if magic[:5] == b"XICAS": return "XICAS"
    raise ValueError(f"unknown dataset format in {path}: {magic!r}")


if __name__ == "__main__":
    import sys
    fmt = detect_format(sys.argv[1])
    ds = (SceneDataset if fmt == "XICAS" else CaliperDataset)(sys.argv[1])
    print(f"format: {fmt}, loaded {len(ds)} records")
    s = ds[0]
    print({k: tuple(v.shape) if hasattr(v, "shape") else v for k, v in s.items()})

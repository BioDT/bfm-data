"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


class Acc:
    __slots__ = ("sum", "sumsq", "count", "nan", "vmin", "vmax")

    def __init__(self):
        self.sum = self.sumsq = 0.0
        self.count = self.nan = 0
        self.vmin = math.inf
        self.vmax = -math.inf

    def update(self, arr: np.ndarray):
        nan_mask = np.isnan(arr)
        self.nan += int(nan_mask.sum())
        vals = arr[~nan_mask]
        if vals.size:
            self.sum += float(vals.sum())
            self.sumsq += float((vals**2).sum())
            self.count += vals.size
            self.vmin = float(min(self.vmin, vals.min()))
            self.vmax = float(max(self.vmax, vals.max()))

    def finish(self):
        if self.count == 0:
            return {
                "mean": 0,
                "std": 0,
                "min": None,
                "max": None,
                "count_valid": 0,
                "count_nan": self.nan,
                "nan_pct": 1.0,
            }

        mean = self.sum / self.count
        var = self.sumsq / self.count - mean**2
        std = math.sqrt(max(var, 0.0))
        total = self.count + self.nan
        return {
            "mean": _py(mean),
            "std": _py(std),
            "min": _py(self.vmin),
            "max": _py(self.vmax),
            "count_valid": self.count,
            "count_nan": self.nan,
            "nan_pct": _py(self.nan / total),
        }


def compute_batch_stats(
    batch_dir: Path, out_json: Path = Path("all_batches_stats.json")
):

    files = sorted(batch_dir.glob("batch_*.pt"))
    if not files:
        raise RuntimeError("no batch files")

    acc: Dict[str, Dict[str, Acc]] = {}
    acc_per_channel: Dict[str, Dict[str, List[Acc]]] = {}

    for f in tqdm(files, desc="Scanning", unit="batch"):
        batch = torch.load(f, map_location="cpu")

        for slot, var_dict in batch.items():
            if not slot.endswith("_variables"):
                continue
            for var, ten in var_dict.items():
                matrix: np.ndarray = ten.cpu().numpy().astype(np.float64)
                # print(matrix.shape)
                if matrix.ndim == 3:
                    acc.setdefault(slot, {}).setdefault(var, Acc())
                    arr = matrix.ravel()
                    acc[slot][var].update(arr)
                elif matrix.ndim == 4:
                    # also has the channel dimension
                    channel_dim = matrix.shape[1]
                    acc_per_channel.setdefault(slot, {}).setdefault(
                        var, [Acc() for _ in range(channel_dim)]
                    )
                    for channel_id in range(channel_dim):
                        channel = matrix[:, channel_id, :, :]
                        arr = channel.ravel()
                        acc_per_channel[slot][var][channel_id].update(arr)
                else:
                    raise ValueError(f"{slot}.{var} has shape: {matrix.shape}")
                #  Shapes (2,H,W) or (2,C,H,W)

    out = defaultdict(dict)
    for slot, varmap in acc.items():
        for v, a in varmap.items():
            out[slot][v] = a.finish()
    for slot, varmap in acc_per_channel.items():
        for v, a in varmap.items():
            stats = [el.finish() for el in a]
            # stack the results dicts
            stats_stacked = {}
            for k in stats[0]:
                stats_stacked[k] = [el[k] for el in stats]
            out[slot][v] = stats_stacked
    # out = {
    #     slot: {v: a.finish() for v, a in varmap.items()} for slot, varmap in acc.items()
    # }
    # out_per_channel = {
    #     slot: {v: a.finish() for v, a in varmap.items()} for slot, varmap in acc.items()
    # }

    with open(out_json, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"Wrote {out_json}  ({len(files)} batches)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stats for per-slot batches")
    p.add_argument("--batch_dir", default="batches", type=Path)
    p.add_argument("--out", default="all_batches_stats.json", type=Path)
    args = p.parse_args()
    compute_batch_stats(args.batch_dir, args.out)

"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import json, math
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
            return {"mean": 0, "std": 0, "min": None, "max": None,
                    "count_valid": 0, "count_nan": self.nan, "nan_pct": 1.0}
        
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean**2
        std = math.sqrt(max(var, 0.0))
        total = self.count + self.nan
        return {"mean": _py(mean), "std": _py(std),
                "min": _py(self.vmin), "max": _py(self.vmax),
                "count_valid": self.count, "count_nan":self.nan,
                "nan_pct": _py(self.nan/total)}

def compute_batch_stats(batch_dir: Path, out_json: Path = Path("all_batches_stats.json")):
    pt_files = sorted(batch_dir.glob("batch_*.pt"))
    if not pt_files:
        raise RuntimeError("No batch_*.pt files found")

    first = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    slot_vars: dict[str, List[str]] = first["variable_names"]
    print("slot vars", slot_vars)
    acc: Dict[str, Dict[str, Acc]] = {
        slot: {v: Acc() for v in varlist} for slot, varlist in slot_vars.items()
    }

    for pt in tqdm(pt_files, desc="Scanning batches", unit="batch"):
        batch = torch.load(pt, map_location="cpu", weights_only=False)
        for slot, varlist in batch["variable_names"].items():
            tensor = batch.get(slot)
            if tensor is None: continue
            is_atmo = slot == "atmospheric_variables"

            for vi, var in enumerate(varlist):
                if var not in acc.setdefault(slot, {}):
                    acc[slot][var] = Acc()

                arr = tensor[:, vi]
                arr_np = arr.cpu().numpy().astype(np.float64).ravel()
                acc[slot][var].update(arr_np)

    out: Dict[str, Dict[str, Dict]] = {}
    for slot, varmap in acc.items():
        out[slot] = {v: a.finish() for v, a in varmap.items()}

    with open(out_json, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"Wrote {out_json} ({len(pt_files)} batches)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stats for per-slot batches")
    p.add_argument("batch_dir", type=Path)
    p.add_argument("--out", default="all_batches_stats.json", type=Path)
    args = p.parse_args()
    compute_batch_stats(args.batch_dir, args.out)
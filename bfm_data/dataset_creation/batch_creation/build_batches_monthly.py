"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import logging
from pathlib import Path
from typing import Dict, List
import re
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import torch
from tqdm import tqdm  
from scan_biocube import (GRID_LAT, GRID_LON,
    EXPECTED_LAT, EXPECTED_LON, NC_ENGINE, _START_DATE, _END_DATE)


VAR_CONFIG = {
    "surface_variables"       : [],#["msl", "t2m", "u10", "v10"],
    "single_variables"        : [],#["stl", "z", "lsm"],
    "atmospheric_variables"   : [], # include all
    "edaphic_variables"       : ['stl1', 'stl2', 'swvl1', 'swvl2'],
    "climate_variables"       : [],#["tp", "d2m"],
    "forest_variables"        : [], 
    "agriculture_variables"   : ['Agriculture', 'Arable', 'Cropland'],
    "land_variables"          : [],#["Land", "NDVI"],
    "species_variables"       : [],
    "redlist_variables"       : [],
    "vegetation_variables"    : [],
    "misc_variables"          : [],
}
BATCH_DIR = Path("batches")
STATS_FILE = Path("batch_stats.parquet")
DTYPE = np.float32
CHUNKS = {"time": 1, "latitude": 160, "longitude": 280}
YEAR_SUFFIX  = re.compile(r"_(\d{4})$")
MMYY_SUFFIX  = re.compile(r"_(\d{1,2})/(\d{4})$")

log = logging.getLogger("batch_builder")
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")

def _wrap_lon(ds: xr.Dataset) -> xr.Dataset:
    lon = (((ds.longitude + 180) % 360) - 180).astype(ds.longitude.dtype)
    ds = ds.assign_coords(longitude=lon).sortby("longitude")
    _, idx = np.unique(ds.longitude.values, return_index=True)
    return ds.isel(longitude=idx)

def _std_time(ds: xr.Dataset) -> xr.Dataset:
    return ds if "time" in ds.coords else ds.rename({"valid_time":"time"})

def _ensure_two(arr: np.ndarray, name: str):
    if arr.shape[0] != 2:
        raise ValueError(f"{name}: expected 2 monthly slices, found {arr.shape[0]}")
    return arr

def _harmonise_ts(ts) -> str:
    return pd.to_datetime(ts, utc=False, errors="coerce").strftime("%Y-%m-%d")

def _stack_no_channel(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack variables that have shape (2, H, W) -> (2, V, H, W)."""
    return torch.stack(tensors, dim=1) if tensors else torch.empty(0)

def _stack_with_channel(var2ten: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Stack pressure-level variables.
    Each tensor here already has shape (2, C, H, W).
    We want (2, V, C, H, W) where V = number of variables.
    """
    ordered = [var2ten[k] for k in sorted(var2ten)]
    return torch.stack(ordered, dim=1)

def _idx(lat: np.ndarray, lon: np.ndarray):
    """Return (li, lj) integer indices into GRID_LAT/LON or -1 for out-of-bounds."""
    li = np.rint((lat - 32) / 0.25).astype(int)
    lj = np.rint((lon - -25) / 0.25).astype(int)
    ok = (li >= 0) & (li < EXPECTED_LAT) & (lj >= 0) & (lj < EXPECTED_LON)
    li[~ok] = lj[~ok] = -1
    return li, lj, ok

def _maybe_fill_nan(t: torch.Tensor, fill_nan: bool) -> torch.Tensor:
    if fill_nan:
        return torch.nan_to_num(t, nan=0.0)
    return t

def _load_agriculture_slice(csv_path: Path,
                            months: list[pd.Timestamp]) -> xr.Dataset | None:
    df = pd.read_csv(csv_path)

    if df["Longitude"].max() > 180:
        df["Longitude"] = ((df["Longitude"] + 180) % 360) - 180

    year_cols = {int(re.search(r"(\d{4})$", c).group(1)): c
                 for c in df.columns if re.search(r"(\d{4})$", c)}

    wanted = ["Agriculture", "Agriculture_Irrigated", "Arable", "Cropland"]
    coords = {"time": [m.to_numpy() for m in months],
              "latitude": GRID_LAT, "longitude": GRID_LON}

    ds_vars = {}
    for var in wanted:
        sub = df[df["Variable"] == var]
        if sub.empty: continue

        li, lj, ok = _idx(sub["Latitude"].to_numpy(),
                          sub["Longitude"].to_numpy())
        grids=[]
        for ts in months:
            yr = ts.year
            col = year_cols.get(yr)
            grid = np.zeros((EXPECTED_LAT, EXPECTED_LON), dtype=DTYPE)
            if col:
                vals = sub.loc[ok, col].astype(DTYPE).to_numpy()
                grid[li[ok], lj[ok]] = vals
                log.debug("%s: %s %d written nz=%d",
                          csv_path.name, var, yr, int((vals!=0).sum()))
            grids.append(grid)

        ds_vars[var] = xr.DataArray(
            np.stack(grids, axis=0), coords=coords,
            dims=("time","latitude","longitude")
        )

    return xr.Dataset(ds_vars).chunk(CHUNKS) if ds_vars else None

# To be used for with the Combined files
def _load_csv_tile(csv_path: Path,
                   variables: list[str],
                   t0: pd.Timestamp,
                   t1: pd.Timestamp) -> xr.Dataset | None:
    df = pd.read_csv(csv_path).rename(
        columns={"Latitude": "latitude", "Longitude": "longitude"}
    )
    log.info("Reading CSV %s with columns: %s", csv_path.name, list(df.columns)[:10])

    if "agriculture" in csv_path.name.lower() and "Variable" in open(csv_path).readline():
        return _load_agriculture_slice(csv_path, [t0, t1])
    
    if df["longitude"].max() > 180:
        df["longitude"] = ((df["longitude"] + 180) % 360) - 180
        print("Wrapped longitudes to −180…180 in %s", csv_path.name)

    df["latitude"]  = np.round(df["latitude"]  / 0.25) * 0.25
    df["longitude"] = np.round(df["longitude"] / 0.25) * 0.25
    df = df.drop_duplicates(subset=["latitude","longitude"], keep="first")

    lat_map = {v: i for i, v in enumerate(GRID_LAT)}
    lon_map = {v: j for j, v in enumerate(GRID_LON)}

    layout_A = "Variable" in df.columns
    header = df.columns.tolist()

    months = [t0, t1]
    coords = {"time": [m.to_numpy() for m in months],
              "latitude": GRID_LAT,
              "longitude": GRID_LON}

    ds_vars = {}
    for var in variables:
        grids = []

        if layout_A:
            rows = df[df["Variable"].astype(str).str.strip() == var]
            if rows.empty:
                log.info("%s: no rows for variable %s", csv_path.name, var)
                continue
            li = rows["latitude"].map(lat_map).to_numpy(dtype=int, na_value=-1)
            lj = rows["longitude"].map(lon_map).to_numpy(dtype=int, na_value=-1)
            good_row = (li >= 0) & (lj >= 0)
            li, lj = li[good_row], lj[good_row]
        else:
            li = df["latitude"].map(lat_map).to_numpy(dtype=int, na_value=-1)
            lj = df["longitude"].map(lon_map).to_numpy(dtype=int, na_value=-1)
            good_row = (li >= 0) & (lj >= 0)
            li, lj = li[good_row], lj[good_row]

        for stamp in months:
            yr, mm = stamp.year, stamp.month
            grid = np.zeros((EXPECTED_LAT, EXPECTED_LON), dtype=DTYPE)

            if layout_A:
                col = next((c for c in header if c.endswith(f"_{yr}")), None)
                if col is None:
                    col = str(yr) if str(yr) in header else None
                if col is None:
                    col = next((c for c in header if c.endswith(f"_{mm:02d}/{yr}")), None)
                if col:
                    vals = rows.loc[good_row, col].astype(DTYPE).to_numpy()
                    grid[li, lj] = vals
                    log.info("%s: %s — layout A — %d col=%s  nz=%d",
                              csv_path.name, var, yr, col, int((vals!=0).sum()))
                else:
                    log.info("%s: %s — layout A — %d column not found",
                              csv_path.name, var, yr)
            else:
                tag_yr = f"{var}_{yr}"
                tag_mo = f"{var}_{mm:02d}/{yr}"
                col = tag_mo if tag_mo in df.columns else tag_yr if tag_yr in df.columns else None
                if col:
                    vals = df.loc[good_row, col].astype(DTYPE).to_numpy()
                    grid[li, lj] = vals
                    log.info("%s: %s — layout B — %d col=%s  nz=%d",
                              csv_path.name, var, yr, col, int((vals!=0).sum()))
                else:
                    log.info("%s: %s — layout B — %d column not found",
                              csv_path.name, var, yr)

            grids.append(grid)

        if len(grids) == 2:
            ds_vars[var] = xr.DataArray(
                np.stack(grids, axis=0),
                coords=coords,
                dims=("time","latitude","longitude"),
            )
        else:
            log.info("%s: variable %s has %d slices (expected 2)",
                        csv_path.name, var, len(grids))

    if not ds_vars:
        log.info("No variable data extracted from %s", csv_path.name)
        return None
    return xr.Dataset(ds_vars).chunk(CHUNKS)


def _load_era5(paths: list[Path], t0, t1) -> xr.Dataset:
    """
    Load ERA-5 tiles, keep exactly two calendar months (t0, t1),
    collapse any intra-month duplicates by taking the first slice
    crop to GRID_LAT/LON and return Dataset with time=2.
    """
    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)

    ds = xr.open_mfdataset(paths, engine=NC_ENGINE,
                           combine="by_coords", chunks="auto", parallel=True)
    ds = _wrap_lon(_std_time(ds))

    times = pd.DatetimeIndex(ds.indexes["time"])
    months = pd.PeriodIndex(times, freq="M")
    wanted = months.isin([t0.to_period("M"), t1.to_period("M")])
    ds= ds.isel(time=np.flatnonzero(wanted))

    # if >2 slices, keep first slice of each month
    if ds.sizes["time"] > 2:
        times = pd.DatetimeIndex(ds.indexes["time"])
        months = pd.PeriodIndex(times, freq="M")
        _, first_idx = np.unique(months, return_index=True)
        first_idx = np.sort(first_idx) # keep chronological order
        ds = ds.isel(time=first_idx)

    ds = ds.assign_coords(time=[t0.to_numpy(), t1.to_numpy()])
    ds = ds.sel(latitude=GRID_LAT, longitude=GRID_LON).chunk(CHUNKS)
    return ds


def _load_csv(csv_path: Path,
              variables: List[str],
              t0: pd.Timestamp,
              t1: pd.Timestamp) -> xr.Dataset | None:
    """
    Convert one CSV tile to an xr.Dataset with exactly the two calendar months
    [t0, t1].  Missing cells are zero-filled (not NaN).

    Returns
    -------
    xr.Dataset or None
    """
    df = pd.read_csv(csv_path).rename(
        columns={"Latitude":"latitude", "Longitude":"longitude"}
    )
    if "agriculture" in csv_path.name.lower() and "Variable" in open(csv_path).readline():
        return _load_agriculture_slice(csv_path, [t0, t1])
    if df["longitude"].max() > 180:
        df["longitude"] = ((df["longitude"] + 180) % 360) - 180
        log.info("%s: wrapped longitudes to −180…180", csv_path.name)

    df["latitude"]  = np.round(df["latitude"]  / 0.25) * 0.25
    df["longitude"] = np.round(df["longitude"] / 0.25) * 0.25

    df = df.drop_duplicates(subset=["latitude","longitude"], keep="first")

    lat_idx = pd.Categorical(df["latitude"], categories=GRID_LAT).codes
    lon_idx = pd.Categorical(df["longitude"], categories=GRID_LON).codes
    good = (lat_idx >= 0) & (lon_idx >= 0)

    months = [t0, t1]
    rasters: Dict[str, List[np.ndarray]] = {v: [] for v in variables}

    for targ in months:
        year = targ.year
        month_tag = f"{targ.month:02d}/{year}"

        for v in variables:
            col = (f"{v}_{month_tag}" if f"{v}_{month_tag}" in df.columns
                   else f"{v}_{year}" if f"{v}_{year}" in df.columns
                   else None)
            grid = np.zeros((EXPECTED_LAT, EXPECTED_LON), dtype=DTYPE)
            if col is not None:
                grid[lat_idx[good], lon_idx[good]] = df.loc[good, col].astype(DTYPE)
            rasters[v].append(grid)

    data_vars = {}
    for v, parts in rasters.items():
        if len(parts) != 2:
            log.warning("%s: %s missing one of the two months", csv_path.name, v)
            continue
        data_vars[v] = xr.DataArray(
            np.stack(parts, axis=0), # shape (time, lat, lon)
            coords={"time": [m.to_numpy() for m in months],
                    "latitude": GRID_LAT,
                    "longitude": GRID_LON},
            dims=("time", "latitude", "longitude"),
        )

    return xr.Dataset(data_vars).chunk(CHUNKS) if data_vars else None


def _load_species(parquet_path: Path,
                        t0: pd.Timestamp,
                        t1: pd.Timestamp) -> xr.Dataset:
    """Return xr.Dataset with every species, one raster per month (t0, t1)."""
    global _MASTER_SPECIES

    cols = ["Species", "Latitude", "Longitude", "Timestamp", "Distribution"]
    df = pd.read_parquet(parquet_path, columns=cols)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce").dt.tz_convert(None)

    if _MASTER_SPECIES is None:
        _MASTER_SPECIES = df["Species"].dropna().unique().tolist()

    df["month_period"] = df["Timestamp"].dt.to_period("M")

    periods = [pd.Timestamp(t0).to_period("M"),
               pd.Timestamp(t1).to_period("M")]

    lat_map = {v:i for i,v in enumerate(GRID_LAT)}
    lon_map = {v:j for j,v in enumerate(GRID_LON)}

    coords = {"time": [t0.to_numpy(), t1.to_numpy()],
              "latitude": GRID_LAT,
              "longitude": GRID_LON}

    data_vars = {}
    for sp in _MASTER_SPECIES:
        df_sp = df[df["Species"] == sp]
        grids = []
        for per in periods:
            sub = df_sp[df_sp["month_period"] == per]
            grid = np.zeros((EXPECTED_LAT, EXPECTED_LON), dtype=DTYPE)
            if not sub.empty:
                li = sub["Latitude"].map(lat_map).to_numpy(dtype=int, na_value=-1)
                lj = sub["Longitude"].map(lon_map).to_numpy(dtype=int, na_value=-1)
                ok = (li>=0)&(lj>=0)
                grid[li[ok], lj[ok]] = sub.loc[ok, "Distribution"].astype(DTYPE).to_numpy()
            grids.append(grid)
        data_vars[sp] = xr.DataArray(
            np.stack(grids, axis=0),
            coords=coords, dims=("time","latitude","longitude")
        )

    return xr.Dataset(data_vars).chunk(CHUNKS)


def _assemble_window(report: pd.DataFrame,
                     t0: pd.Timestamp,
                     t1: pd.Timestamp,
                     fill_nan: bool) -> dict:

    batch = {
        "batch_metadata": {
            "latitudes": GRID_LAT.tolist(),
            "longitudes": GRID_LON.tolist(),
            "timestamp": [str(t0), str(t1)],
            "pressure_levels": None,
            "species_list": None,
        },
        "surface_variables":      {},
        "single_variables":       {},
        "atmospheric_variables":  {},
        "edaphic_variables":      {},
        "climate_variables":      {},
        "misc_variables":         {},
        "forest_variables":       {},
        "land_variables":         {},
        "vegetation_variables":   {},
        "agriculture_variables":  {},
        "redlist_variables":      {},
        "species_variables":      {},
    }

    for _, row in report.query("modality=='copernicus'").iterrows():
        ds = _load_era5([Path(row.path)], t0, t1)
        slot = row.planned_slot
        for v in ds.data_vars:
            if VAR_CONFIG[slot] and v not in VAR_CONFIG[slot]:
                continue
            da = ds[v].astype(DTYPE).values # (2,H,W) or (2,C,H,W)
            ten = torch.from_numpy(da)
            batch[slot][v] = _maybe_fill_nan(ten, fill_nan)
            if "pressure_level" in ds[v].dims:
                batch["batch_metadata"]["pressure_levels"] = (ds[v]["pressure_level"].values.astype(float).tolist())

    for mod in ("forest", "agriculture", "land", "redlist", "vegetation"):
        for _, row in report.query("modality==@mod").iterrows():
            ds = _load_csv(Path(row.path), row.variables, t0, t1)
            if ds is None: continue
            slot = row.planned_slot
            for v in ds.data_vars:
                if VAR_CONFIG[slot] and v not in VAR_CONFIG[slot]:
                    continue
                ten = torch.from_numpy(ds[v].values.astype(DTYPE))
                batch[slot][v] = _maybe_fill_nan(ten, fill_nan)

    sp_df = report.query("modality=='species'")
    if not sp_df.empty:
        ds = _load_species(Path(sp_df.iloc[0].path), t0, t1)
        slot = sp_df.iloc[0].planned_slot
        for v in ds.data_vars:
            ten = torch.from_numpy(ds[v].values.astype(DTYPE))
            batch[slot][v] = _maybe_fill_nan(ten, fill_nan)
        batch["batch_metadata"]["species_list"] = list(ds.data_vars)

    return batch


def build_batches(report_path: Path,
                  out_dir: Path = BATCH_DIR,
                  start: str = _START_DATE,
                  end: str = _END_DATE,
                  max_batches: int | None = None,
                  fill_nan: bool = False) -> None:
    
    out_dir.mkdir(parents=True, exist_ok=True)
    report = pd.read_parquet(report_path)

    dates = pd.date_range(start, end, freq="MS")
    if max_batches: dates = dates[:max_batches]

    for i, t0 in enumerate(tqdm(dates, desc="Building Batches")):
        t1 = t0 + pd.DateOffset(months = 1)
        if t1 > pd.Timestamp(end): break
        batch = _assemble_window(report, t0, t1, fill_nan=fill_nan)
        fname = f"batch_{t0:%Y-%m-%d}_to_{t1:%Y-%m-%d}.pt"
        torch.save(batch, out_dir / fname)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="catalog_report.parquet", type=Path)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--start", default=_START_DATE)
    p.add_argument("--end", default=_END_DATE)
    p.add_argument("--fill_nan", default=False)
    args = p.parse_args()
    build_batches(args.report, max_batches=args.max_batches,
                  start=args.start, end=args.end, fill_nan=args.fill_nan)
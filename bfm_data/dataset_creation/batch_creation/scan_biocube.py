"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import re, json, logging, multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import xarray as xr
import pyarrow.parquet as pq
import torch


ROOT = Path("/data")
LAT_START, LAT_END = 32.0, 72.0
LON_START, LON_END = -25.0, 45.0
RESOLUTION = 0.25 # degrees
GRID_LAT = np.round(np.arange(LAT_START, LAT_END + 1e-6, RESOLUTION), 3)
GRID_LON = np.round(np.arange(LON_START, LON_END + 1e-6, RESOLUTION), 3)
EXPECTED_LAT, EXPECTED_LON = 161, 281 # 32->72 and –25->45 inclusive
_START_DATE, _END_DATE = "2000-01-01", "2020-06-01" # NDVI goes till 2020-06

NUM_WORKERS = 10
NC_ENGINE = "h5netcdf"
CSV_KWARGS = dict(low_memory=False, dtype_backend="pyarrow")

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("batch_catalog")


_SLOT_GROUPS = {
    "surface_variables"      : {"msl", "t2m", "u10", "v10"},
    "single_variables"       : {"stl", "z", "lsm"},
    "atmospheric_variables"  : {"t", "q", "u", "v", "z"},
    "edaphic_variables"      : {"stl1", "swvl2", "stl2", "swvl2"},
    "climate_variables"      : {"d2m", "tp", "avg_sdswrf", "avg_sdswrfcs", "avg_snlwrf", "avg_snswrf", "avg_tprate", "csfr", "sd", "smlt"},
}

def _infer_slot(vars_: list[str]) -> str:
    """
    Decide slot by **largest intersection** with canonical groups.
    Falls back to 'misc_variables' if no match.
    """
    vset = set(vars_)
    best_slot, best_score = "misc_variables", 0
    for slot, group in _SLOT_GROUPS.items():
        score = len(vset & group)
        if score > best_score:
            best_slot, best_score = slot, score
    return best_slot


def assert_shape(t: torch.Tensor | Tuple[int, ...], *shape: int) -> None:
    """Fail fast with an informative message if tensor shape is wrong."""
    if isinstance(t, torch.Tensor):
        ok = tuple(t.shape) == shape
    else:
        ok = tuple(t) == shape
    assert ok, f"Shape {tuple(t.shape) if isinstance(t, torch.Tensor) else t} ≠ {shape}"


@dataclass
class DataReport:
    path: str
    modality: str
    variables: List[str]
    dtype: str
    time_min: str | None
    time_max: str | None
    lon_min: float | None
    lon_max: float | None
    lat_min: float | None
    lat_max: float | None
    planned_slot: str
    pressure_levels: list[float] | None = None
    species_list: list[str] | None = None


    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def discover_files(root: Path = ROOT) -> Dict[str, List[Path]]:
    """Return mapping {modality -> [files]} deduced from the folder names."""
    patterns = {
        "forest": "**/Forest/Europe_forest_data.csv",
        "land": "**/Land/Europe_land_data.csv",
        "vegetation": "**/Land/Europe_ndvi_monthly_un_025.csv",
        "agriculture": "**/Agriculture/*.csv",
        "redlist": "**/RedList/Europe_red_list_index.csv",
        "species": "**/Species/europe_species.parquet",
        "copernicus": "**/Copernicus/ERA5-monthly/**/*.nc",
    }
    out: Dict[str, List[Path]] = {k: [] for k in patterns}
    for mod, pat in patterns.items():
        out[mod].extend(root.rglob(pat))
        log.info("Found %d %s files", len(out[mod]), mod)
    return out


def _scan_nc(path: Path) -> DataReport:
    """
    Scan a NetCDF for its variable list, spatial bounds, dtype, and 
    picks either 'time' or 'valid_time' as the temporal coordinate.
    """
    with xr.open_dataset(path, engine=NC_ENGINE, chunks={}, decode_times=True) as ds:
        vars_ = list(ds.data_vars)
        lon0, lon1 = float(ds.longitude.min()), float(ds.longitude.max())
        lat0, lat1 = float(ds.latitude.min()), float(ds.latitude.max())

        # choose the temporal coord with the most timestamps
        available = [c for c in ("time", "valid_time") if c in ds.coords]
        if not available:
            raise ValueError(f"{path!r} has no 'time' or 'valid_time' coords")
        lengths = {c: ds[c].size for c in available}
        t_coord = max(lengths, key=lengths.get)

        t0 = pd.to_datetime(ds[t_coord].min().values).date()
        t1 = pd.to_datetime(ds[t_coord].max().values).date()

        # extract any pressure_level channel
        if "pressure_level" in ds.coords:
            pl = ds["pressure_level"].values
            pressure_levels = [float(x) for x in pl.flatten()]
        else:
            pressure_levels = None

        return DataReport(
            path=str(path),
            modality="copernicus",
            variables=vars_,
            dtype=str(ds[vars_[0]].dtype),
            time_min=str(t0),
            time_max=str(t1),
            lon_min=lon0,
            lon_max=lon1,
            lat_min=lat0,
            lat_max=lat1,
            planned_slot=_infer_slot(vars_),
            pressure_levels=pressure_levels,

        )

def _scan_csv(path: Path, modality: str) -> DataReport:
    """
    Scan a CSV in one of two layouts and return its DataReport:
      - Layout B: a 'Variable' column + bare-year or MM/YYYY headers
      - Layout A: header-encoded var_date columns only

    Variables:
      - If 'Variable' exists → unique rows from that column.
      - Else → base-names from regex `(.+?)_(YYYY|MM/YYYY)`.

    Dates:
      - Any header matching YYYY or MM/YYYY or var_YYYY or var_MM/YYYY.
    """
    header = pd.read_csv(path, nrows=0, **CSV_KWARGS)
    cols   = header.columns.tolist()

    vars_set: set[str] = set()
    years:    list[int] = []

    hdr_re   = re.compile(r"^(?P<var>.+?)_(?:(?P<Y>\d{4})|(?P<m>\d{1,2})/(?P<Y2>\d{4}))$")
    year_re  = re.compile(r"^(?P<Y>\d{4})$")
    mY_re    = re.compile(r"^(?P<m>\d{1,2})/(?P<Y>\d{4})$")

    if "Variable" in cols:
        vcol = pd.read_csv(path, usecols=["Variable"], dtype_backend="pyarrow")["Variable"]
        vars_set.update(vcol.dropna().astype(str).str.strip().unique())
    else:
        for col in cols:
            m = hdr_re.match(col)
            if not m:
                continue
            vars_set.add(m.group("var"))

    if not vars_set:
        raise ValueError(f"No variables discovered in {path.name}")

    for col in cols:
        if m := hdr_re.match(col):
            # <var>_YYYY or <var>_MM/YYYY
            Y = m.group("Y") or m.group("Y2")
            years.append(int(Y))
        elif m := year_re.match(col):
            years.append(int(m.group("Y")))
        elif m := mY_re.match(col):
            years.append(int(m.group("Y")))

    if not years:
        raise ValueError(f"No date columns (YYYY or var_YYYY) in {path.name}")
    t0, t1 = f"{min(years)}-01-01", f"{max(years)}-01-01"

    data_cols = [c for c in cols if re.search(r"\d", c)]
    if data_cols:
        dtype_str = str(
            pd.read_csv(path, nrows=1, usecols=[data_cols[0]], **CSV_KWARGS)
              .dtypes[data_cols[0]]
        )
    else:
        dtype_str = "float64"

    samp = pd.read_csv(path, nrows=10_000, **CSV_KWARGS)
    lat0, lat1 = float(samp["Latitude"].min()), float(samp["Latitude"].max())
    lon0, lon1 = float(samp["Longitude"].min()), float(samp["Longitude"].max())

    return DataReport(
        path=str(path),
        modality=modality,
        variables=sorted(vars_set),
        dtype=dtype_str,
        time_min=t0,
        time_max=t1,
        lon_min=lon0,
        lon_max=lon1,
        lat_min=lat0,
        lat_max=lat1,
        planned_slot=f"{modality}_variables",
    )


def _scan_parquet(path: Path) -> DataReport:
    """
    Extracts metadata from a Parquet file, tolerating a few bad Timestamps:
      - Reads minimal columns into Pandas for stats (Latitude, Longitude, Timestamp[, Species]).
      - Parses Timestamp with errors='coerce', drops nulls.
      - Logs any dropped rows but does not error.
    """
    pf = pq.ParquetFile(path)
    try:
        schema = pf.schema_arrow
        data_cols = [
            n for n in schema.names
            if n not in ("Latitude", "Longitude", "year", "month", "Species", "Timestamp")
        ]
        dtype_str = str(schema.field(data_cols[0]).type)
    except AttributeError:
        schema = pf.schema
        data_cols = [
            schema.column(i).name
            for i in range(schema.num_columns)
            if schema.column(i).name not in ("Latitude", "Longitude", "year", "month", "Species", "Timestamp")
        ]
        idx = schema.get_field_index(data_cols[0])
        dtype_str = str(schema.column(idx).physical_type)

    cols = ["Latitude", "Longitude", "Timestamp"]
    if "Species" in pf.schema.names:
        cols.append("Species")
    df = pd.read_parquet(path, columns=cols, engine="pyarrow")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    n_total = len(df)
    df = df.dropna(subset=["Timestamp"])
    n_dropped = n_total - len(df)
    if n_dropped:
        log.warning("%r: dropped %d/%d rows with invalid Timestamps",
                    path.name, n_dropped, n_total)

    lat0, lat1 = float(df["Latitude"].min()), float(df["Latitude"].max())
    lon0, lon1 = float(df["Longitude"].min()), float(df["Longitude"].max())

    t0 = df["Timestamp"].min().isoformat()
    t1 = df["Timestamp"].max().isoformat()

    species_list = None
    if "Species" in df.columns:
        species_list = df["Species"].dropna().unique().tolist()

    return DataReport(
        path=str(path),
        modality="species",
        variables=data_cols,
        dtype=dtype_str,
        time_min=t0,
        time_max=t1,
        lon_min=lon0,
        lon_max=lon1,
        lat_min=lat0,
        lat_max=lat1,
        planned_slot="species_variables",
        pressure_levels=None,
        species_list=species_list,
    )


def build_report(root: Path = ROOT, workers: int = max(mp.cpu_count() - 1, 1)) -> pd.DataFrame:
    files = discover_files(root)
    tasks: List[Tuple[str, Path]] = []
    for mod, flist in files.items():
        tasks.extend((mod, p) for p in flist)

    with mp.Pool(workers) as pool:
        results: List[DataReport] = []
        for mod, path in tasks:
            if mod == "copernicus":
                results.append(pool.apply_async(_scan_nc, (path,)).get())
            elif mod in {"forest", "redlist", "land", "vegetation", "agriculture"}:
                results.append(pool.apply_async(_scan_csv, (path, mod)).get())
            elif mod == "species":
                results.append(pool.apply_async(_scan_parquet, (path,)).get())

    df = pd.DataFrame([asdict(r) for r in results])
    assert (df.lon_min.min() <= LON_START) and (df.lon_max.max() >= LON_END), "lon range mismatch"
    assert (df.lat_min.min() <= LAT_START) and (df.lat_max.max() >= LAT_END), "lat range mismatch"
    return df


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Scan data tree and output JSON report")
    ap.add_argument("--root", type=Path, default=ROOT, help="Root data directory")
    ap.add_argument("--out", type=Path, default=Path("catalog_report.parquet"))
    args = ap.parse_args()

    report_df = build_report(args.root)
    report_df.to_parquet(args.out, compression="zstd")
    log.info("Wrote %d records to %s", len(report_df), args.out)

"""
RUN WITH:
python scan_biocube.py --root biocube_data/data --out catalog_report.parquet
"""
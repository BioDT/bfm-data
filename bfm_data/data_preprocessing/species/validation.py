#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
from tqdm import tqdm

from plot_with_coordinates import read_zip_df, _norm, _eea_decode, TRANS_3035_4326

def load_folder(folder: Path) -> Dict[int, pd.DataFrame]:
    """Read each ZIP in folder into a normalized DataFrame, keyed by specieskey."""
    dfs: Dict[int, pd.DataFrame] = {}
    zips = sorted(folder.rglob("*.zip"))
    if not zips:
        sys.exit(f"[ERROR] No .zip files found in {folder}")
    for zp in tqdm(zips, desc="Loading zips"):
        df = read_zip_df(zp)
        df = _norm(df)
        key = int(df["specieskey"].iloc[0])
        dfs[key] = df
    return dfs

def preprocess_species(df: pd.DataFrame,
                       res_deg: float = 0.25, start_year: int = 2000) -> pd.DataFrame:
    """Map EEA->WGS, bin to res_deg grid, ensure full year/month grid with zeros.

    Any missing yearmonthday causes KeyError with context.
    """
    # derive date parts
    if 'year' not in df.columns or 'month' not in df.columns:
        if 'yearmonthday' not in df.columns:
            # include specieskey for context
            sk = df.get('specieskey', pd.Series([None])).iloc[0]
            raise KeyError(f"Species {sk}: missing 'yearmonthday' and no 'year'/'month' columns")
        dt = pd.to_datetime(df['yearmonthday'], errors='coerce')
        df['year'], df['month'], df['day'] = dt.dt.year, dt.dt.month, dt.dt.day

    # drop invalid codes
    df = df.dropna(subset=['eeacellcode']).copy()
    coords = df['eeacellcode'].map(_eea_decode, na_action='ignore')
    df = df[coords.notna()].copy()
    df['x'], df['y'] = zip(*coords.loc[df.index])

    # project to WGS84
    df['lon'], df['lat'] = TRANS_3035_4326.transform(df['x'].values, df['y'].values)

    # bin to grid
    df['occurrences'] = pd.to_numeric(df.get('occurrences',0), errors='coerce').fillna(0)
    df['lon_bin'] = (df['lon']//res_deg)*res_deg + res_deg/2
    df['lat_bin'] = (df['lat']//res_deg)*res_deg + res_deg/2

    grouped = (
        df.groupby(['specieskey','year','month','lon_bin','lat_bin'], as_index=False)
          ['occurrences'].sum()
          .rename(columns={'occurrences':'n','lon_bin':'lon','lat_bin':'lat'})
    )
    # determine fill span
    y_data_min = int(grouped['year'].min())
    y_data_max = int(grouped['year'].max())
    y0 = min(start_year, y_data_min)
    y1 = y_data_max
    years_full = list(range(y0, y1+1))
    months_full = list(range(1,13))
    lon_vals = grouped['lon'].unique()
    lat_vals = grouped['lat'].unique()

    # Cartesian product
    grid = pd.MultiIndex.from_product(
        [[grouped['specieskey'].iat[0]], years_full, months_full, lon_vals, lat_vals],
        names=['specieskey','year','month','lon','lat']
    ).to_frame(index=False)

    full = (
        grid.merge(grouped, how='left', on=['specieskey','year','month','lon','lat'])
            .fillna({'n': 0})
    )
    return full

def compute_stats(df: pd.DataFrame) -> Dict:
    """Return summary of 'n' species occurances and temporal coverage."""
    total = int(df["n"].sum())
    mean_ = float(df["n"].mean())
    mx = int(df["n"].max())
    years = df["year"].unique()
    span = (int(years.min()), int(years.max()))
    return {
        "total_occ": total,
        "mean_occ": mean_,
        "max_occ": mx,
        "years_span": span,
        "n_years": len(years),
        "n_months": df["month"].nunique()
    }

def save_species_data(dfs: Dict[int,pd.DataFrame], outdir: Path, fmt: str):
    """Write each species df to outdir/<specieskey>.<fmt>."""
    outdir.mkdir(parents=True, exist_ok=True)
    for key, df in dfs.items():
        fout = outdir/f"{key}.{fmt}"
        if fmt=="parquet":
            df.to_parquet(fout)
        else:
            df.to_csv(fout, index=False)

def validate_loading(outdir: Path) -> List[int]:
    """Check that each file in outdir is loadable. Return list of successful keys."""
    loaded: List[int] = []
    for f in sorted(outdir.iterdir()):
        try:
            if f.suffix==".parquet":
                pd.read_parquet(f)
            elif f.suffix==".csv":
                pd.read_csv(f)
            else:
                continue
            loaded.append(int(f.stem))
        except Exception as e:
            print(f"[WARN] Failed loading {f.name}: {e}")
    return loaded

def main():
    ap = argparse.ArgumentParser(
        description="Prepare GBIF EEA-cube zips into unified 0.25° WGS-84 grids"
    )
    ap.add_argument("--input-folder", "-i", type=Path, required=True,
                    help="Folder of <specieskey>.zip GBIF downloads")
    ap.add_argument("--output-folder", "-o", type=Path, required=True,
                    help="Where to write processed per-species files")
    ap.add_argument("--summary", "-s", type=Path, default=None,
                    help="CSV file path to write summary statistics")
    ap.add_argument("--res-deg", "-r", type=float, default=0.25,
                    help="Target grid resolution in degrees")
    ap.add_argument("--fmt", choices=["parquet","csv"], default="parquet")
    args = ap.parse_args()

    raw_dfs = load_folder(args.input_folder)

    proc_dfs: Dict[int,pd.DataFrame] = {}
    summary_rows = []
    for key, df in tqdm(raw_dfs.items(), desc="Preprocessing"):
        df2 = preprocess_species(df, res_deg=args.res_deg)
        proc_dfs[key] = df2

        stats = compute_stats(df2)
        stats.update({"specieskey": key})
        summary_rows.append(stats)

    save_species_data(proc_dfs, args.output_folder, args.fmt)
    print(f"[OK] Saved {len(proc_dfs)} species to {args.output_folder}/*.{args.fmt}")

    loaded = validate_loading(args.output_folder)
    print(f"[OK] Successfully loaded {len(loaded)}/{len(proc_dfs)} files")

    if args.summary:
        pd.DataFrame(summary_rows).to_csv(args.summary, index=False)
        print(f"[OK] Summary written to {args.summary}")

"""
python validation.py \
  --input-folder ./gbif_manual \
  --output-folder ./processed \
  --summary val_summary_stats.csv \
  --res-deg 0.25 \
  --fmt parquet
"""
if __name__ == "__main__":
    main()
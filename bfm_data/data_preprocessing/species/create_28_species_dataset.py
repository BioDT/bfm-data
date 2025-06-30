import argparse, sys, gc
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

from plot_with_coordinates import read_zip_df, _norm, _eea_decode, TRANS_3035_4326


def load_folder(folder: Path) -> Dict[int, pd.DataFrame]:
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
                       res_deg: float = 0.25) -> pd.DataFrame:
    if 'year' not in df.columns or 'month' not in df.columns:
        if 'yearmonthday' not in df.columns:
            sk = df.get('specieskey', pd.Series([None])).iloc[0]
            raise KeyError(f"Species {sk}: missing 'yearmonthday' and no 'year'/'month' columns")
        dt = pd.to_datetime(df['yearmonthday'], errors='coerce')
        df['year'], df['month'], df['day'] = dt.dt.year, dt.dt.month, dt.dt.day

    df = df.dropna(subset=['eeacellcode']).copy()
    coords = df['eeacellcode'].map(_eea_decode, na_action='ignore')
    df = df[coords.notna()].copy()
    df['x'], df['y'] = zip(*coords.loc[df.index])
    df['lon'], df['lat'] = TRANS_3035_4326.transform(df['x'].values, df['y'].values)

    df['occurrences'] = pd.to_numeric(df.get('occurrences', 0), errors='coerce').fillna(0)

    df['lon_bin'] = np.round(df['lon'] / res_deg) * res_deg
    df['lat_bin'] = np.round(df['lat'] / res_deg) * res_deg

    grouped = (
        df.groupby(['specieskey', 'year', 'month', 'lon_bin', 'lat_bin'], as_index=False)
          ['occurrences'].sum()
          .rename(columns={'occurrences': 'Distribution', 'lon_bin': 'Longitude', 'lat_bin': 'Latitude'})
    )

    return grouped, df


def main():
    ap = argparse.ArgumentParser(description="Process GBIF EEA zips into one combined Parquet file")
    ap.add_argument("--input-folder", "-i", type=Path, required=True,
                    help="Folder with GBIF specieskey ZIP files")
    ap.add_argument("--output-file", "-o", type=Path, required=True,
                    help="Single output Parquet file (combined result)")
    ap.add_argument("--summary", "-s", type=Path, default=None,
                    help="Optional summary CSV with statistics")
    ap.add_argument("--res-deg", "-r", type=float, default=0.25,
                    help="Target resolution in degrees")
    args = ap.parse_args()

    raw_dfs = load_folder(args.input_folder)

    all_species_dfs = []
    summary_rows = []

    for key, df_raw in tqdm(raw_dfs.items(), desc="Processing"):
        try:
            df_grouped, df_raw = preprocess_species(df_raw, res_deg=args.res_deg)

            df_out = pd.DataFrame({
                "Species": df_grouped["specieskey"],
                "Latitude": df_grouped["Latitude"],
                "Longitude": df_grouped["Longitude"],
                "Timestamp": pd.to_datetime(dict(
                    year=df_grouped["year"],
                    month=df_grouped["month"],
                    day=1)),
                "Distribution": df_grouped["Distribution"],
                "Phylum": df_raw["phylum"].iloc[0] if 'phylum' in df_raw else None,
                "Class": df_raw["class"].iloc[0] if 'class' in df_raw else None,
                "Order": df_raw["order"].iloc[0] if 'order' in df_raw else None,
                "Family": df_raw["family"].iloc[0] if 'family' in df_raw else None,
                "Genus": df_raw["genus"].iloc[0] if 'genus' in df_raw else None,
            })

            all_species_dfs.append(df_out)

            if args.summary:
                summary_rows.append({
                    "specieskey": key,
                    "total_dist": df_grouped["Distribution"].sum(),
                    "n_months": df_grouped.shape[0]
                })

            del df_out, df_grouped, df_raw
            gc.collect()

        except Exception as e:
            print(f"Failed to process species {key}: {e}")

    if all_species_dfs:
        df_combined = pd.concat(all_species_dfs, ignore_index=True)
        df_combined.to_parquet(args.output_file, engine="pyarrow", index=False, compression='snappy')
        print(f"Combined Parquet written to: {args.output_file}")
    else:
        print("No data processed successfully. No Parquet written.")

    if args.summary and summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.summary, index=False)
        print(f"Summary written to: {args.summary}")


if __name__ == "__main__":
    main()

from __future__ import annotations
import re
import io, zipfile, functools
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyproj import Transformer

mpl.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
})

sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

TRANS_3035_4326 = Transformer.from_crs(3035, 4326, always_xy=True)


def read_zip_df(zip_path: str | Path, sep: str = "\t") -> pd.DataFrame:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"), sep=sep)
    return df

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    lowmap = {c.lower(): c for c in df.columns}
    for tgt in ("specieskey", "year", "month", "day", "n", "occurrences",
                "yearmonthday", "eeacellcode"):
        l = tgt.lower()
        if l in lowmap and tgt not in df.columns:
            rename[lowmap[l]] = tgt
    return df.rename(columns=rename) if rename else df

@functools.lru_cache(maxsize=200_000)
def _eea_decode(code: str) -> Tuple[int, int]:
    """
    Decode centre Easting/Northing (m) of a 10 km EEA cell.

    Accepts:
      - Numeric 8-digit codes (e.g. "3950243")
      - Alphanumeric prefixes like "10kmE395N243" or "E395N243"
      - Glitched West/South variants like "10kmW136N317" or "10kmE415S016"

    Returns:
      (easting_m, northing_m) relative to the false origin in EPSG:3035.
    """
    s = str(code).strip()
    # strip leading "10km" if present
    if s.lower().startswith("10km"):
        s = s[4:]

    # regex for E/W easting and N/S northing
    m = re.match(r'^([EW])(\d+)([NS])(\d+)$', s, flags=re.IGNORECASE)
    if m:
        ew, e_str, ns, n_str = m.groups()
        e_val = int(e_str) * 10_000 + 5_000
        n_val = int(n_str) * 10_000 + 5_000
        # West of false origin → negative easting
        if ew.upper() == "W":
            e_val = -e_val
        # South of false origin → negative northing
        if ns.upper() == "S":
            n_val = -n_val
        return e_val, n_val

    # plain numeric code: first 4 digits = E/10km, next 4 = N/10km
    if s.isdigit() and len(s) == 8:
        return int(s[:4]) * 10_000 + 5_000, int(s[4:]) * 10_000 + 5_000

    raise ValueError(f"Bad EEA cell code: {code}")


def plot_eea_monthly(df: pd.DataFrame, *, species_key: int,
                     year: int, month: int, figsize=(8, 6)) -> plt.Figure:
    """Scatter 10km cell centres coloured by occurrence counts (LAEA metres)."""
    df = _norm(df)

    # extract year/month if not present
    if "year" not in df.columns or "month" not in df.columns:
        dt = pd.to_datetime(df["yearmonthday"], errors="coerce")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month

    df["occurrences"] = pd.to_numeric(df["occurrences"], errors="coerce").fillna(0)

    sub = df[(df["specieskey"] == species_key) &
             (df["year"] == year) & (df["month"] == month)]
    if sub.empty:
        raise ValueError("No data for requested slice")

    sub = sub.dropna(subset=["eeacellcode"]).copy()
    sub["x"], sub["y"] = zip(*sub["eeacellcode"].map(_eea_decode))
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=sub, x="x", y="y", size="occurrences", hue="occurrences",
                    sizes=(10, 300), palette="magma", alpha=0.7, ax=ax, legend=False)
    ax.set_aspect("equal", "box")
    ax.set_title(f"Species {species_key} – {year}-{month:02d} (EEA 10 km grid)")
    ax.set_xlabel("Easting (m, EPSG:3035)")
    ax.set_ylabel("Northing (m, EPSG:3035)")
    return fig

def plot_time_series(df: pd.DataFrame, *, species_key: int,
                     freq: str = "M", figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
    """Line plot of occurrences through time for a single species.

    Parameters
    ----------
    df : DataFrame returned by `read_zip_df` (cube file)
    species_key : GBIF specieskey to plot
    freq : pandas offset alias 'M' for monthly, 'Y' for yearly, etc.
    figsize : size of the matplotlib Figure
    """
    df = _norm(df)
    if "yearmonthday" not in df.columns:
        raise ValueError("DataFrame lacks 'yearmonthday' column - not a raw cube?")

    ts = df[df["specieskey"] == species_key].copy()
    if ts.empty:
        raise ValueError("specieskey not found in DataFrame")

    ts["occurrences"] = pd.to_numeric(ts["occurrences"], errors="coerce").fillna(0)
    ts["date"] = pd.to_datetime(ts["yearmonthday"], errors="coerce")
    ts = (ts.set_index("date")["occurrences"]
              .resample(freq).sum()
              .rename("n"))

    fig, ax = plt.subplots(figsize=figsize)
    ts.plot(ax=ax, marker="o", linewidth=2)
    ax.set(title=f"Time series of occurrences -specieskey {species_key}",
           xlabel="Date", ylabel="Occurrences")
    ax.grid(True, alpha=0.3)
    return fig

## V1 Working but nan eecellcode creates trouble
# def eea_to_wgs_points(df: pd.DataFrame) -> pd.DataFrame:
#     """Map each EEA 10 km cell to its WGS‑84 centre without binning or cropping."""
#     df = _norm(df).copy()
#     if "year" not in df.columns or "date" not in df.columns:
#         dt = pd.to_datetime(df["yearmonthday"], errors="coerce")
#         df["year"] = dt.dt.year
#         df["month"] = dt.dt.month
#         df["day"] = dt.dt.day

#     df = df.dropna(subset=["eeacellcode"]).copy()
#     df["x"], df["y"] = zip(*df["eeacellcode"].map(_eea_decode))
#     df["lon"], df["lat"] = TRANS_3035_4326.transform(df["x"].values, df["y"].values)
#     df["occurrences"] = pd.to_numeric(df["occurrences"], errors="coerce").fillna(0)
#     return df[["specieskey", "year", "month", "day", "lon", "lat", "occurrences"]]

def eea_to_wgs_points(df: pd.DataFrame) -> pd.DataFrame:
    df = _norm(df).copy()

    # derive year/month/day from yearmonthday if needed
    if "year" not in df.columns or "day" not in df.columns:
        dt = pd.to_datetime(df["yearmonthday"], errors="coerce")
        df["year"], df["month"], df["day"] = dt.dt.year, dt.dt.month, dt.dt.day

    # ensure occurrences exists
    df["occurrences"] = pd.to_numeric(df.get("occurrences", 0), errors="coerce").fillna(0)

    # drop truly empty codes
    mask_empty = df["eeacellcode"].isna() | df["eeacellcode"].astype(str).str.strip().eq("")
    if mask_empty.any():
        print(f"⚠ Dropping {mask_empty.sum()} rows with empty eeacellcode")
    df = df[~mask_empty].copy()

    # map only non-nulls; ignore NaNs automatically
    coords = df["eeacellcode"].map(_eea_decode, na_action="ignore")
    # coords is a Series of tuples or NaN; drop rows where mapping didn't happen
    valid = coords.notna()
    if (~valid).any():
        print(f"⚠ Dropping {(~valid).sum()} rows where decode failed or was NaN")
    df = df[valid].copy()

    # unpack
    df["x"], df["y"] = zip(*coords[valid])
    df["lon"], df["lat"] = TRANS_3035_4326.transform(df["x"].values, df["y"].values)

    return df[["specieskey", "year", "month", "day", "lon", "lat", "occurrences"]]


def eea_to_wgs_grid(df: pd.DataFrame, *, res_deg: float = 0.1,
                    bounds: Tuple[float, float, float, float] = (-30, 50, 34, 72),
                    fill_missing: bool = True) -> pd.DataFrame:
    """Convert EEA cube to WGS-84 grid *and* optionally back-fill empty bins.

    Parameters
    ----------
    df : raw cube DataFrame (after `read_zip_df`)
    res_deg : output WGS-84 grid step in degrees (default 0.1°)
    bounds : (lon_min, lon_max, lat_min, lat_max)
    fill_missing : if True, emit rows with `n=0` for grid bins that have **any**
        record for *any* month/year but are absent for the current slice.  This
        yields a rectangular grid - crucial for heat-maps or rasters.
    """
    df = _norm(df)
    # print(df.columns)
    if not {"eeacellcode", "yearmonthday", "occurrences"}.issubset(df.columns):
        raise ValueError("EEA cube lacks mandatory columns")

    # ── decode EEA codes, drop NaNs ─────────────────────────────────────────
    df = df.dropna(subset=["eeacellcode"], how="any").copy()
    df["x"], df["y"] = zip(*df["eeacellcode"].map(_eea_decode))

    # ── project ─────────────────────────────────────────────────────────────
    df["lon"], df["lat"] = TRANS_3035_4326.transform(df["x"].values, df["y"].values)

    # ── clip to study window ────────────────────────────────────────────────
    lon_min, lon_max, lat_min, lat_max = bounds
    df = df[(df["lon"].between(lon_min, lon_max)) & (df["lat"].between(lat_min, lat_max))]

    # ── bin ─────────────────────────────────────────────────────────────────
    df["occurrences"] = pd.to_numeric(df["occurrences"], errors="coerce").fillna(0)
    df["lon_bin"] = (df["lon"] // res_deg) * res_deg + res_deg / 2
    df["lat_bin"] = (df["lat"] // res_deg) * res_deg + res_deg / 2

    grouped = (df.groupby(["specieskey", "year", "month", "lon_bin", "lat_bin"], as_index=False)
                 ["occurrences"].sum()
                 .rename(columns={"occurrences": "n", "lon_bin": "lon", "lat_bin": "lat"}))

    if not fill_missing:
        return grouped

    # ── optional grid completion ───────────────────────────────────────────
    full_lon = pd.Series(np.arange(lon_min + res_deg/2, lon_max, res_deg), name="lon")
    full_lat = pd.Series(np.arange(lat_min + res_deg/2, lat_max, res_deg), name="lat")
    grid = (full_lon.to_frame().assign(key=1)
              .merge(full_lat.to_frame().assign(key=1), on="key")
              .drop(columns="key"))

    # Build full cartesian product per species / year / month
    meta_cols = ["specieskey", "year", "month"]
    meta = grouped[meta_cols].drop_duplicates()
    meta["key"] = 1
    grid["key"] = 1
    full = (meta.merge(grid, on="key")
                 .drop("key", axis=1)
                 .merge(grouped, how="left",
                        on=meta_cols + ["lon", "lat"]).fillna({"n": 0}))
    return full

# V1 issues with nans
def plot_grid_comparison(df_eea: pd.DataFrame, df_wgs: pd.DataFrame,
                         species_key: int, year: int, month: int,
                         figsize=(10, 5)) -> plt.Figure:
    """Sidebyside scatter: EEA cell centres vs one-to-one WGS84 points.

    Handles both gridded (`n`) and point (`occurrences`) WGS DataFrames.
    """
    df_eea = _norm(df_eea)
    df_wgs = _norm(df_wgs)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── EEA side ───────────────────────────────────────────────────────────
    # sub_e = df_eea[(df_eea["specieskey"] == species_key) &
    #                (df_eea["year"] == year) & (df_eea["month"] == month)].copy()
    # sub_e["x"], sub_e["y"] = zip(*sub_e["eeacellcode"].map(_eea_decode))
    # sns.scatterplot(ax=axes[0], data=sub_e, x="x", y="y",
    #                 size="occurrences", hue="occurrences",
    #                 sizes=(10, 300), palette="magma", alpha=0.7, legend=False)
    # axes[0].set_title("EEA 10 km grid")
    # axes[0].set_aspect("equal", "box")
        # ── EEA side ───────────────────────────────────────────────────────────
    sub_e = df_eea[
        (df_eea["specieskey"] == species_key) &
        (df_eea["year"]       == year)        &
        (df_eea["month"]      == month)
    ].copy()

    # Drop blank codes
    mask_empty = sub_e["eeacellcode"].isna() | sub_e["eeacellcode"].astype(str).str.strip().eq("")
    if mask_empty.any():
        print(f"⚠ Dropping {mask_empty.sum()} rows with missing eeacellcode (EEA side)")
    sub_e = sub_e[~mask_empty]

    # Compute or reuse coordinates
    if "x" not in sub_e.columns or "y" not in sub_e.columns:
        # decode only the valid codes
        coords = sub_e["eeacellcode"].map(_eea_decode, na_action="ignore")
        valid = coords.notna()
        if (~valid).any():
            print(f"Dropping {(~valid).sum()} rows with invalid codes")
        sub_e = sub_e[valid].copy()
        sub_e["x"], sub_e["y"] = zip(*coords[valid])

    # Plot
    sns.scatterplot(
        ax=axes[0], data=sub_e,
        x="x", y="y",
        size="occurrences", hue="occurrences",
        sizes=(10, 300), palette="magma",
        alpha=0.7, legend=False
    )
    axes[0].set_title("EEA 10 km grid")
    axes[0].set_aspect("equal", "box")


    # ── WGS side ───────────────────────────────────────────────────────────
    col_size = "n" if "n" in df_wgs.columns else "occurrences"
    sub_w = df_wgs[(df_wgs["specieskey"] == species_key) &
                   (df_wgs["year"] == year) & (df_wgs["month"] == month)].copy()
    sns.scatterplot(ax=axes[1], data=sub_w, x="lon", y="lat",
                    size=col_size, hue=col_size,
                    sizes=(10, 300), palette="magma", alpha=0.7, legend=False)
    axes[1].set_title("WGS84 points/grid")
    axes[1].set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    return fig


def plot_grid_comparison_1(df_eea: pd.DataFrame, df_wgs: pd.DataFrame,
                         species_key: int, year: int, month: int,
                         figsize=(10, 5)) -> plt.Figure:
    """Side-by-side scatter of EEA cell centres vs WGS-grid bins for a slice."""
    # Normalize column names
    df_eea = _norm(df_eea)
    df_wgs = _norm(df_wgs)

    # Filter the slice
    sub_e = df_eea.loc[
        (df_eea["specieskey"] == species_key) &
        (df_eea["year"]       == year)        &
        (df_eea["month"]      == month)
    ].copy()

    # Drop missing or empty codes
    mask_empty = sub_e["eeacellcode"].isna() | sub_e["eeacellcode"].astype(str).str.strip().eq("")
    if mask_empty.any():
        print(f"Dropping {mask_empty.sum()} EEA rows with empty eeacellcode")
    sub_e = sub_e[~mask_empty]

    # Safe decode, ignore NaNs
    coords = sub_e["eeacellcode"].map(_eea_decode, na_action="ignore")
    valid = coords.notna()
    if (~valid).any():
        print(f"Dropping {(~valid).sum()} EEA rows with invalid codes")
    sub_e = sub_e[valid].copy()
    sub_e["x"], sub_e["y"] = zip(*coords[valid])

    # Plot EEA side
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    sns.scatterplot(ax=axes[0], data=sub_e,
                    x="x", y="y",
                    size="occurrences", hue="occurrences",
                    sizes=(10, 300), palette="magma",
                    alpha=0.7, legend=False)
    axes[0].set_title("EEA 10 km grid")
    axes[0].set_aspect("equal", "box")

    # Prepare WGS side
    sub_w = df_wgs.loc[
        (df_wgs["specieskey"] == species_key) &
        (df_wgs["year"]       == year)        &
        (df_wgs["month"]      == month)
    ].copy()
    # Drop rows without lon/lat
    sub_w = sub_w.dropna(subset=["lon", "lat"])

    # Plot WGS side
    sns.scatterplot(ax=axes[1], data=sub_w,
                    x="lon", y="lat",
                    size=sub_w.get("n", sub_w.get("occurrences", pd.Series(1))),
                    hue=sub_w.get("n", sub_w.get("occurrences", pd.Series(1))),
                    palette="magma", legend=False, alpha=0.7)
    axes[1].set_title("WGS-84 grid")
    axes[1].set_aspect("equal", "datalim")

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EEA cube quick-look plotter")
    parser.add_argument("csv", help="TSV path (unzipped cube)")
    parser.add_argument("--species", type=int, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--out", help="PNG output path")
    parser.add_argument("--mode", choices=["eea", "monthly"], default="eea")
    args = parser.parse_args()

    df_raw = read_zip_df(args.csv)
    if args.mode == "eea":
        fig = plot_eea_monthly(df_raw, species_key=args.species,
                               year=args.year, month=args.month)
    else:
        from datetime import datetime
        df_raw = _norm(df_raw)
        df_raw["n"] = pd.to_numeric(df_raw["occurrences"], errors="coerce").fillna(0)
        fig = sns.lineplot()
        # Placeholder for monthly trend quick plot
        fig = plt.gcf()

    if args.out:
        fig.savefig(args.out, dpi=600, bbox_inches="tight")
    else:
        plt.show()

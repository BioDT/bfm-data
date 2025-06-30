import argparse
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data-folder", type=str, required=True)
cli_args, _ = parser.parse_known_args()
DATA_DIR = Path(cli_args.data_folder)

# Cached loader
@st.cache_data(show_spinner="Loading parquet…", ttl=600)
def load_species(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

# Sidebar controls 
st.sidebar.header("Controls")
if not DATA_DIR.exists():
    st.sidebar.error(f"Folder {DATA_DIR} does not exist")
    st.stop()
files = sorted(DATA_DIR.glob("*.parquet"))
if not files:
    st.sidebar.error("No .parquet files found")
    st.stop()

species_ids = [f.stem for f in files]
sel_id = st.sidebar.selectbox("Species key", species_ids)

# Session state for year/month navigation
if "year" not in st.session_state or st.session_state.species != sel_id:
    df_tmp = load_species(DATA_DIR/f"{sel_id}.parquet")
    st.session_state.species = sel_id
    st.session_state.year   = int(df_tmp.year.min())
    st.session_state.month  = 1

df = load_species(DATA_DIR/f"{sel_id}.parquet")
min_year, max_year = int(df.year.min()), int(df.year.max())

# year/month selectors with buttons
col1,col2 = st.sidebar.columns(2)
if col1.button("<- Year"):
    st.session_state.year = max(min_year, st.session_state.year-1)
if col2.button("Year ->"):
    st.session_state.year = min(max_year, st.session_state.year+1)

col3,col4 = st.sidebar.columns(2)
if col3.button("<- Mon"):
    st.session_state.month = 12 if st.session_state.month==1 else st.session_state.month-1
if col4.button("Mon ->"):
    st.session_state.month = 1  if st.session_state.month==12 else st.session_state.month+1

sel_year  = st.sidebar.slider("Year", min_year, max_year, st.session_state.year, key="yr_slider")
st.session_state.year = sel_year
sel_month = st.sidebar.slider("Month", 1, 12, st.session_state.month, key="mo_slider")
st.session_state.month = sel_month

# Data slice & stats
val_col = "n" if "n" in df.columns else "occurrences"
sub = df[(df.year==st.session_state.year)&(df.month==st.session_state.month)]
occ_sum = int(sub[val_col].sum())

st.title(f"Species {sel_id} - {st.session_state.year}-{st.session_state.month:02d}")
st.subheader(f"Occurrences in slice: {occ_sum}")

# Cartopy map 
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-30, 50, 34, 72])
ax.coastlines("10m", linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.gridlines(draw_labels=True, xlocs=range(-30,51,10), ylocs=range(35,73,5),
             linewidth=0.25, linestyle=":", color="gray")

if not sub.empty:
    sizes = sub[val_col]/sub[val_col].max()*300
    sc = ax.scatter(sub.lon, sub.lat, s=sizes, c=sub[val_col], cmap="magma",
                    alpha=0.8, transform=ccrs.PlateCarree())
    cb = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
    cb.set_label("Occurrences")
else:
    ax.text(0.5,0.5,"No data", transform=ax.transAxes, ha="center", va="center")

st.pyplot(fig, use_container_width=True)

# Expandable: raw table
with st.expander("Show raw data"):
    st.dataframe(sub)

# Timeseries plot 
st.subheader("Monthly trend (all years)")
trend = (df.groupby(["year","month"], as_index=False)[val_col].sum()
           .assign(date=lambda d: pd.to_datetime(d.year.astype(str)+"-"+d.month.astype(str), format="%Y-%m")))
trend = trend.sort_values("date")
# smooth with 3‑month rolling window to reduce spikes
trend["smooth"] = trend[val_col].rolling(window=3, center=True, min_periods=1).mean()
plt2 = sns.relplot(data=trend, x="date", y="smooth", kind="line", height=3, aspect=4/3)
plt2.set_ylabels("Occurrences (3-month smoothed)")
st.pyplot(plt2.fig, use_container_width=True)


#### RUN: streamlit run viewer.py -- --data-folder ./processed
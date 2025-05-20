"""
Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license.

Streamlit viewer for per-modality & variable two-month batches.

run:
    streamlit run batch_viewer.py --data_dir ./batches
"""
import argparse
from pathlib import Path
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point


def get_cli_dir() -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_dir", default="batches", type=Path)
    ns, _ = p.parse_known_args()
    return ns.data_dir
DATA_DIR = get_cli_dir()

st.sidebar.title("Batch viewer")
all_batches = sorted(DATA_DIR.glob("batch_*.pt"))
if not all_batches:
    st.sidebar.error(f"No batch_*.pt files in {DATA_DIR}")
    st.stop()

batch_name = st.sidebar.selectbox(
    "Choose batch", [p.name for p in all_batches])
batch = torch.load(DATA_DIR / batch_name, map_location="cpu")

slot_options = [k for k in batch.keys() if k.endswith("_variables")]
slot = st.sidebar.selectbox("Variable slot", slot_options)

var_list = list(batch.get(slot, {}).keys())
if not var_list:
    st.warning(f"No variables stored for slot '{slot}'.")
    st.stop()

var_choice = st.sidebar.multiselect(
    "Variable(s)", var_list, default=[var_list[0]])

if not var_choice:
    st.info("Pick at least one variable")
    st.stop()

pl_sel = None
example_tensor = next(iter(batch[slot].values()))
if example_tensor.ndim == 4:# (2,C,H,W)
    pl_list = batch.get("batch_metadata", {}).get("pressure_levels")
    if pl_list is None:
        pl_list = list(range(example_tensor.shape[1]))
    pl_sel = st.sidebar.multiselect("Pressure levels", pl_list, default=[pl_list[0]])
    if not pl_sel:
        st.stop()

def _plot_maps(day0, day1, lats, lons, title):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), subplot_kw=dict(projection=proj))
    for a, data, lbl in zip(ax, (day0, day1), ("Month 1", "Month 2")):
        data = np.asarray(data)
        cyc, lon_cyc = add_cyclic_point(data, coord=lons)
        mesh = a.pcolormesh(lon_cyc, lats, cyc, transform=proj, cmap="viridis")
        a.add_feature(cfeature.COASTLINE, linewidth=0.4)
        a.set_title(f"{title}\n{lbl}")
    fig.colorbar(mesh, ax=ax.ravel().tolist(), shrink=0.75)
    st.pyplot(fig)

meta = batch["batch_metadata"]
lats = np.array(meta["latitudes"])
lons = np.array(meta["longitudes"])
ts= meta["timestamp"]

st.header(batch_name)
st.caption(f"{ts[0]}  ->  {ts[1]}")

for v in var_choice:
    tensor = batch[slot][v]
    if tensor.ndim == 3:
        _plot_maps(tensor[0], tensor[1], lats, lons, v)
    else:
        for pl in pl_sel:
            cidx = batch["batch_metadata"]["pressure_levels"].index(pl)
            _plot_maps(tensor[0, cidx], tensor[1, cidx], lats, lons,
                        f"{v}  {pl} hPa")
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
    st.sidebar.error("No batch_*.pt files found.")
    st.stop()

batch_name = st.sidebar.selectbox("Select batch", [p.name for p in all_batches])
batch = torch.load(DATA_DIR / batch_name, map_location="cpu", weights_only=False)

available_slots = [k for k in batch.keys() if k.endswith("_variables")]
slot = st.sidebar.selectbox("Data slot", available_slots)

var_names = batch["variable_names"].get(slot, [])
if not var_names:
    st.warning(f"No variable list stored for slot '{slot}'.")
    st.stop()
var_choice = st.sidebar.multiselect("Variable(s)", var_names, default=[var_names[0]])
if not var_choice:
    st.info("Pick at least one variable.")
    st.stop()

pl_sel = None
if slot == "atmospheric_variables":
    pl = batch["pressure_levels"]
    pl_sel = st.sidebar.multiselect("Pressure level(s) hPa", pl, default=[pl[0]])
    if not pl_sel:
        st.stop()

def plot_maps(day0, day1, lats, lons, title):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), subplot_kw=dict(projection=proj))
    for ax_i, data, lbl in zip(ax, (day0, day1), ("Month 1", "Month 2")):
        data = np.asarray(data)
        d_cyc, lon_cyc = add_cyclic_point(data, coord=lons)
        mesh = ax_i.pcolormesh(lon_cyc, lats, d_cyc, cmap="viridis", transform=proj)
        ax_i.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax_i.set_title(f"{title}\n{lbl}")
    fig.colorbar(mesh, ax=ax.ravel().tolist(), shrink=0.75)
    st.pyplot(fig)

lats = np.array(batch["batch_metadata"]["latitudes"])
lons = np.array(batch["batch_metadata"]["longitudes"])
ts = batch["batch_metadata"]["timestamp"]

st.header(batch_name)
st.caption(f"Times: {ts[0]} -> {ts[1]}")

tensor = batch[slot]

for v in var_choice:
    vidx = var_names.index(v)
    if slot == "atmospheric_variables":
        for pl in pl_sel:
            cidx = batch["pressure_levels"].index(pl)
            data = tensor[:, vidx, cidx].numpy()
            plot_maps(data[0], data[1], lats, lons, f"{v} {pl}hPa")
    else:
        data = tensor[:, vidx].numpy()
        plot_maps(data[0], data[1], lats, lons, v)

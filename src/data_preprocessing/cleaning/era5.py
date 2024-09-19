# src/data_preprocessing/cleaning/era5.py

import numpy as np
import torch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as RGI


def interpolate(
    v: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    lat_new: torch.Tensor,
    lon_new: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate a variable `v` with latitudes `lat` and longitudes `lon` to new latitudes
    `lat_new` and new longitudes `lon_new`.
    """
    return torch.from_numpy(
        interpolate_numpy(
            v.double().numpy(),
            lat.double().numpy(),
            lon.double().numpy(),
            lat_new.double().numpy(),
            lon_new.double().numpy(),
        )
    ).float()


def interpolate_numpy(
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_new: np.ndarray,
    lon_new: np.ndarray,
) -> np.ndarray:
    """Like :func:`.interpolate`, but for NumPy tensors."""

    assert (np.diff(lon) > 0).all()
    lon = np.concatenate((lon[-1:] - 360, lon, lon[:1] + 360))

    batch_shape = v.shape[:-2]
    v = v.reshape(-1, *v.shape[-2:])

    vs_regridded = []
    for vi in v:
        vi = np.concatenate((vi[:, -1:], vi, vi[:, :1]), axis=1)

        rgi = RGI(
            (lat, lon),
            vi,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        lat_new_grid, lon_new_grid = np.meshgrid(
            lat_new,
            lon_new,
            indexing="ij",
            sparse=True,
        )
        vs_regridded.append(rgi((lat_new_grid, lon_new_grid)))

    v_regridded = np.stack(vs_regridded, axis=0)
    v_regridded = v_regridded.reshape(*batch_shape, lat_new.shape[0], lon_new.shape[0])

    return v_regridded

# src/data_preprocessing/transformation/era5.py

from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import xarray

from src.config import paths


def normalise_surface_variable(
    tensor: torch.Tensor,
    variable_name: str,
    stats: Optional[dict[str, tuple[float, float]]] = None,
    unormalise: bool = False,
) -> torch.Tensor:
    """
    Apply normalization or unnormalization to a surface-level variable.

    Args:
        tensor (torch.Tensor): The input tensor representing the surface-level variables.
        variable_name (str): The name of the surface-level variable.
        unormalise (bool, optional): If True, applies unnormalization by reversing the
            normalization process. If False, applies normalization. Defaults to False.

    Returns:
        torch.Tensor: The normalized or unnormalized tensor, depending on the value of the `reverse` parameter.
    """

    if stats and variable_name in stats:
        location, scale = stats[variable_name]
    else:
        location = locations[variable_name]
        scale = scales[variable_name]

    if unormalise:
        return tensor * scale + location
    else:
        return (tensor - location) / scale


def normalise_atmospheric_variables(
    tensor: torch.Tensor,
    variable_name: str,
    pressure_levels: tuple[int | float, ...],
    unormalise: bool = False,
) -> torch.Tensor:
    """
    Apply normalization or unnormalization to a atmospheric-level variable.

    Args:
        tensor (torch.Tensor): The input tensor representing the atmospheric-level variables.
        variable_name (str): The name of the atmospheric-level variable.
        pressure_levels (tuple[int | float, ...]): A tuple of pressure levels (in hPa) corresponding
            to the levels at which the atmospheric variable is measured.
        unormalise (bool, optional): If True, applies unnormalization by reversing the
            normalization process. If False, applies normalization. Defaults to False.

    Returns:
        torch.Tensor: The normalized or unnormalized tensor, depending on the value of the `reverse` parameter.
    """
    level_locations: list[int | float] = []
    level_scales: list[int | float] = []

    for level in pressure_levels:
        level_locations.append(locations[f"{variable_name}_{level}"])
        level_scales.append(scales[f"{variable_name}_{level}"])
    location = torch.tensor(level_locations, dtype=tensor.dtype, device=tensor.device)
    scale = torch.tensor(level_scales, dtype=tensor.dtype, device=tensor.device)

    if unormalise:
        return tensor * scale[..., None, None] + location[..., None, None]
    else:
        return (tensor - location[..., None, None]) / scale[..., None, None]


unnormalise_surface_variables = partial(normalise_surface_variable, reverse=True)
unnormalise_atmospheric_variables = partial(
    normalise_atmospheric_variables, reverse=True
)


def get_mean_standard_deviation(
    surface_path: str, atmospheric_path: str
) -> Tuple[Dict[str, float], Dict[str, float]]:

    surface_dataset = xarray.open_dataset(
        Path(paths.ERA5_DIR) / surface_path, engine="netcdf4"
    )
    atmospheric_dataset = xarray.open_dataset(
        Path(paths.ERA5_DIR) / atmospheric_path, engine="netcdf4"
    )

    locations: dict[str, float] = {}
    scales: dict[str, float] = {}
    surface_variables = ["t2m", "u10", "v10", "msl"]
    atmospheric_variables = ["t", "u", "v", "q"]

    if "level" in atmospheric_dataset.coords:
        pressure_levels = atmospheric_dataset.level.values
    else:
        pressure_levels = []

    for surface_variable in surface_variables:
        if surface_variable in surface_dataset:
            data = surface_dataset[surface_variable].values
            locations[surface_variable] = np.mean(data)
            scales[surface_variable] = np.std(data)
        else:
            print(f"Variable {surface_variable} not found in the dataset.")

    for atmospheric_variable in atmospheric_variables:
        if atmospheric_variable in atmospheric_dataset:
            for level in pressure_levels:
                data = atmospheric_dataset[atmospheric_variable].sel(level=level).values
                locations[f"{atmospheric_variable}_{level}"] = np.mean(data)
                scales[f"{atmospheric_variable}_{level}"] = np.std(data)
        else:
            print(f"Variable {atmospheric_variable} not found in the dataset.")

    return locations, scales


locations, scales = get_mean_standard_deviation(
    "ERA5-Reanalysis-surface-2001-01-01-2001-12-31.nc",
    "ERA5-Reanalysis-pressure-2001-01-01-2001-02-31.nc",
)

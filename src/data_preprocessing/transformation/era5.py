# src/data_preprocessing/transformation/era5.py

from functools import partial
from typing import Dict, Tuple

import numpy as np
import torch
import xarray as xr


def normalise_surface_variable(
    tensor: torch.Tensor,
    variable_name: str,
    locations: Dict[str, float],
    scales: Dict[str, float],
    unormalise: bool = False,
) -> torch.Tensor:
    """
    Apply normalization or unnormalization to a surface-level variable.

    Args:
        tensor (torch.Tensor): The input tensor representing the surface-level variables.
        variable_name (str): The name of the surface-level variable.
        locations (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the mean for that variable.
        scales (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the standard deviation for that variable.
        unormalise (bool, optional): If True, applies unnormalization by reversing the
                                    normalization process. If False, applies normalization. Defaults to False.

    Returns:
        torch.Tensor: The normalized or unnormalized tensor, depending on the value of the `reverse` parameter.
    """
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
    locations: Dict[str, float],
    scales: Dict[str, float],
    unormalise: bool = False,
) -> torch.Tensor:
    """
    Apply normalization or unnormalization to a atmospheric-level variable.

    Args:
        tensor (torch.Tensor): The input tensor representing the atmospheric-level variables.
        variable_name (str): The name of the atmospheric-level variable.
        pressure_levels (tuple[int | float, ...]): A tuple of pressure levels (in hPa) corresponding
                                    to the levels at which the atmospheric variable is measured.
        locations (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the mean for that variable.
        scales (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the standard deviation for that variable.
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
    surface_dataset: xr.Dataset,
    single_dataset: xr.Dataset,
    atmospheric_dataset: xr.Dataset,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute the mean (location) and standard deviation (scale) for surface, single-level,
    and atmospheric variables from the given datasets.

    Args:
        surface_dataset (xarray.Dataset): Dataset containing surface-level variables such as
                                          2-meter temperature (t2m), mean sea-level pressure (msl), etc.
        single_dataset (xarray.Dataset): Dataset containing single-level variables such as
                                         geopotential (z), land-sea mask (lsm), soil type (slt), etc.
        atmospheric_dataset (xarray.Dataset): Dataset containing atmospheric variables measured at
                                              different pressure levels such as temperature (t),
                                              u-component of wind (u), v-component of wind (v), etc.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]:
            - A dictionary (locations) where the key is the variable name or variable_name_pressure_level
              and the value is the mean (location) for that variable.
            - A dictionary (scales) where the key is the variable name or variable_name_pressure_level
              and the value is the standard deviation (scale) for that variable.
    """

    locations: dict[str, float] = {}
    scales: dict[str, float] = {}
    surface_variables = ["t2m", "u10", "v10", "msl"]
    single_variables = ["z", "lsm", "slt"]
    atmospheric_variables = ["t", "u", "v", "q", "z"]

    if "pressure_level" in atmospheric_dataset.coords:
        pressure_levels = atmospheric_dataset.pressure_level.values
    else:
        pressure_levels = []

    for surface_variable in surface_variables:
        if surface_variable in surface_dataset:
            data = surface_dataset[surface_variable].values
            locations[surface_variable] = np.mean(data)
            scales[surface_variable] = np.std(data)
        else:
            print(f"Variable {surface_variable} not found in the dataset.")

    for single_variable in single_variables:
        if single_variable in single_dataset:
            data = single_dataset[single_variable].values
            locations[single_variable] = np.mean(data)
            scales[single_variable] = np.std(data)
        else:
            print(f"Variable {single_variable} not found in the dataset.")

    for atmospheric_variable in atmospheric_variables:
        if atmospheric_variable in atmospheric_dataset:
            for level in pressure_levels:
                data = (
                    atmospheric_dataset[atmospheric_variable]
                    .sel(pressure_level=level)
                    .values
                )
                locations[f"{atmospheric_variable}_{level}"] = np.mean(data)
                scales[f"{atmospheric_variable}_{level}"] = np.std(data)
        else:
            print(f"Variable {atmospheric_variable} not found in the dataset.")

    return locations, scales

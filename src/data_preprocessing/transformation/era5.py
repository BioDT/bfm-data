import dataclasses
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import xarray

from src.config import settings
from src.data_preprocessing.batch import DataBatch
from src.data_preprocessing.metadata import BatchMetadata


def _fmap(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "DataBatch":
    """
    Applies a function `f` to all tensor elements within the DataBatch.

    Args:
        f (Callable[[torch.Tensor], torch.Tensor]): A function that takes a torch.Tensor
        as input and returns a torch.Tensor, applied to each tensor in the DataBatch.

    Returns:
        DataBatch: A new DataBatch instance with the function `f` applied to its tensors.
    """
    surface_variables = {key: f(value) for key, value in self.surface_variables.items()}
    single_variables = {key: f(value) for key, value in self.single_variables.items()}
    atmospheric_variables = {
        key: f(value) for key, value in self.atmospheric_variables.items()
    }
    return DataBatch(
        surface_variables=surface_variables,
        single_variables=single_variables,
        atmospheric_variables=atmospheric_variables,
        batch_metadata=BatchMetadata(
            latitude=f(self.batch_metadata.latitude),
            longitude=f(self.batch_metadata.longitude),
            pressure_levels=self.batch_metadata.pressure_levels,
            time=self.batch_metadata.timestamp,
            prediction_step=self.batch_metadata.prediction_step,
        ),
    )


DataBatch._fmap = _fmap


def to(self, device: str | torch.device) -> "DataBatch":
    """
    Moves all tensors within the DataBatch to the specified device.

    Args:
        device (str | torch.device): The device to which the tensors should be moved.

    Returns:
        DataBatch: A new DataBatch instance with all tensors moved to the specified device.
    """
    return self._fmap(lambda x: x.to(device))


DataBatch.to = to


def type(self, t: type) -> "DataBatch":
    """
    Converts all tensors within the DataBatch to the specified data type.

    Args:
        t (type): The data type to which the tensors should be converted.

    Returns:
        DataBatch: A new DataBatch instance with all tensors converted to the specified type.
    """
    return self._fmap(lambda x: x.type(t))


DataBatch.type = type


def normalize_data(self) -> "DataBatch":
    """
    Normalise all variables in the batch.

    Returns:
        DataBatch: A new batch with all variables normalized.
    """
    normalized_surface_variables = {
        key: normalise_surface_variable(value, key)
        for key, value in self.surface_variables.items()
    }
    normalized_single_variables = {
        key: normalise_surface_variable(value, key)
        for key, value in self.single_variables.items()
    }
    normalized_atmospheric_variables = {
        key: normalise_atmospheric_variables(
            value, key, self.batch_metadata.pressure_levels
        )
        for key, value in self.atmospheric_variables.items()
    }

    return DataBatch(
        surface_variables=normalized_surface_variables,
        static_variables=normalized_single_variables,
        atmospheric_variables=normalized_atmospheric_variables,
        batch_metadata=self.batch_metadata,
    )


DataBatch.normalize_data = normalize_data


def unnormalise_data(self) -> "DataBatch":
    """
    Unnormalise all variables in the batch.

    Returns:
        DataBatch: A new batch with all variables unnormalized.
    """
    unnormalized_surface_variables = {
        key: normalise_surface_variable(value, key)
        for key, value in self.surface_variables.items()
    }
    unnormalized_single_variables = {
        key: normalise_surface_variable(value, key)
        for key, value in self.single_variables.items()
    }
    unnormalized_atmospheric_variables = {
        key: normalise_atmospheric_variables(
            value, key, self.batch_metadata.pressure_levels
        )
        for key, value in self.atmospheric_variables.items()
    }

    return DataBatch(
        surface_variables=unnormalized_surface_variables,
        static_variables=unnormalized_single_variables,
        atmospheric_variables=unnormalized_atmospheric_variables,
        batch_metadata=self.batch_metadata,
    )


DataBatch.unnormalise_data = unnormalise_data


def normalise_surface_variable(
    tensor: torch.Tensor,
    variable_name: str,
    reverse: bool = False,
) -> torch.Tensor:
    """
    Apply normalization or unnormalization to a surface-level variable.

    Args:
        tensor (torch.Tensor): The input tensor representing the surface-level variables.
        variable_name (str): The name of the surface-level variable.
        reverse (bool, optional): If True, applies unnormalization by reversing the
            normalization process. If False, applies normalization. Defaults to False.

    Returns:
        torch.Tensor: The normalized or unnormalized tensor, depending on the value of the `reverse` parameter.
    """
    location = locations[variable_name]
    scale = scales[variable_name]

    if reverse:
        return tensor * scale + location
    else:
        return (tensor - location) / scale


DataBatch.normalise_surface_variable = normalise_surface_variable


def normalise_atmospheric_variables(
    tensor: torch.Tensor,
    variable_name: str,
    pressure_levels: tuple[int | float, ...],
    reverse: bool = False,
) -> torch.Tensor:
    """
    Apply normalization or unnormalization to a atmospheric-level variable.

    Args:
        tensor (torch.Tensor): The input tensor representing the atmospheric-level variables.
        variable_name (str): The name of the atmospheric-level variable.
        pressure_levels (tuple[int | float, ...]): A tuple of pressure levels (in hPa) corresponding
            to the levels at which the atmospheric variable is measured.
        reverse (bool, optional): If True, applies unnormalization by reversing the
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

    if reverse:
        return tensor * scale[..., None, None] + location[..., None, None]
    else:
        return (tensor - location[..., None, None]) / scale[..., None, None]


unnormalise_surface_variables = partial(normalise_surface_variable, reverse=True)
unnormalise_atmospheric_variables = partial(
    normalise_atmospheric_variables, reverse=True
)

DataBatch.normalise_atmospheric_variables = normalise_atmospheric_variables
DataBatch.unnormalise_surface_variables = unnormalise_surface_variables
DataBatch.unnormalise_atmospheric_variables = unnormalise_atmospheric_variables


def get_mean_standard_deviation(
    surface_path: str, atmospheric_path: str
) -> Tuple[Dict[str, float], Dict[str, float]]:

    surface_dataset = xarray.open_dataset(
        Path(settings.ERA5_DIR) / surface_path, engine="netcdf4"
    )
    atmospheric_dataset = xarray.open_dataset(
        Path(settings.ERA5_DIR) / atmospheric_path, engine="netcdf4"
    )

    locations: dict[str, float] = {}
    scales: dict[str, float] = {}
    surface_variables = ["2t", "10u", "10v", "msl"]
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
    "ERA5-Reanalysis-surface-2010-2023.nc", "ERA5-Reanalysis-pressure-2010-2023.nc"
)

DataBatch.get_mean_standard_deviation = get_mean_standard_deviation

# src/dataset_creation/preprocessing.py

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing.transformation.text import label_encode
from src.dataset_creation.batch import DataBatch
from src.dataset_creation.load_data import load_species_data


def preprocess_and_normalize_species_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess and normalize the species dataset. This includes:
    - Scaling numeric columns such as Latitude and Longitude using StandardScaler.
    - Converting timestamp data to a standardized format (datetime without timezone).
    - Label encoding categorical columns such as Species, Phylum, Class, etc.
    - Converting Image, Audio, eDNA, and Description columns to tensors.

    Args:
        dataset (pd.DataFrame): The input dataset containing species data.

    Returns:
        pd.DataFrame: The preprocessed and normalized dataset with tensors and label-encoded categories.
    """

    if "Timestamp" in dataset.columns:
        dataset["Timestamp"] = pd.to_datetime(
            dataset["Timestamp"], errors="coerce", utc=True
        ).dt.tz_localize(None)

    dataset["Timestamp"] = dataset["Timestamp"].apply(
        lambda ts: ts.to_pydatetime() if pd.notnull(ts) else None
    )
    dataset["Timestamp"] = dataset["Timestamp"].apply(
        lambda ts: round_to_nearest_hour(ts) if pd.notnull(ts) else None
    )

    dataset["Timestamp"] = dataset["Timestamp"].apply(
        lambda ts: (np.datetime64(ts).astype("datetime64[s]").tolist(),)
        if pd.notnull(ts)
        else None
    )

    dataset["Species"] = label_encode(dataset, "Species")
    dataset["Phylum"] = label_encode(dataset, "Phylum")
    dataset["Class"] = label_encode(dataset, "Class")
    dataset["Order"] = label_encode(dataset, "Order")
    dataset["Family"] = label_encode(dataset, "Family")
    dataset["Genus"] = label_encode(dataset, "Genus")
    dataset["Redlist"] = label_encode(dataset, "Redlist")

    dataset["Latitude"] = dataset["Latitude"].apply(
        lambda lat: round_to_nearest_grid(lat) if pd.notnull(lat) else np.nan
    )
    dataset["Longitude"] = dataset["Longitude"].apply(
        lambda lon: round_to_nearest_grid(lon) if pd.notnull(lon) else np.nan
    )

    dataset["Image"] = dataset["Image"].apply(
        lambda x: np.array(x) if isinstance(x, torch.Tensor) else x
    )

    dataset["Audio"] = dataset["Audio"].apply(
        lambda x: np.array(x) if isinstance(x, torch.Tensor) else x
    )
    dataset["eDNA"] = dataset["eDNA"].apply(
        lambda x: np.array(x) if isinstance(x, torch.Tensor) else x
    )
    dataset["Description"] = dataset["Description"].apply(
        lambda x: np.array(x) if isinstance(x, torch.Tensor) else x
    )
    dataset["Latitude"] = dataset["Latitude"].apply(
        lambda x: torch.tensor(x, dtype=torch.float16)
    )
    dataset["Longitude"] = dataset["Longitude"].apply(
        lambda x: torch.tensor(x, dtype=torch.float16)
    )

    return dataset


def round_to_nearest_grid(value: float, grid_spacing: float = 0.25) -> float:
    """
    Rounds a given value (latitude or longitude) to the nearest grid point.

    Args:
        value (float): The value to round.
        grid_spacing (float): The grid spacing, default is 0.25 for ERA5.

    Returns:
        float: The value rounded to the nearest grid point.
    """
    return round(value / grid_spacing) * grid_spacing


def round_to_nearest_hour(species_time: datetime, interval_hours: int = 6) -> datetime:
    """
    Rounds a given datetime to the nearest ERA5 time slot (00:00, 06:00, 12:00, 18:00).

    Args:
        species_time (datetime): The species timestamp.
        interval_hours (int): Interval between time slots in hours (default is 6 for ERA5 slots).

    Returns:
        datetime: The timestamp rounded to the nearest time slot based on the given interval.
    """
    time_slots = [
        (datetime.min + timedelta(hours=i)).time() for i in range(0, 24, interval_hours)
    ]

    species_time_only = species_time.time()
    closest_time = min(
        time_slots,
        key=lambda t: abs(
            timedelta(hours=species_time_only.hour, minutes=species_time_only.minute)
            - timedelta(hours=t.hour)
        ),
    )

    return species_time.replace(
        hour=closest_time.hour, minute=0, second=0, microsecond=0
    )


def merge_timestamps(
    climate_dataset: xr.Dataset, species_dataset: pd.DataFrame
) -> List[datetime]:
    """
    Merge timestamps from the xarray climate dataset and species data.

    Args:
        climate_dataset (xarray.Dataset): Dataset containing climate variables with timestamps.
        species_dataset (pd.DataFrame): DataFrame containing species variables with timestamps.

    Returns:
        List[Tuple[datetime.datetime]]: Sorted list of all unique timestamps from both datasets in tuple format.
    """
    climate_timestamps = {
        (pd.to_datetime(ts).to_pydatetime(),)
        for ts in set(
            climate_dataset["valid_time"].values.astype("datetime64[s]").tolist()
        )
        if ts is not None
    }

    species_timestamps = {
        (pd.to_datetime(ts[0]).to_pydatetime(),)
        if isinstance(ts, tuple)
        else (pd.to_datetime(ts).to_pydatetime(),)
        for ts in species_dataset["Timestamp"].unique()
        if ts is not None and pd.notna(ts)
    }

    all_timestamps = sorted(climate_timestamps | species_timestamps)

    return all_timestamps


def initialize_climate_tensors(
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    T: int,
    pressure_levels: int = 13,
) -> Dict[str, torch.Tensor]:
    """
    Create empty tensors for surface, atmospheric, and single variables based on the dataset's variables.

    Args:
        lat_range (np.ndarray): Latitude range.
        lon_range (np.ndarray): Longitude range.
        T (int): Number of timestamps.
        pressure_levels (int): Number of pressure levels for atmospheric variables.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Dictionary of empty tensors for climate data.
    """

    return {
        "surface": {
            "t2m": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float32),
            "msl": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float32),
            "u10": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float32),
            "v10": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float32),
        },
        "single": {
            "z": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float32),
            "lsm": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float32),
            "slt": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float32),
        },
        "atmospheric": {
            "z": torch.empty(
                T,
                pressure_levels,
                len(lat_range),
                len(lon_range),
                dtype=torch.float32,
            ),
            "t": torch.empty(
                T,
                pressure_levels,
                len(lat_range),
                len(lon_range),
                dtype=torch.float32,
            ),
            "u": torch.empty(
                T,
                pressure_levels,
                len(lat_range),
                len(lon_range),
                dtype=torch.float32,
            ),
            "v": torch.empty(
                T,
                pressure_levels,
                len(lat_range),
                len(lon_range),
                dtype=torch.float32,
            ),
            "q": torch.empty(
                T,
                pressure_levels,
                len(lat_range),
                len(lon_range),
                dtype=torch.float32,
            ),
        },
    }


def initialize_species_tensors(
    lat_range: np.ndarray, lon_range: np.ndarray, T: int
) -> Dict[str, torch.Tensor]:
    """
    Create empty tensors for species data.

    Args:
        lat_range (np.ndarray): Latitude range.
        lon_range (np.ndarray): Longitude range.
        T (int): Number of timestamps.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of empty tensors for species data.
    """
    return {
        "Species": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float16),
        "Image": torch.empty(T, len(lat_range), len(lon_range), 3, 64, 64),
        "Audio": torch.empty(T, len(lat_range), len(lon_range), 1, 13, 1),
        "Description": torch.empty(T, len(lat_range), len(lon_range), 1, 64, 64),
        "eDNA": torch.empty(T, len(lat_range), len(lon_range), 256),
        "Phylum": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float16),
        "Class": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float16),
        "Order": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float16),
        "Family": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float16),
        "Genus": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float16),
        "Redlist": torch.empty(T, len(lat_range), len(lon_range), dtype=torch.float16),
    }


def reset_climate_tensors(surfaces_variables, single_variables, atmospheric_variables):
    """
    Reset the climate-related tensors to zero. This function iterates over the dictionary of variables
    and sets each tensor to zero, ensuring that no previous values remain for further computations.

    Args:
        surfaces_variables (dict): Dictionary of surface variable tensors.
        single_variables (dict): Dictionary of single-level variable tensors.
        atmospheric_variables (dict): Dictionary of atmospheric variable tensors.
    """
    for var_name, tensor in surfaces_variables.items():
        surfaces_variables[var_name] = torch.empty_like(tensor)

    for var_name, tensor in single_variables.items():
        single_variables[var_name] = torch.empty_like(tensor)

    for var_name, tensor in atmospheric_variables.items():
        atmospheric_variables[var_name] = torch.empty_like(tensor)


def reset_species_tensors(species_variables):
    """
    Reset the species-related tensors to zero. For tensors with more than two dimensions,
    the function sets all values to zero. For lower-dimensional tensors (e.g., vectors or matrices),
    the behavior is the same.

    Args:
        species_variables (dict): Dictionary of species-related tensors.
    """
    for var_name, tensor in species_variables.items():
        species_variables[var_name] = torch.empty_like(tensor)


def preprocess_era5(
    batch: DataBatch,
    dtype: torch.dtype,
    device: torch.device,
    locations: Dict[str, float],
    scales: Dict[str, float],
) -> DataBatch:
    """
    Prepares the batch by applying data type conversion, normalization, cropping,
    and transferring the batch to the specified device.

    Args:
        batch (DataBatch): The input batch containing data.
        dtype (torch.dtype): The target data type for the batch.
        device (torch.device): The device to which the batch should be transferred (e.g., CPU or GPU).
            locations (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the mean for that variable.
            scales (Dict[str, float]): A dictionary where the key is the variable name
                                        and the value is the standard deviation for that variable.

    Returns:
        DataBatch: The prepared batch ready for use in the model.
    """
    batch = batch.type(dtype)
    batch = batch.normalize_data(locations, scales)
    patch_size = 4
    batch = batch.crop(patch_size=patch_size)
    batch = batch.to(device)

    return batch

# src/dataset_creation/preprocessing.py

import glob
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data_preprocessing.transformation.text import label_encode
from src.dataset_creation.batch import DataBatch

pd.set_option("future.no_silent_downcasting", True)


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
        dataset["Timestamp"] = dataset["Timestamp"].apply(
            lambda x: np.nan if x in ["Unknown", "Uknown", None, np.nan] else x
        )

        dataset["Timestamp"] = dataset["Timestamp"].apply(standardize_timestamp_format)

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

    if "Image" in dataset.columns:
        dataset["Image"] = dataset["Image"].apply(
            lambda x: np.array(x) if isinstance(x, torch.Tensor) else x
        )
    if "Audio" in dataset.columns:
        dataset["Audio"] = dataset["Audio"].apply(
            lambda x: np.array(x) if isinstance(x, torch.Tensor) else x
        )
    if "eDNA" in dataset.columns:
        dataset["eDNA"] = dataset["eDNA"].apply(
            lambda x: np.array(x) if isinstance(x, torch.Tensor) else x
        )
    if "Description" in dataset.columns:
        dataset["Description"] = dataset["Description"].apply(
            lambda x: np.array(x) if isinstance(x, torch.Tensor) else x
        )

    dataset["Latitude"] = dataset["Latitude"].apply(
        lambda x: torch.tensor(x, dtype=torch.float16)
    )
    dataset["Longitude"] = dataset["Longitude"].apply(
        lambda x: torch.tensor(x, dtype=torch.float16)
    )

    if "Distribution" in dataset.columns:
        scaler = StandardScaler()
        dataset["Distribution"] = scaler.fit_transform(
            dataset[["Distribution"]].fillna(0)
        )
        dataset["Distribution"] = dataset["Distribution"].apply(
            lambda x: torch.tensor(x, dtype=torch.float16)
        )

    return dataset


def standardize_timestamp_format(ts):
    """
    Converts a timestamp to a standardized ISO 8601 format without timezone information.

    Args:
        ts (Any): The timestamp to standardize. Can be a string, datetime, or other format that
                  can be parsed by pd.to_datetime().

    Returns:
        str or None: The timestamp in ISO 8601 format without timezone if conversion succeeds;
                     None if the input is invalid or missing.
    """
    if pd.notnull(ts):
        ts = pd.to_datetime(ts, utc=True).tz_localize(None)
        return ts.strftime("%Y-%m-%dT%H:%M:%S")
    return None


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


def crop_lat_lon(
    lat_range: np.ndarray, lon_range: np.ndarray, crop_lat_n: int, crop_lon_n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop latitude and longitude arrays to keep only the first 'crop_lat_n' for latitude
    and 'crop_lon_n' for longitude.

    Args:
        lat_range (np.ndarray): Array of latitude values.
        lon_range (np.ndarray): Array of longitude values.
        crop_lat_n (int): Number of latitude values to keep.
        crop_lon_n (int): Number of longitude values to keep.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cropped arrays of latitude and longitude values.
    """
    cropped_lat_range = lat_range[: min(len(lat_range), crop_lat_n)]
    cropped_lon_range = lon_range[: min(len(lon_range), crop_lon_n)]

    return cropped_lat_range, cropped_lon_range


def rescale_sort_lat_lon(
    lat_range: np.ndarray, lon_range: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rescale longitude values to [0, 360] and sort both latitude and longitude arrays.

    Args:
        lat_range (np.ndarray): Array containing latitude values.
        lon_range (np.ndarray): Array containing longitude values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Sorted arrays of latitude and rescaled longitude values.
    """
    lat_range_clean = [lat for lat in lat_range if not np.isnan(lat)]
    sorted_lat_range = sorted(lat_range_clean, reverse=True)

    lon_range_clean = [lon for lon in lon_range if not np.isnan(lon)]
    lon_range_rescaled = [
        (lon + 360) % 360 if lon < 0 else lon for lon in lon_range_clean
    ]
    sorted_lon_range = sorted(lon_range_rescaled)

    lat_range = np.array(sorted_lat_range)
    lon_range = np.array(sorted_lon_range)
    lat_range = lat_range.astype(float)
    lon_range = lon_range.astype(float)

    return lat_range, lon_range


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
    placeholder_value: float = float("nan"),
) -> Dict[str, torch.Tensor]:
    """
    Create tensors for surface, atmospheric, and single variables based on the dataset's variables, initialized with a placeholder value (NaN by default).

    Args:
        lat_range (np.ndarray): Latitude range.
        lon_range (np.ndarray): Longitude range.
        T (int): Number of timestamps.
        pressure_levels (int): Number of pressure levels for atmospheric variables.
        placeholder_value (float): Value to initialize tensors with. Default is NaN.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Dictionary of initialized tensors for climate data.
    """

    return {
        "surface": {
            "t2m": torch.full(
                (T, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "msl": torch.full(
                (T, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "u10": torch.full(
                (T, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "v10": torch.full(
                (T, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
        },
        "single": {
            "z": torch.full(
                (T, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "lsm": torch.full(
                (T, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "slt": torch.full(
                (T, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
        },
        "atmospheric": {
            "z": torch.full(
                (T, pressure_levels, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "t": torch.full(
                (T, pressure_levels, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "u": torch.full(
                (T, pressure_levels, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "v": torch.full(
                (T, pressure_levels, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
            "q": torch.full(
                (T, pressure_levels, len(lat_range), len(lon_range)),
                placeholder_value,
                dtype=torch.float32,
            ),
        },
    }


def initialize_species_tensors(
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    T: int,
    placeholder_value: float = float("nan"),
) -> Dict[str, torch.Tensor]:
    """
    Create tensors for species data, initialized with a placeholder value (NaN by default).

    Args:
        lat_range (np.ndarray): Latitude range.
        lon_range (np.ndarray): Longitude range.
        T (int): Number of timestamps.
        placeholder_value (float): Value to initialize tensors with. Default is NaN.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of initialized tensors for species data.
    """
    return {
        "Species": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        # "Image": torch.full(T, len(lat_range), len(lon_range), 3, 64, 64) placeholder_value, dtype=torch.float16),
        # "Audio": torch.full(T, len(lat_range), len(lon_range), 1, 13, 1) placeholder_value, dtype=torch.float16),
        "Description": torch.full(
            (T, len(lat_range), len(lon_range), 1, 64, 64),
            placeholder_value,
            dtype=torch.float16,
        ),
        # "eDNA": torch.full(T, len(lat_range), len(lon_range), 256) placeholder_value, dtype=torch.float16),
        "Phylum": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "Class": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "Order": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "Family": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "Genus": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "Redlist": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "Distribution": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
    }


def initialize_species_extinction_tensors(
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    T: int,
    placeholder_value: float = float("nan"),
) -> Dict[str, torch.Tensor]:
    """
    Create tensors for extinct species data, initialized with a placeholder value (NaN by default).

    Args:
        lat_range (np.ndarray): Latitude range.
        lon_range (np.ndarray): Longitude range.
        T (int): Number of timestamps.
        placeholder_value (float): Value to initialize tensors with. Default is NaN.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of initialized tensors for extinct species data.
    """
    return {
        "ExtinctionValue": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
    }


def initialize_land_tensors(
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    T: int,
    placeholder_value: float = float("nan"),
) -> Dict[str, torch.Tensor]:
    """
    Create tensors for land data, initialized with a placeholder value (NaN by default).

    Args:
        lat_range (np.ndarray): Latitude range.
        lon_range (np.ndarray): Longitude range.
        T (int): Number of timestamps.
        placeholder_value (float): Value to initialize tensors with. Default is NaN.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of initialized tensors for land data.
    """
    return {
        "Land": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "NDVI": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
    }


def initialize_agriculture_tensors(
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    T: int,
    placeholder_value: float = float("nan"),
) -> Dict[str, torch.Tensor]:
    """
    Create tensors for agriculture data, initialized with a placeholder value (NaN by default).

    Args:
        lat_range (np.ndarray): Latitude range.
        lon_range (np.ndarray): Longitude range.
        T (int): Number of timestamps.
        placeholder_value (float): Value to initialize tensors with. Default is NaN.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of initialized tensors for agriculture data.
    """
    return {
        "AgricultureLand": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "AgricultureIrrLand": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "ArableLand": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
        "Cropland": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
    }


def initialize_forest_tensors(
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    T: int,
    placeholder_value: float = float("nan"),
) -> Dict[str, torch.Tensor]:
    """
    Create tensors for forest data, initialized with a placeholder value (NaN by default).

    Args:
        lat_range (np.ndarray): Latitude range.
        lon_range (np.ndarray): Longitude range.
        T (int): Number of timestamps.
        placeholder_value (float): Value to initialize tensors with. Default is NaN.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of initialized tensors for forest data.
    """
    return {
        "Forest": torch.full(
            (T, len(lat_range), len(lon_range)), placeholder_value, dtype=torch.float16
        ),
    }


def reset_climate_tensors(
    surfaces_variables: dict,
    single_variables: dict,
    atmospheric_variables: dict,
    placeholder_value: float,
):
    """
    Reset the climate-related tensors to zero. This function iterates over the dictionary of variables
    and sets each tensor to zero, ensuring that no previous values remain for further computations.

    Args:
        surfaces_variables (dict): Dictionary of surface variable tensors.
        single_variables (dict): Dictionary of single-level variable tensors.
        atmospheric_variables (dict): Dictionary of atmospheric variable tensors.
        placeholder_value (float): Value to reset tensors with. Default is NaN.
    """
    for var_name, tensor in surfaces_variables.items():
        surfaces_variables[var_name] = torch.full_like(tensor, placeholder_value)

    for var_name, tensor in single_variables.items():
        single_variables[var_name] = torch.full_like(tensor, placeholder_value)

    for var_name, tensor in atmospheric_variables.items():
        atmospheric_variables[var_name] = torch.full_like(tensor, placeholder_value)


def reset_tensors(variables: dict, placeholder_value: float = float("nan")):
    """
    Reset the tensors to the placeholder value. This function sets all values to the specified placeholder (NaN by default).

    Args:
        variables (dict): Dictionary of tensors.
        placeholder_value (float): Value to reset tensors with. Default is NaN.
    """
    for var_name, tensor in variables.items():
        variables[var_name] = torch.full_like(tensor, placeholder_value)


def preprocess_era5(
    batch: DataBatch,
    dtype: torch.dtype,
    device: torch.device,
    locations: Dict[str, float],
    scales: Dict[str, float],
    crop_mode: str = "truncate",
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
        crop_mode (str): The mode for adjusting the batch dimensions ('truncate' or 'pad').

    Returns:
        DataBatch: The prepared batch ready for use in the model.
    """
    batch = batch.type(dtype)
    batch = batch.normalize_data(locations, scales)
    patch_size = 4
    batch = batch.crop(patch_size=patch_size, mode=crop_mode)
    batch = batch.to(device)

    return batch

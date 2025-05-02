"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import base64
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.data_preprocessing.preprocessing import preprocess_audio, preprocess_image
from src.data_preprocessing.transformation.audio import resize_audio_tensor
from src.data_preprocessing.transformation.image import resize_image_tensor
from src.data_preprocessing.transformation.text import resize_generic_tensor
from src.dataset_creation.batch import DataBatch


def load_era5_datasets(surface_file: str, single_file: str, atmospheric_dataset: str):
    """
    Load ERA5 datasets for surface, single, and atmospheric variables.

    Args:
        surface_file (str): Path to the ERA5 surface dataset.
        single_file (str): Path to the ERA5 single-level dataset.
        atmospheric_dataset (str): Path to the ERA5 pressure-level dataset.

    Returns:
        xarray.Dataset, xarray.Dataset, xarray.Dataset: Loaded surface, single, and atmospheric datasets.
    """
    surface_dataset = xr.open_dataset(surface_file)
    single_dataset = xr.open_dataset(single_file)
    atmospheric_dataset = xr.open_dataset(atmospheric_dataset)

    return (
        surface_dataset,
        single_dataset,
        atmospheric_dataset,
    )


def load_world_bank_data(filepath: str) -> pd.DataFrame:
    """
    Load files from world bank, like forestry, agriculture, land data from a CSV file
    and return it as a DataFrame.
    """
    data = pd.read_csv(filepath, low_memory=False)
    # uniform all the longitudes to be in the range of -180 to 180
    data["Longitude"] = data["Longitude"].apply(lambda x: x - 360.0 if x > 180.0 else x)
    return data


def load_era5_files_grouped_by_date(directory: str) -> List[Tuple[str, str, str]]:
    """
    Load ERA5 files from the directory and group them by date (pressure, single, surface).

    Args:
        directory (str): Directory containing sorted ERA5 NetCDF files.

    Returns:
        List[Tuple[str, str, str]]: A list of tuples, where each tuple contains the pressure, single, and surface file paths for a specific date.
    """
    era5_files = sorted(os.listdir(directory))
    grouped_files = []

    for file in era5_files:
        if "pressure" in file:
            date_str = file.split("-")[-3]

            atmospheric_file = os.path.join(directory, file)
            single_file = os.path.join(directory, file.replace("pressure", "single"))
            surface_file = os.path.join(directory, file.replace("pressure", "surface"))

            if os.path.exists(single_file) and os.path.exists(surface_file):
                grouped_files.append((atmospheric_file, single_file, surface_file))
            else:
                print(
                    f"Skipping date {date_str} as corresponding single or surface file is missing."
                )

    return grouped_files


def resize_to_target(data, target_shape: tuple) -> torch.Tensor:
    """
    Resizes or pads data to match the target shape, handling image, audio, and generic tensor data.

    Args:
        data (torch.Tensor or np.ndarray): The input data to resize.
        target_shape (tuple): The desired shape (e.g., (3, 64, 64) for images).

    Returns:
        torch.Tensor: The data resized or padded to the target shape.

    Raises:
        TypeError: If the data type is unsupported for resizing.
    """
    if isinstance(data, torch.Tensor):
        if data.dim() == 3 and data.shape[0] == 3:
            return resize_image_tensor(data, target_shape)

        elif data.dim() == 3 and target_shape[-1] == 1:
            return resize_audio_tensor(data, target_shape)

        else:
            return resize_generic_tensor(data, target_shape)

    elif isinstance(data, np.ndarray):
        return resize_to_target(torch.tensor(data), target_shape)

    else:
        raise TypeError("Unsupported data type for resizing")


def load_species_data(species_file: str) -> pd.DataFrame:
    """
    Loads and processes species data from a Parquet file, deserializing multimodal data fields
    and resizing them to consistent shapes if necessary.

    This function reads a Parquet file containing species data, deserializes base64-encoded arrays,
    and applies reshaping to standardize data shapes for images, audio, descriptions, and eDNA.

    Args:
        species_file (str): Path to the Parquet file containing serialized species data.

    Returns:
        pd.DataFrame: A DataFrame with the processed species data, including deserialized
                      arrays with consistent shapes.
    """
    species_dataset = pd.read_parquet(species_file)

    species_dataset["Image"] = species_dataset["Image"].apply(
        lambda x: (
            resize_to_target(
                deserialize_array(x, [(3, 64, 64), (3, 128, 128)]), (3, 64, 64)
            )
            if isinstance(x, str)
            else x
        )
    )

    species_dataset["Audio"] = species_dataset["Audio"].apply(
        lambda x: (
            resize_to_target(deserialize_array(x, [(1, 13, 1)]), (1, 13, 1))
            if isinstance(x, str)
            else x
        )
    )

    species_dataset["Description"] = species_dataset["Description"].apply(
        lambda x: (
            resize_to_target(deserialize_array(x, [(64, 64), (128,)]), (1, 64, 64))
            if isinstance(x, str)
            else x
        )
    )

    species_dataset["eDNA"] = species_dataset["eDNA"].apply(
        lambda x: (
            resize_to_target(deserialize_array(x, [(256,)]), (256,))
            if isinstance(x, str)
            else x
        )
    )

    species_dataset["Timestamp"] = species_dataset["Timestamp"].apply(extract_timestamp)

    return species_dataset


def deserialize_array(arr_str, expected_shapes, target_shape=None):
    """
    Deserializes a base64-encoded string back into a tensor and adjusts the shape if necessary.

    Args:
        arr_str (str): The base64-encoded string representing the array.
        expected_shapes (list of tuples): List of possible shapes for the tensor.
        target_shape (tuple, optional): Target shape to resize to if the deserialized array does not match expected shapes.

    Returns:
        torch.Tensor: The deserialized and reshaped tensor.

    Raises:
        ValueError: If the deserialized array size does not match any of the expected shapes and no target shape is provided.
    """
    decoded = base64.b64decode(arr_str.encode("utf-8"))
    arr = np.frombuffer(decoded, dtype=np.float32)
    for shape in expected_shapes:
        if arr.size == np.prod(shape):
            return torch.tensor(arr).view(*shape)
    if target_shape:
        print(f"Warning: Resizing from {arr.shape} to {target_shape}")
        return torch.tensor(arr).reshape(*target_shape)
    raise ValueError(f"Shape {arr.shape} did not match any expected {expected_shapes}")


# def deserialize_array(arr_str, expected_shapes):
#     """
#     Deserializes a base64-encoded string back into a tensor with one of the specified shapes.

#     Args:
#         arr_str (str): The base64-encoded string representing the array.
#         expected_shapes (list of tuples): The possible shapes for the tensor.

#     Returns:
#         torch.Tensor: The deserialized tensor.
#     """
#     decoded = base64.b64decode(arr_str.encode("utf-8"))
#     arr = np.frombuffer(decoded, dtype=np.float32)

#     for shape in expected_shapes:
#         if isinstance(shape, tuple) and arr.size == np.prod(shape):
#             return torch.tensor(arr).view(*shape)

#     raise ValueError(f"Deserialized array size {arr.size} does not match any expected shapes {expected_shapes}.")


def extract_timestamp(x):
    """
    Extract the timestamp from the input value.

    This function handles three types of input:
    1. If the input is a non-empty NumPy array, it extracts and returns the first element as a tuple.
    2. If the input is a string or a pandas Timestamp, it returns the input as a tuple.
    3. If the input is none of the above, it returns None.

    Args:
        x: The input value, which can be a NumPy array, string, or pandas Timestamp.

    Returns:
        tuple: A tuple containing the extracted timestamp if the input is valid, otherwise None.
    """
    if isinstance(x, np.ndarray) and len(x) > 0:
        return x[0]
    elif isinstance(x, (str, pd.Timestamp)):
        return x
    return None


def load_batches(batch_directory: str) -> list:
    """
    Load all saved DataBatch objects from the specified directory.

    Args:
        batch_directory (str): The directory where the batch .pt files are stored.

    Returns:
        list: A list of DataBatch objects loaded from the directory.
    """
    batches = []

    for batch_file in sorted(os.listdir(batch_directory)):
        if batch_file.endswith(".pt"):
            batch_path = os.path.join(batch_directory, batch_file)
            print(f"Loading batch: {batch_path}")

            batch = torch.load(batch_path)
            batches.append(batch)

    return batches

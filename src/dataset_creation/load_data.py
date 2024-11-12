# src/dataset_creation/load_data.py

import base64
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr

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
    data = pd.read_csv(filepath)
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


def load_species_data(species_file: str):
    """
    Load species data from a Parquet file and process the multimodal data.

    This function reads the species dataset from a Parquet file, processes the data by deserializing arrays,
    and extracts timestamps. The deserialization of arrays converts string-encoded data back into
    their original array form with the expected shapes for images, audio, and other fields.

    Args:
        species_file (str): Path to the Parquet file containing species data.

    Returns:
        pd.DataFrame: A DataFrame containing the processed species data with deserialized arrays.
    """
    species_dataset = pd.read_parquet(species_file)

    species_dataset["Image"] = species_dataset["Image"].apply(
        lambda x: deserialize_array(x, (3, 64, 64)) if isinstance(x, str) else x
    )

    species_dataset["Audio"] = species_dataset["Audio"].apply(
        lambda x: deserialize_array(x, (1, 13, 1)) if isinstance(x, str) else x
    )

    species_dataset["Description"] = species_dataset["Description"].apply(
        lambda x: deserialize_array(x, (64, 64)) if isinstance(x, str) else x
    )

    species_dataset["eDNA"] = species_dataset["eDNA"].apply(
        lambda x: deserialize_array(x, (256,)) if isinstance(x, str) else x
    )

    species_dataset["Timestamp"] = species_dataset["Timestamp"].apply(extract_timestamp)

    return species_dataset


def deserialize_array(arr_str, shape):
    """
    Deserializes a base64-encoded string back into a tensor with the specified shape.

    Args:
        arr_str (str): The base64-encoded string representing the array.
        shape (tuple): The target shape for the tensor.

    Returns:
        torch.Tensor: The deserialized tensor.
    """
    decoded = base64.b64decode(arr_str.encode("utf-8"))

    arr = np.frombuffer(decoded, dtype=np.float32)
    reshaped_tensor = torch.tensor(arr).view(*shape)

    return reshaped_tensor


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
        return (x[0],)
    elif isinstance(x, (str, pd.Timestamp)):
        return (x,)
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


def print_batch_variables(batch: DataBatch) -> None:
    """
    Print all the variable values from the given DataBatch.

    Args:
        batch (DataBatch): The batch object containing climate and species data.
    """
    print("Surface Variables:")
    for var_name, var_data in batch.surface_variables.items():
        print(f"{var_name}: {var_data}")

    print("\nSingle Variables:")
    for var_name, var_data in batch.single_variables.items():
        print(f"{var_name}: {var_data}")

    print("\nAtmospheric Variables:")
    for var_name, var_data in batch.atmospheric_variables.items():
        print(f"{var_name}: {var_data}")

    print("\nSpecies Variables:")
    for var_name, var_data in batch.species_variables.items():
        print(f"{var_name}: {var_data}")

    print("\nMetadata:")
    print(f"Latitudes: {batch.batch_metadata.latitudes}")
    print(f"Longitudes: {batch.batch_metadata.longitudes}")
    print(f"Timestamps: {batch.batch_metadata.timestamp}")
    print(f"Pressure Levels: {batch.batch_metadata.pressure_levels}")

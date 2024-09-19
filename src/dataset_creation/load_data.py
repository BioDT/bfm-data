# src/dataset_creation/load_data.py

import pandas as pd
import pyarrow.parquet as pq
import torch
import xarray as xr


def load_era5_datasets(surface_file: str, single_file: str, pressure_file: str):
    """
    Load ERA5 datasets for surface, single, and atmospheric variables.

    Args:
        surface_file (str): Path to the ERA5 surface dataset.
        single_file (str): Path to the ERA5 single-level dataset.
        pressure_file (str): Path to the ERA5 pressure-level dataset.

    Returns:
        xarray.Dataset, xarray.Dataset, xarray.Dataset: Loaded surface, single, and pressure datasets.
    """
    surface_variables_dataset = xr.open_dataset(surface_file)
    single_variables_dataset = xr.open_dataset(single_file)
    pressure_variables_dataset = xr.open_dataset(pressure_file)

    return (
        surface_variables_dataset,
        single_variables_dataset,
        pressure_variables_dataset,
    )


import dask.dataframe as dd


def load_species_data(parquet_file: str, batch_size: int):
    """
    Loads the species dataset from a Parquet file in batches and converts the necessary fields to PyTorch tensors,
    then combines all batches into a single pandas DataFrame.

    Args:
        parquet_file (str): Path to the Parquet file.
        batch_size (int): Number of rows to process per batch.

    Returns:
        pd.DataFrame: The combined and processed species dataset.
    """
    parquet_reader = pq.ParquetFile(parquet_file)

    tensor_columns = ["Image", "Audio", "eDNA", "Description"]

    # batch_list = []

    for batch in parquet_reader.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()

        for col in tensor_columns:
            df[col] = df[col].apply(lambda x: torch.tensor(x, dtype=torch.float32))

    #     batch_list.append(df)

    # species_dataset = pd.concat(batch_list, ignore_index=True)

    # return species_dataset

    yield df

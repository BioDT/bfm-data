# src/data_creation/save_data.py

import base64

import numpy as np
import pandas as pd
import torch

from src.dataset_creation.batch import DataBatch


def serialize_array(array):
    """
    Serialize a NumPy array to a Base64-encoded string.

    This function takes a NumPy array, converts it to 32-bit float precision (if not already),
    serializes the array to bytes, and then encodes it into a Base64 string for safe storage or
    transmission as text.

    Args:
        array (np.ndarray): The input NumPy array to be serialized.

    Returns:
        str: The Base64-encoded string representing the serialized NumPy array.
    """
    array = np.array(array, dtype=np.float32)
    serialized = base64.b64encode(array.tobytes()).decode("utf-8")
    return serialized


def save_as_parquet(dataset: pd.DataFrame, filepath: str):
    """
    Save the given dataset as a Parquet file. All tensors are converted back into lists
    for efficient storage in Parquet format.

    Args:
        dataset (pd.DataFrame): The dataset to be saved.
        filepath (str): The path to the output Parquet file.
    """
    dataset["Image"] = dataset["Image"].apply(
        lambda x: serialize_array(np.array(x))
        if isinstance(x, (torch.Tensor, np.ndarray))
        else x
    )
    dataset["Audio"] = dataset["Audio"].apply(
        lambda x: serialize_array(np.array(x))
        if isinstance(x, (torch.Tensor, np.ndarray))
        else x
    )
    dataset["eDNA"] = dataset["eDNA"].apply(
        lambda x: serialize_array(np.array(x))
        if isinstance(x, (torch.Tensor, np.ndarray))
        else x
    )
    dataset["Description"] = dataset["Description"].apply(
        lambda x: serialize_array(x)
        if isinstance(x, (torch.Tensor, np.ndarray, list))
        else x
    )
    dataset["Latitude"] = dataset["Latitude"].apply(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    )
    dataset["Longitude"] = dataset["Longitude"].apply(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    )
    dataset["Distribution"] = dataset["Distribution"].apply(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    )

    dataset.to_parquet(filepath, engine="pyarrow")


def save_batch_metadata_to_parquet(batches: list[DataBatch], filepath: str):
    """
    Save the metadata of each batch (excluding the actual tensor data) into a Parquet file for future use.

    Args:
        batches (list[DataBatch]): List of DataBatch objects.
        filepath (str): Path to the output Parquet file for storing batch metadata.
    """
    batch_metadata_list = []

    for batch in batches:
        metadata = {
            "species_names": batch.species_names.item(),
            "phylum": batch.phylum.item(),
            "class_": batch.class_.item(),
            "order": batch.order.item(),
            "family": batch.family.item(),
            "genus": batch.genus.item(),
            "redlist": batch.redlist.item(),
            "latitude": batch.batch_metadata.species_latitude.numpy().tolist(),
            "longitude": batch.batch_metadata.species_longitude.numpy().tolist(),
            "timestamp": batch.batch_metadata.species_timestamp,
        }
        batch_metadata_list.append(metadata)

    batch_metadata_df = pd.DataFrame(batch_metadata_list)
    batch_metadata_df.to_parquet(filepath)

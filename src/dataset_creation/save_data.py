# src/data_creation/save_data.py

import pandas as pd
import torch

from src.data_preprocessing.batch import DataBatch


def save_as_parquet(dataset: pd.DataFrame, filepath: str):
    """
    Save the given dataset as a Parquet file. All tensors are converted back into numpy arrays
    for efficient storage in Parquet format.

    Args:
        dataset (pd.DataFrame): The dataset to be saved.
        filepath (str): The path to the output Parquet file.
    """
    dataset["Image"] = dataset["Image"].apply(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    )
    dataset["Audio"] = dataset["Audio"].apply(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    )
    dataset["eDNA"] = dataset["eDNA"].apply(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    )
    dataset["Description"] = dataset["Description"].apply(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    )
    dataset["Latitude"] = dataset["Latitude"].apply(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x
    )
    dataset["Longitude"] = dataset["Longitude"].apply(
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

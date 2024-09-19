# src/dataset_creation/preprocessing.py

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing.transformation.text import label_encode


def normalize_species_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the species dataset by scaling numeric columns such as Image, Audio, eDNA,
    and Description using StandardScaler. Convert tensors back and forth for normalization
    and re-apply tensor formatting. Categorical data such as Species, Phylum, Class, etc.
    are label encoded.

    Args:
        dataset (pd.DataFrame): The input dataset containing species data.

    Returns:
        pd.DataFrame: The normalized and processed dataset with tensors converted back for each feature.
    """
    scaler = StandardScaler()

    dataset["Image"] = dataset["Image"].apply(lambda x: np.nan_to_num(np.array(x)))
    dataset["Image"] = dataset["Image"].apply(
        lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten() if len(x) > 0 else x
    )

    dataset["Audio"] = dataset["Audio"].apply(lambda x: np.nan_to_num(np.array(x)))
    dataset["Audio"] = dataset["Audio"].apply(
        lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten() if len(x) > 0 else x
    )

    dataset["eDNA"] = dataset["eDNA"].apply(
        lambda x: np.nan_to_num(np.array(x)) if x is not None else np.array([])
    )
    dataset["eDNA"] = dataset["eDNA"].apply(
        lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten() if len(x) > 0 else x
    )

    dataset["Description"] = dataset["Description"].apply(
        lambda x: np.nan_to_num(np.array(x)) if x is not None else np.array([])
    )
    dataset["Description"] = dataset["Description"].apply(
        lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten() if len(x) > 0 else x
    )

    if "Latitude" in dataset.columns and "Longitude" in dataset.columns:
        dataset[["Latitude", "Longitude"]] = scaler.fit_transform(
            dataset[["Latitude", "Longitude"]].fillna(0)
        )

    dataset["Species"] = label_encode(dataset, "Species")
    dataset["Phylum"] = label_encode(dataset, "Phylum")
    dataset["Class"] = label_encode(dataset, "Class")
    dataset["Order"] = label_encode(dataset, "Order")
    dataset["Family"] = label_encode(dataset, "Family")
    dataset["Genus"] = label_encode(dataset, "Genus")
    dataset["Redlist"] = label_encode(dataset, "Redlist")

    dataset["Image"] = dataset["Image"].apply(
        lambda x: torch.tensor(x, dtype=torch.float32)
    )
    dataset["Audio"] = dataset["Audio"].apply(
        lambda x: torch.tensor(x, dtype=torch.float32)
    )
    dataset["eDNA"] = dataset["eDNA"].apply(
        lambda x: torch.tensor(x, dtype=torch.float32) if x is not None else None
    )
    dataset["Description"] = dataset["Description"].apply(
        lambda x: torch.tensor(x, dtype=torch.float32) if x is not None else None
    )
    dataset["Latitude"] = dataset["Latitude"].apply(
        lambda x: torch.tensor(x, dtype=torch.float32)
    )
    dataset["Longitude"] = dataset["Longitude"].apply(
        lambda x: torch.tensor(x, dtype=torch.float32)
    )

    return dataset


def pad_tensors(tensor_list: list, pad_value: float = 0) -> torch.Tensor:
    """
    Pad a list of tensors to the maximum size found in the list. This ensures all tensors have
    uniform size for batching.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to be padded.
        pad_value (float): Value to pad with. Default is 0.

    Returns:
        torch.Tensor: A tensor containing padded tensors.
    """
    max_size = max(tensor.size(0) for tensor in tensor_list)
    padded_tensors = [
        F.pad(tensor, (0, max_size - tensor.size(0)), value=pad_value)
        for tensor in tensor_list
    ]
    return torch.stack(padded_tensors)

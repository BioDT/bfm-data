"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import base64
import hashlib
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch


def serialize_array(array: np.ndarray) -> str:
    """
    Serialize a NumPy array or tensor to a Base64-encoded string in a deterministic manner.
    Handles `None` or invalid values gracefully.

    Args:
        array (np.ndarray): The input NumPy array to be serialized.

    Returns:
        serialized (str): The Base64-encoded string representing the serialized NumPy array or "error" if serialization fails.
    """
    try:
        if array is None:
            return "none"

        if isinstance(array, torch.Tensor):
            array = array.numpy()

        if isinstance(array, str):
            return array

        array = np.array(array, dtype=np.float32)
        array = np.sort(array.flatten())
        serialized = base64.b64encode(array.tobytes()).decode("utf-8")
        return serialized

    except Exception as e:
        logging.error(f"Error serializing array: {array}. Error: {e}")
        return "error"


def generate_unique_id(row: pd.Series, serialized_columns: List[str]) -> str:
    """
    Generate a stable unique ID by hashing non-serialized columns.
    This function combines non-serialized values (like numerical columns) and serialized
    data (like arrays or tensors) to create a deterministic unique identifier for a row.
    It also includes the file path hash to ensure uniqueness across different files.

    Args:
        row (pd.Series): A row from the DataFrame containing data for which the unique ID is to be generated.
        serialized_columns (List[str]): A list of column names that should be serialized (e.g., Image, Audio, eDNA).

    Returns:
        str: A unique identifier (MD5 hash) generated from the row data, including non-serialized values.

    Raises:
        Exception: If an error occurs during the generation of the unique ID, an exception is logged and raised.
    """
    try:
        non_serialized_values = "_".join(
            str(row[col])
            for col in row.index
            if col not in serialized_columns and pd.notnull(row[col])
        )

        file_path_hash = (
            hashlib.md5(row["File_path"].encode()).hexdigest()
            if "file_path" in row
            else "none"
        )

        full_id = f"{non_serialized_values}_{file_path_hash}"
        unique_id = hashlib.md5(full_id.encode()).hexdigest()

        return unique_id
    except Exception as e:
        logging.error(f"Error generating unique_id for row {row.to_dict()}: {e}")
        raise


def save_incrementally(data_batch: pd.DataFrame, filepath: str) -> None:
    """
    Save data incrementally to a Parquet file, skipping duplicates based on `unique_id`.

    This function processes and serializes specific columns of a data batch (such as images, audio, eDNA, and descriptions),
    generates unique IDs based on serialized and non-serialized data, and saves the data to a Parquet file,
    skipping any rows with duplicate `unique_id`s that already exist in the file.

    Args:
        data_batch (pd.DataFrame): A DataFrame containing the data to be saved.
        filepath (str): The path to the Parquet file where the data should be saved.

    Returns:
        None
    """
    if data_batch.empty:
        logging.warning("Data batch is empty. Skipping save operation.")
        return

    columns_to_serialize = ["Image", "Audio", "eDNA", "Description"]

    for column in columns_to_serialize:
        if column in data_batch.columns:
            data_batch[column] = data_batch[column].apply(
                lambda x: serialize_array(
                    x.numpy() if isinstance(x, torch.Tensor) else x
                )
                if isinstance(x, (torch.Tensor, np.ndarray))
                else x
            )

    if "unique_id" not in data_batch.columns:
        data_batch["unique_id"] = data_batch.apply(
            lambda row: generate_unique_id(row, columns_to_serialize), axis=1
        )

    logging.debug(f"Generated unique_ids: {data_batch['unique_id'].tolist()}")

    for column in ["Latitude", "Longitude", "Distribution"]:
        if column in data_batch.columns:
            data_batch[column] = data_batch[column].apply(
                lambda x: x.item() if isinstance(x, torch.Tensor) else x
            )

    if "Timestamp" in data_batch.columns:
        data_batch["Timestamp"] = data_batch["Timestamp"].apply(
            lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notnull(x)
            else None
        )

    if os.path.exists(filepath):
        try:
            existing_data = pd.read_parquet(filepath)

            existing_data = existing_data.dropna(how="all", axis=1)
            data_batch = data_batch.dropna(how="all", axis=1)

            existing_unique_ids = existing_data["unique_id"].tolist()
            new_rows = data_batch[~data_batch["unique_id"].isin(existing_unique_ids)]

            skipped_rows = len(data_batch) - len(new_rows)
            if skipped_rows > 0:
                skipped_details = data_batch[
                    data_batch["unique_id"].isin(existing_unique_ids)
                ]
                logging.info(
                    f"Skipped {skipped_rows} duplicate rows based on unique_id: {skipped_details}"
                )

            combined_data = pd.concat([existing_data, new_rows], ignore_index=True)
        except Exception as e:
            logging.error(f"Error reading existing data: {e}")
            combined_data = data_batch
    else:
        combined_data = data_batch

    try:
        combined_data.to_parquet(filepath, index=False, engine="pyarrow")
        logging.info(
            f"Data saved successfully to {filepath}. Total rows: {len(combined_data)}."
        )
    except Exception as e:
        logging.error(f"Error saving data: {e}")

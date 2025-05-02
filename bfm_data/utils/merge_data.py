"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import os
from datetime import datetime
from typing import Tuple

import pandas as pd


def average_coordinates_and_time(
    image_lat: float,
    image_lon: float,
    image_time: str,
    audio_lat: float,
    audio_lon: float,
    audio_time: str,
) -> Tuple[float, float, datetime]:
    """
    This function calculates the average latitude, longitude, and timestamp between an image and an audio file.

    Parameters:
    - image_lat (float): Latitude of the image.
    - image_lon (float): Longitude of the image.
    - image_time (str or datetime): Timestamp of the image.
    - audio_lat (float): Latitude of the audio recording.
    - audio_lon (float): Longitude of the audio recording.
    - audio_time (str or datetime): Timestamp of the audio recording.

    Returns:
    - avg_lat (float): Averaged latitude between the image and audio.
    - avg_lon (float): Averaged longitude between the image and audio.
    - avg_time (datetime): Averaged timestamp between the image and audio.
    """
    avg_lat = (image_lat + audio_lat) / 2
    avg_lon = (image_lon + audio_lon) / 2

    try:
        image_time = pd.to_datetime(image_time, errors="raise")
        audio_time = pd.to_datetime(audio_time, errors="raise")
        avg_time = image_time + (audio_time - image_time) / 2
    except (pd.errors.OutOfBoundsDatetime, OverflowError):
        avg_time = pd.NaT

    return avg_lat, avg_lon, avg_time


def matching_image_audios(
    image_metadata: pd.DataFrame, audio_metadata: pd.DataFrame
) -> list:
    """
    This function matches each image with every audio and vice versa.

    Parameters:
    - image_metadata (DataFrame): A DataFrame containing metadata for images.
    - audio_metadata (DataFrame): A DataFrame containing metadata for audio files.

    Returns:
    - matched_pairs (list): A list of dictionaries, where each dictionary contains the image path, audio path,
                            averaged latitude, averaged longitude, and averaged timestamp for a single pair.
    """
    matched_pairs = []

    for _, image_row in image_metadata.iterrows():
        for _, audio_row in audio_metadata.iterrows():
            avg_lat, avg_lon, avg_time = average_coordinates_and_time(
                image_row["Latitude"],
                image_row["Longitude"],
                image_row["Timestamp"],
                audio_row["Latitude"],
                audio_row["Longitude"],
                audio_row["Timestamp"],
            )

            matched_pairs.append(
                {
                    "Image_path": image_row["Image_path"],
                    "Audio_path": audio_row["Audio_path"],
                    "Latitude": avg_lat,
                    "Longitude": avg_lon,
                    "Timestamp": avg_time,
                    "File_path": image_row["File_path"] + audio_row["File_path"],
                }
            )

    return matched_pairs


def extract_metadata_from_csv(metadata_file_path: str, file_type: str) -> dict:
    """
    Extracts metadata from the given CSV file and derives the corresponding image or audio file path.

    Parameters:
    - metadata_file_path (str): Path to the metadata CSV file (either for images or audios).
    - file_type (str): Specifies whether the file is 'image' or 'audio' to derive the correct path.

    Returns:
    - metadata (dict): A dictionary containing the derived image or audio path, latitude, longitude, and timestamp.
    """
    metadata = pd.read_csv(metadata_file_path)

    if file_type == "image":
        return {
            "Image_path": metadata_file_path.replace("_image.csv", ".jpg"),
            "Latitude": metadata["Latitude"].values[0],
            "Longitude": metadata["Longitude"].values[0],
            "Timestamp": metadata["Timestamp"].values[0],
            "Kingdom": metadata["Kingdom"].values[0],
            "Phylum": metadata["Phylum"].values[0],
            "Class": metadata["Class"].values[0],
            "Order": metadata["Order"].values[0],
            "Family": metadata["Family"].values[0],
            "Genus": metadata["Genus"].values[0],
        }
    elif file_type == "audio":
        return {
            "Audio_path": metadata_file_path.replace("_audio.csv", ".wav"),
            "Latitude": metadata["Latitude"].values[0],
            "Longitude": metadata["Longitude"].values[0],
            "Timestamp": metadata["Timestamp"].values[0],
            "Kingdom": metadata["Kingdom"].values[0],
            "Phylum": metadata["Phylum"].values[0],
            "Class": metadata["Class"].values[0],
            "Order": metadata["Order"].values[0],
            "Family": metadata["Family"].values[0],
            "Genus": metadata["Genus"].values[0],
        }


def folder_has_all_modalities(species_path: str) -> bool:
    """
    Checks if a species folder contains at least one image, one audio, and one eDNA file.

    Parameters:
    - species_path (str): Path to the species folder.

    Returns:
    - has_all_modalities (bool): True if the folder contains at least one image, one audio, and one eDNA file, otherwise False.
    """
    has_images = any(file.endswith("_image.csv") for file in os.listdir(species_path))
    has_audios = any(file.endswith("_audio.csv") for file in os.listdir(species_path))
    has_edna = any(file.endswith("_edna.csv") for file in os.listdir(species_path))

    return has_images and has_audios and has_edna


def extract_species_names(file_path: str) -> list:
    """
    Extract species names from a file containing paths.

    Args:
        file_path (str): The path to the text file containing species paths.
        Ex. : /data/projects/biodt/storage/folders_with_only_jpg.txt

    Returns:
        list: A list of species names.
    """
    species_names = []

    with open(file_path, "r") as file:
        for line in file:
            species_name = line.strip().split("/")[-1]
            species_names.append(species_name)

    return species_names


def find_closest_lat_lon(
    target_lat: float, target_lon: float, distribution_df: pd.DataFrame
) -> pd.Series:
    """
    Find the closest latitude and longitude entry in the distribution dataset to a specified target location.

    Args:
        target_lat (float): The latitude of the target location.
        target_lon (float): The longitude of the target location.
        distribution_df (pd.DataFrame): A DataFrame containing distribution data with 'Latitude' and 'Longitude' columns.

    Returns:
        pd.Series: The row in the distribution DataFrame that is closest to the target location, based on Euclidean distance.
    """
    distribution_df["Distance"] = (
        (distribution_df["Latitude"] - target_lat) ** 2
        + (distribution_df["Longitude"] - target_lon) ** 2
    ) ** 0.5
    closest_row = distribution_df.loc[distribution_df["Distance"].idxmin()]
    return closest_row


def merge_world_bank_data(file_paths: list, variable_names: list, output_path: str):
    """
    Main function to process world bank data like agriculture, forest, land files.
    Processes globally if no region specified.

    Args:
        file_paths (list): Paths to the CSV files.
        variable_names (list): Names of the variables for each file (to be added as a column).
        output_path (str): Path to save the merged output CSV file.
    """
    data_frames = []

    if len(file_paths) != len(variable_names):
        raise ValueError("file_paths and variable_names must have the same length.")

    for file_path, variable_name in zip(file_paths, variable_names):
        df = pd.read_csv(file_path)
        df.insert(0, "Variable", variable_name)
        data_frames.append(df)

    merged_df = pd.concat(data_frames, ignore_index=True)

    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")


def save_sorted_timestamps(parquet_file_path: str, output_csv_path: str) -> None:
    """
    Extracts, cleans, and sorts timestamps from a parquet file and saves them to a CSV.

    Args:
        parquet_file_path (str): Path to the input parquet file containing a 'Timestamp' column.
        output_csv_path (str): Path to save the sorted timestamps as a CSV file.
    """
    df = pd.read_parquet(parquet_file_path)

    if isinstance(df["Timestamp"].iloc[0], list):
        df["Timestamp"] = df["Timestamp"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
        )
    else:
        df["Timestamp"] = df["Timestamp"].astype(str).str.strip("[]'")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["Timestamp"])

    if df.empty:
        print("No valid timestamps found after conversion.")
        return

    sorted_df = df[["Timestamp"]].sort_values(by="Timestamp")
    sorted_df.to_csv(output_csv_path, index=False)
    print("Sorted timestamps saved to:", output_csv_path)

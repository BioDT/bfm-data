# src/utils/merge_data.py

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

    image_time = pd.to_datetime(image_time)
    audio_time = pd.to_datetime(audio_time)
    avg_time = image_time + (audio_time - image_time) / 2

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
        }
    elif file_type == "audio":
        return {
            "Audio_path": metadata_file_path.replace("_audio.csv", ".wav"),
            "Latitude": metadata["Latitude"].values[0],
            "Longitude": metadata["Longitude"].values[0],
            "Timestamp": metadata["Timestamp"].values[0],
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

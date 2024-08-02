# src/utils/merge_data.py

import os
import random

import pandas as pd
import pyarrow as pa

from src.config import settings


def processing():
    """
    Processes files in a directory structure to create a combined dataset of images and sounds.

    This function walks through a directory specified in the settings.LIFE_DIR configuration,
    identifies files by their type (image, sound, csv, unknown), and organizes them into a
    Pandas DataFrame. The function then groups the files by species, pairs images and sounds
    together, and saves the combined data to a Parquet file.

    Steps:
    1. Traverse the directory tree rooted at settings.LIFE_DIR.
    2. Identify file types based on their extensions.
    3. Create a DataFrame to index the files with columns for country, species, file type, and file path.
    4. Group the indexed files by species.
    5. For each species, pair images with sounds and create a combined DataFrame.
    6. Save the combined DataFrame to a Parquet file.

    Output:
    - A CSV file named "file_index.csv" containing the indexed files.
    - A Parquet file named "combined_data.parquet" containing paired images and sounds for each species.
    """
    records = []

    for root, _, files in os.walk(settings.LIFE_DIR):
        for file in files:
            file_path = os.path.join(root, file)

            country = root.split("/")[2]
            species = root.split("/")[3]
            file_type = (
                "image"
                if file.lower().endswith((".png", ".jpg", ".jpeg"))
                else "sound"
                if file.lower().endswith((".wav", ".mp3"))
                else "csv"
                if file.lower().endswith(".csv")
                else "unknown"
            )
            records.append(
                {
                    "country": country,
                    "species": species,
                    "file_type": file_type,
                    "file_path": file_path,
                }
            )

    df_files = pd.DataFrame(records)
    df_files.to_csv("file_index.csv", index=False)

    grouped = df_files.groupby("species")

    combined_records = []

    for species, group in grouped:
        images = group[group["file_type"] == "image"]["file_path"].tolist()
        sounds = group[group["file_type"] == "sound"]["file_path"].tolist()

        if images and sounds:
            max_len = max(len(images), len(sounds))

            paired_images = images * (max_len // len(images)) + random.sample(
                images, max_len % len(images)
            )
            paired_sounds = sounds * (max_len // len(sounds)) + random.sample(
                sounds, max_len % len(sounds)
            )

            for img, snd in zip(paired_images, paired_sounds):
                combined_records.append(
                    {"species": species, "image": img, "sound": snd}
                )

    df_combined = pd.DataFrame(combined_records)
    df_combined.to_parquet("combined_data.parquet", engine="pyarrow")

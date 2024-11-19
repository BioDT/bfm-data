# src/data_creation/create_species_dataset.py

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import xarray as xr
from dateutil import parser

from src.config import paths
from src.data_preprocessing.preprocessing import (
    preprocess_audio,
    preprocess_edna,
    preprocess_image,
    preprocess_text,
)
from src.dataset_creation.preprocessing import preprocess_and_normalize_species_data
from src.dataset_creation.save_data import save_as_parquet
from src.utils.merge_data import (
    extract_metadata_from_csv,
    find_closest_lat_lon,
    matching_image_audios,
)


def create_species_dataset(root_folder: str, filepath: str) -> pd.DataFrame:
    """
    This function creates a species dataset by walking through the folder structure, loading image, audio,
    and eDNA metadata, and matching the images with audios based on metadata such as latitude, longitude,
    and timestamp.

    Args:
        root_folder (str): The path to the root folder containing species subfolders, each with images, audios, and eDNA files.
        filepath (str): The path to the output file to save processed data.

    Returns:
        None
    """
    max_rows_per_species: int = 3
    rows = []

    for species_folder in os.listdir(root_folder):

        species_path = os.path.join(root_folder, species_folder)
        if not os.path.isdir(species_path):
            continue

        image_metadata_list = []
        audio_metadata_list = []
        edna_file_path = None
        description_file_path = None
        distribution_file_path = None
        taxonomic_data = {}

        for file_name in os.listdir(species_path):
            file_path = os.path.join(species_path, file_name)

            if file_name.endswith("_image.csv"):
                image_metadata = extract_metadata_from_csv(file_path, "image")
                image_metadata_list.append(image_metadata)
                if not edna_file_path and not taxonomic_data:
                    taxonomic_data = {
                        "Phylum": image_metadata.get("Phylum"),
                        "Class": image_metadata.get("Class"),
                        "Order": image_metadata.get("Order"),
                        "Family": image_metadata.get("Family"),
                        "Genus": image_metadata.get("Genus"),
                    }
            elif file_name.endswith("_audio.csv"):
                audio_metadata = extract_metadata_from_csv(file_path, "audio")
                audio_metadata_list.append(audio_metadata)
                if not edna_file_path and not taxonomic_data:
                    taxonomic_data = {
                        "Phylum": audio_metadata.get("Phylum"),
                        "Class": audio_metadata.get("Class"),
                        "Order": audio_metadata.get("Order"),
                        "Family": audio_metadata.get("Family"),
                        "Genus": audio_metadata.get("Genus"),
                    }
            elif file_name.endswith("_edna.csv"):
                edna_file_path = file_path
            elif file_name.endswith("_description.csv"):
                description_file_path = file_path
            elif file_name.endswith("_distribution.csv"):
                distribution_file_path = pd.read_csv(file_path)

        image_metadata_df = (
            pd.DataFrame(image_metadata_list) if image_metadata_list else None
        )
        audio_metadata_df = (
            pd.DataFrame(audio_metadata_list) if audio_metadata_list else None
        )
        distribution_data = (
            pd.read_csv(distribution_file_path)
            if isinstance(distribution_file_path, str)
            else pd.DataFrame()
        )

        if image_metadata_df is not None and audio_metadata_df is not None:
            matched_data = matching_image_audios(image_metadata_df, audio_metadata_df)
            if len(matched_data) > max_rows_per_species:
                matched_data = matched_data[:max_rows_per_species]
        elif image_metadata_df is not None:
            matched_data = image_metadata_df.to_dict("records")
        elif audio_metadata_df is not None:
            matched_data = audio_metadata_df.to_dict("records")
        else:
            matched_data = []

        edna_data = pd.read_csv(edna_file_path) if edna_file_path else pd.DataFrame()
        description_data = (
            pd.read_csv(description_file_path)
            if description_file_path
            else pd.DataFrame()
        )

        if matched_data:
            for match in matched_data:
                row = {
                    "Species": species_folder,
                    "Latitude": match.get("Latitude"),
                    "Longitude": match.get("Longitude"),
                    "Timestamp": match.get("Timestamp"),
                    "Image": None,
                    "Audio": None,
                    "eDNA": None,
                    "Description": None,
                    "Distribution": None,
                }

                if "Image_path" in match:
                    image_file = match["Image_path"]
                    image_features = preprocess_image(
                        image_file,
                        crop=False,
                        denoise=True,
                        blur=False,
                        augmentation=True,
                        normalize=True,
                    )
                    row["Image"] = image_features["normalised_image"]

                if "Audio_path" in match:
                    audio_file = match["Audio_path"]
                    audio_features = preprocess_audio(
                        audio_file,
                        rem_silence=False,
                        red_noise=True,
                        resample=True,
                        normalize=True,
                        mfcc=True,
                        convert_spectrogram=False,
                        convert_log_mel_spectrogram=False,
                    )
                    row["Audio"] = audio_features["mfcc_features"]

                all_ednas = []

                if not edna_data.empty:
                    for _, edna_row in edna_data.iterrows():
                        edna_sequence = edna_row["Nucleotides"]

                        edna_features = preprocess_edna(
                            edna_sequence,
                            clean=True,
                            replace_ambiguous=True,
                            threshold=0.5,
                            extract_kmer=True,
                            k=4,
                            one_hot_encode=False,
                            max_length=256,
                            vectorize_kmers=True,
                            normalise=True,
                        )

                        edna = edna_features.get("normalised_vector", None)
                        if edna is not None:
                            all_ednas.append(edna)
                    row["eDNA"] = (
                        torch.stack(all_ednas).mean(dim=0)
                        if all_ednas
                        else torch.zeros(256)
                    )

                if not edna_data.empty:
                    biological_features = {
                        "Phylum": edna_data["Phylum"].iloc[0],
                        "Class": edna_data["Class"].iloc[0],
                        "Order": edna_data["Order"].iloc[0],
                        "Family": edna_data["Family"].iloc[0],
                        "Genus": edna_data["Genus"].iloc[0],
                    }
                else:
                    biological_features = taxonomic_data

                row.update(biological_features)

                if (
                    not description_data.empty
                    and "description" in description_data.columns
                ):
                    description_features = preprocess_text(
                        description_data["description"].iloc[0],
                        clean=True,
                        use_bert=True,
                        max_length=512,
                    )
                    row["Description"] = description_features.get(
                        "bert_embeddings", [0.0] * 128
                    )
                    row["Redlist"] = description_data.get("redlist", [None])[0]
                else:
                    row["Description"] = [0.0] * 128
                    row["Redlist"] = None

                if not distribution_data.empty:
                    closest_dist = find_closest_lat_lon(
                        row["Latitude"], row["Longitude"], distribution_data
                    )
                    year = row["Timestamp"].year
                    if str(year) in closest_dist:
                        row["Distribution"] = closest_dist[str(year)]

                rows.append(row)

        elif not edna_data.empty:
            for _, edna_row in edna_data.iterrows():
                row = {
                    "Species": species_folder,
                    "Latitude": edna_row["Latitude"],
                    "Longitude": edna_row["Longitude"],
                    "Timestamp": edna_row["Timestamp"],
                    "eDNA": preprocess_edna(
                        edna_row["Nucleotides"],
                        clean=True,
                        replace_ambiguous=True,
                        threshold=0.5,
                        extract_kmer=True,
                        k=4,
                        one_hot_encode=False,
                        max_length=256,
                        vectorize_kmers=True,
                        normalise=True,
                    ).get("normalised_vector", torch.zeros(256)),
                    "Phylum": edna_row["Phylum"],
                    "Class": edna_row["Class"],
                    "Order": edna_row["Order"],
                    "Family": edna_row["Family"],
                    "Genus": edna_row["Genus"],
                    "Image": None,
                    "Audio": None,
                    "Description": None,
                    "Distribution": None,
                }

                if (
                    not description_data.empty
                    and "description" in description_data.columns
                ):
                    description_features = preprocess_text(
                        description_data["description"].iloc[0],
                        clean=True,
                        use_bert=True,
                        max_length=512,
                    )
                    row["Description"] = description_features.get(
                        "bert_embeddings", [0.0] * 128
                    )
                    row["Redlist"] = description_data.get("redlist", [None])[0]
                else:
                    row["Description"] = [0.0] * 128
                    row["Redlist"] = None

                if not distribution_data.empty:
                    closest_dist = find_closest_lat_lon(
                        row["Latitude"], row["Longitude"], distribution_data
                    )
                    year = row["Timestamp"].year
                    if str(year) in closest_dist:
                        row["Distribution"] = closest_dist[str(year)]

                print(
                    f"Description shape for row: {torch.tensor(row['Description']).shape}"
                )

                rows.append(row)

        if not distribution_data.empty:
            for _, dist_row in distribution_data.iterrows():
                for year in range(
                    int(dist_row.columns[3]), int(dist_row.columns[-1]) + 1
                ):
                    for month in range(1, 13):
                        monthly_row = {
                            "Species": species_folder,
                            "Latitude": dist_row["Latitude"],
                            "Longitude": dist_row["Longitude"],
                            "Timestamp": datetime(year, month, 1),
                            "Distribution": dist_row.get(str(year), None),
                            "Image": None,
                            "Audio": None,
                            "Description": None,
                            "Phylum": None,
                            "Class": None,
                            "Order": None,
                            "Family": None,
                            "Genus": None,
                            "Redlist": None,
                        }
                        rows.append(monthly_row)

    species_dataset = pd.DataFrame(rows)
    normalized_dataset = preprocess_and_normalize_species_data(species_dataset)

    save_as_parquet(normalized_dataset, filepath)


# def gather_all_years_in_folder(species_folder_path: str) -> List[int]:
#     """
#     Gathers all years from timestamps in CSV files within the specified folder.

#     Args:
#         species_folder_path (str): Path to the species folder.

#     Returns:
#         List[int]: A list of unique years found across all metadata files in the folder.
#     """
#     years = []

#     with os.scandir(species_folder_path) as it:
#         for entry in it:
#             if entry.is_file() and entry.name.endswith(".csv"):
#                 file_path = entry.path
#                 try:
#                     data = pd.read_csv(file_path)
#                     if 'Timestamp' in data.columns:
#                         data['Year'] = data['Timestamp'].apply(lambda x: extract_year(str(x)))
#                         years.extend(data['Year'].dropna().unique())
#                 except Exception as e:
#                     print(f"Error reading {file_path}: {e}")

#     return list(set(years))

# def extract_year(timestamp: str) -> int:
#     """
#     Attempts to parse a timestamp and extract the year.

#     Args:
#         timestamp (str): The timestamp to parse.

#     Returns:
#         int: The year extracted from the timestamp, or NaN if parsing fails.
#     """
#     try:

#         parsed_date = parser.parse(timestamp, fuzzy=True)
#         return parsed_date.year
#     except (ValueError, TypeError):
#         return None

# def is_folder_within_period(years: List[int], start_year: int, end_year: int) -> bool:
#     """
#     Checks if any year in the list falls within the specified period.

#     Args:
#         years (List[int]): List of years from a species folder.
#         start_year (int): Start year of the period.
#         end_year (int): End year of the period.

#     Returns:
#         bool: True if any year falls within the period; False otherwise.
#     """
#     for year in years:
#         if year and start_year <= year <= end_year:
#             return True
#     return False

# def save_incrementally(data_batch: pd.DataFrame, filepath: str) -> None:
#     """
#     Saves data incrementally to a single parquet file.

#     Args:
#         data_batch (pd.DataFrame): Batch of data to save.
#         filepath (str): The path to the output parquet file.
#     """
#     # Convert torch tensors if present
#     for column in data_batch.columns:
#         if data_batch[column].apply(lambda x: isinstance(x, torch.Tensor)).any():
#             data_batch[column] = data_batch[column].apply(lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)

#     if os.path.exists(filepath):
#         # If the file exists, load it and append the new data
#         existing_data = pd.read_parquet(filepath)
#         data_batch = pd.concat([existing_data, data_batch], ignore_index=True)

#     # Save data as a single parquet file (overwrite with the full data each time)
#     data_batch.to_parquet(filepath, index=False)


# def create_species_dataset(root_folder: str, filepath: str, start_year: int = 2000, end_year: int = 2020) -> None:
#     """
#     Creates a species dataset by walking through the folder structure, loading metadata, and saving incrementally.

#     Args:
#         root_folder (str): Path to the root folder containing species subfolders.
#         filepath (str): Path to the output parquet file.
#         start_year (int): The start year for time-based filtering.
#         end_year (int): The end year for time-based filtering.

#     Returns:
#         None
#     """
#     with ProcessPoolExecutor() as executor:
#         futures = []

#         for folder in os.listdir(root_folder):
#             species_folder_path = os.path.join(root_folder, folder)
#             if os.path.isdir(species_folder_path):
#                 timestamps = gather_all_years_in_folder(species_folder_path)
#                 if is_folder_within_period(timestamps, start_year, end_year):
#                     futures.append(executor.submit(process_species_folder, species_folder_path))

#         for future in as_completed(futures):
#             try:
#                 data_batch = future.result()
#                 if not data_batch.empty:
#                     save_incrementally(data_batch, filepath)
#             except Exception as e:
#                 print(f"Error processing folder: {e}")


# def process_species_folder(species_folder_path: str) -> pd.DataFrame:
#     """
#     Processes metadata and data files within a species folder, creating rows with combined information
#     from image, audio, eDNA, description, and distribution data.

#     Args:
#         species_folder_path (str): Path to a species folder.

#     Returns:
#         pd.DataFrame: DataFrame containing processed data rows for the species.
#     """
#     species_name = os.path.basename(species_folder_path)
#     rows = []
#     max_rows_per_species = 3

#     image_metadata_list, audio_metadata_list = [], []
#     edna_file_path, description_file_path, distribution_file_path = None, None, None
#     taxonomic_data = {}

#     with os.scandir(species_folder_path) as it:
#         for entry in it:
#             file_path = entry.path
#             if entry.name.endswith("_image.csv"):
#                 image_metadata = extract_metadata_from_csv(file_path, "image")
#                 image_metadata_list.append(image_metadata)
#                 taxonomic_data = taxonomic_data or extract_taxonomy(image_metadata)
#             elif entry.name.endswith("_audio.csv"):
#                 audio_metadata = extract_metadata_from_csv(file_path, "audio")
#                 audio_metadata_list.append(audio_metadata)
#                 taxonomic_data = taxonomic_data or extract_taxonomy(audio_metadata)
#             elif entry.name.endswith("_edna.csv"):
#                 edna_file_path = file_path
#             elif entry.name.endswith("_description.csv"):
#                 description_file_path = file_path
#             elif entry.name.endswith("_distribution.csv"):
#                 distribution_file_path = file_path

#     image_metadata_df = pd.DataFrame(image_metadata_list) if image_metadata_list else None
#     audio_metadata_df = pd.DataFrame(audio_metadata_list) if audio_metadata_list else None
#     distribution_data = pd.read_csv(distribution_file_path) if distribution_file_path else pd.DataFrame()

#     if image_metadata_df is not None and audio_metadata_df is not None:
#         matched_data = matching_image_audios(image_metadata_df, audio_metadata_df)
#         if len(matched_data) > max_rows_per_species:
#             matched_data = matched_data[:max_rows_per_species]
#     elif image_metadata_df is not None:
#         matched_data = image_metadata_df.to_dict("records")[:max_rows_per_species]
#     elif audio_metadata_df is not None:
#         matched_data = audio_metadata_df.to_dict("records")[:max_rows_per_species]
#     else:
#         matched_data = []

#     edna_data = pd.read_csv(edna_file_path) if edna_file_path else pd.DataFrame()
#     description_data = pd.read_csv(description_file_path) if description_file_path else pd.DataFrame()

#     for match in matched_data:
#         row = create_row_from_match(match, species_name, edna_data, description_data, distribution_data, taxonomic_data)
#         rows.append(row)

#     if not matched_data and not edna_data.empty:
#         for _, edna_row in edna_data.iterrows():
#             row = create_row_from_edna(edna_row, species_name, description_data, distribution_data)
#             rows.append(row)

#     if not distribution_data.empty:
#         rows.extend(create_distribution_rows(distribution_data, species_name))

#     species_dataset = pd.DataFrame(rows)

#     normalized_dataset = preprocess_and_normalize_species_data(species_dataset)

#     return normalized_dataset


# def create_row_from_match(match: Dict, species_name: str, edna_data: pd.DataFrame, description_data: pd.DataFrame,
#                           distribution_data: pd.DataFrame, taxonomic_data: Dict) -> Dict:
#     """
#     Creates a single row by combining matched image/audio data with additional information.

#     Args:
#         match (dict): Matched image/audio metadata.
#         species_name (str): Name of the species for this row.
#         edna_data (pd.DataFrame): DataFrame containing eDNA data.
#         description_data (pd.DataFrame): DataFrame containing description data.
#         distribution_data (pd.DataFrame): DataFrame containing distribution data.
#         taxonomic_data (dict): Dictionary containing taxonomic data.

#     Returns:
#         dict: A dictionary representing the row, with image, audio, eDNA, description, and distribution information.
#     """
#     row = {
#         "Species": species_name,
#         "Latitude": match.get("Latitude"),
#         "Longitude": match.get("Longitude"),
#         "Timestamp": match.get("Timestamp"),
#         "Image": None,
#         "Audio": None,
#         "eDNA": None,
#         "Description": None,
#         "Distribution": None,
#     }

#     if "Image_path" in match:
#         row["Image"] = preprocess_image(match["Image_path"], denoise=True, augmentation=True, normalize=True)["normalised_image"]
#     if "Audio_path" in match:
#         row["Audio"] = preprocess_audio(match["Audio_path"], red_noise=True, resample=True, normalize=True, mfcc=True)["mfcc_features"]

#     row["eDNA"] = preprocess_edna(edna_data)

#     row.update(extract_biological_features(edna_data, taxonomic_data))
#     row.update(extract_description_features(description_data))
#     row["Distribution"] = find_distribution(row, distribution_data)

#     return row


# def create_row_from_edna(edna_row: pd.Series, species_name: str, description_data: pd.DataFrame, distribution_data: pd.DataFrame) -> Dict:
#     """
#     Creates a single row of data using only eDNA information if no matched image or audio metadata is available.

#     Args:
#         edna_row (pd.Series): A single row from the eDNA DataFrame.
#         species_name (str): Name of the species for this row.
#         description_data (pd.DataFrame): DataFrame containing description data for the species.
#         distribution_data (pd.DataFrame): DataFrame containing distribution data for the species.

#     Returns:
#         Dict: A dictionary representing the row, containing eDNA features, taxonomic, description, and distribution information.
#     """
#     row = {
#         "Species": species_name,
#         "Latitude": edna_row["Latitude"],
#         "Longitude": edna_row["Longitude"],
#         "Timestamp": edna_row["Timestamp"],
#         "eDNA": preprocess_edna(
#             edna_row["Nucleotides"],
#             clean=True,
#             replace_ambiguous=True,
#             threshold=0.5,
#             extract_kmer=True,
#             k=4,
#             one_hot_encode=False,
#             max_length=256,
#             vectorize_kmers=True,
#             normalise=True
#         ).get("normalised_vector", torch.zeros(256)).tolist(),
#         "Image": None,
#         "Audio": None,
#         "Description": None,
#         "Distribution": None,
#     }

#     row.update(extract_biological_features(edna_data=pd.DataFrame([edna_row])))

#     row.update(extract_description_features(description_data))

#     row["Distribution"] = find_distribution(row, distribution_data)

#     return row

# def create_distribution_rows(distribution_data: pd.DataFrame, species_name: str) -> List[Dict]:
#     """
#     Creates monthly distribution rows from distribution data, covering each month for each year specified in the distribution.

#     Args:
#         distribution_data (pd.DataFrame): DataFrame containing distribution information for a species.
#         species_name (str): Name of the species for these distribution rows.

#     Returns:
#         List[Dict]: A list of dictionaries, each representing a row with monthly distribution data.
#     """
#     rows = []
#     for _, dist_row in distribution_data.iterrows():
#         for year in range(int(dist_row.columns[3]), int(dist_row.columns[-1]) + 1):
#             for month in range(1, 13):
#                 monthly_row = {
#                     "Species": species_name,
#                     "Latitude": dist_row["Latitude"],
#                     "Longitude": dist_row["Longitude"],
#                     "Timestamp": datetime(year, month, 1),
#                     "Distribution": dist_row.get(str(year)),
#                     "Image": None,
#                     "Audio": None,
#                     "Description": None,
#                     "Phylum": None,
#                     "Class": None,
#                     "Order": None,
#                     "Family": None,
#                     "Genus": None,
#                     "Redlist": None,
#                 }
#                 rows.append(monthly_row)
#     return rows


# def extract_description_features(description_data: pd.DataFrame) -> Dict:
#     """
#     Extracts text-based features from the description data, including BERT embeddings
#     and Redlist status if available.

#     Args:
#         description_data (pd.DataFrame): DataFrame containing species description information.

#     Returns:
#         Dict: Dictionary containing description embeddings and Redlist status.
#               If unavailable, returns a default embedding and None for Redlist.
#     """
#     if not description_data.empty and "description" in description_data.columns:
#         description_features = preprocess_text(
#             description_data["description"].iloc[0],
#             clean=True,
#             use_bert=True,
#             max_length=512
#         )
#         return {
#             "Description": description_features.get("bert_embeddings", [0.0] * 128),
#             "Redlist": description_data.get("redlist", [None])[0]
#         }
#     return {"Description": [0.0] * 128, "Redlist": None}


# def extract_biological_features(edna_data: pd.DataFrame, taxonomic_data: Dict = None) -> Dict:
#     """
#     Extracts taxonomic features from eDNA data if available, providing a fallback to
#     the existing taxonomic data if eDNA is unavailable.

#     Args:
#         edna_data (pd.DataFrame): DataFrame containing eDNA information, including taxonomic data.
#         taxonomic_data (Dict, optional): Fallback dictionary of taxonomic data if eDNA is missing.

#     Returns:
#         Dict: Dictionary containing the taxonomic information (Phylum, Class, Order, Family, Genus).
#               If unavailable, falls back to `taxonomic_data` or returns an empty dictionary.
#     """
#     if not edna_data.empty:
#         return {
#             "Phylum": edna_data["Phylum"].iloc[0],
#             "Class": edna_data["Class"].iloc[0],
#             "Order": edna_data["Order"].iloc[0],
#             "Family": edna_data["Family"].iloc[0],
#             "Genus": edna_data["Genus"].iloc[0],
#         }
#     return taxonomic_data or {}


# def find_distribution(row: Dict, distribution_data: pd.DataFrame) -> Optional[float]:
#     """
#     Finds the closest matching distribution value based on latitude, longitude,
#     and year from the timestamp in the row data.

#     Args:
#         row (Dict): Dictionary containing the latitude, longitude, and timestamp of the record.
#         distribution_data (pd.DataFrame): DataFrame containing distribution data.

#     Returns:
#         Optional[float]: The distribution value for the closest match in terms of location
#                          and year, or None if no matching distribution is found.
#     """
#     if distribution_data.empty:
#         return None
#     closest_dist = find_closest_lat_lon(row["Latitude"], row["Longitude"], distribution_data)
#     year = row["Timestamp"].year
#     return closest_dist.get(str(year))


# def extract_taxonomy(metadata: Dict) -> Dict:
#     """
#     Extracts taxonomic information from metadata if available.

#     Args:
#         metadata (Dict): Metadata dictionary from an image or audio file.

#     Returns:
#         Dict: A dictionary containing taxonomic information, if present.
#     """
#     taxonomy_fields = ["Phylum", "Class", "Order", "Family", "Genus"]

#     # Ensure that `metadata` is not a DataFrame when calling `.get()`
#     if isinstance(metadata, pd.DataFrame):
#         metadata = metadata.to_dict(orient='records')[0]  # Convert to dictionary

#     return {field: metadata.get(field) for field in taxonomy_fields if field in metadata}

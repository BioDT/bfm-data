"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from dateutil import parser

from src.data_preprocessing.preprocessing import (
    preprocess_audio,
    preprocess_edna,
    preprocess_image,
    preprocess_text,
)
from src.dataset_creation.preprocessing import preprocess_and_normalize_species_data
from src.dataset_creation.save_data import save_incrementally
from src.utils.merge_data import (
    extract_metadata_from_csv,
    find_closest_lat_lon,
    matching_image_audios,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def gather_all_years_in_folder(
    species_folder_path: str, start_year: int, end_year: int
) -> List[int]:
    """
    Gathers all years from timestamps in CSV files within the specified folder
    and filters rows to include only timestamps within the specified range.

    Handles special cases like `distribution.csv`, where years are represented as column headers.

    Args:
        species_folder_path (str): Path to the species folder.
        start_year (int): Start year of the range.
        end_year (int): End year of the range.

    Returns:
        List[int]: A list of unique years found within the range in the folder.
    """
    years = []

    with os.scandir(species_folder_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".csv"):
                file_path = entry.path
                try:
                    if entry.name.endswith("_distribution.csv"):
                        data = pd.read_csv(file_path)
                        distribution_years = [
                            int(year)
                            for year in data.columns
                            if year.isdigit() and start_year <= int(year) <= end_year
                        ]
                        years.extend(distribution_years)
                        logging.info(
                            f"Extracted years from distribution file {file_path}: {distribution_years}"
                        )
                    else:
                        data = pd.read_csv(file_path)
                        if "Timestamp" in data.columns:
                            data["Year"] = data["Timestamp"].apply(
                                lambda x: extract_year(x, file_path=file_path)
                            )
                            filtered_data = data[
                                (data["Year"] >= start_year)
                                & (data["Year"] <= end_year)
                            ]
                            years.extend(
                                filtered_data["Year"].dropna().astype(int).unique()
                            )
                except Exception as e:
                    logging.warning(f"Error reading {file_path}: {e}")

    return list(set(years))


def extract_year(timestamp: str, file_path: Optional[str] = None) -> Optional[int]:
    """
    Attempts to parse a timestamp and extract the year. Logs file and folder information on failure.

    Args:
        timestamp (str): The timestamp to parse.
        file_path (str, optional): Path to the file being processed.

    Returns:
        int: The extracted year, or None if parsing fails.
    """
    try:
        parsed_date = parser.parse(timestamp, fuzzy=True)
        return parsed_date.year
    except (ValueError, TypeError):
        log_message = f"Invalid timestamp format: '{timestamp}'. Skipping."
        if file_path:
            log_message += f" File: {file_path}"
        logging.warning(log_message)
        return None


def is_folder_within_period(years: List[int], start_year: int, end_year: int) -> bool:
    """
    Checks if any year in the list falls within the specified period.

    Args:
        years (List[int]): List of years from a species folder.
        start_year (int): Start year of the period.
        end_year (int): End year of the period.

    Returns:
        bool: True if any year falls within the period; False otherwise.
    """
    return any(start_year <= year <= end_year for year in years if year)


def create_species_dataset(
    root_folder: str, filepath: str, start_year: int = 2000, end_year: int = 2020
) -> None:
    """
    Creates a species dataset by processing each folder individually, checking the years
    and saving data incrementally.

    Args:
        root_folder (str): Path to the root folder containing species subfolders.
        filepath (str): Path to the output parquet file.
        start_year (int): The start year for time-based filtering.
        end_year (int): The end year for time-based filtering.

    Returns:
        None
    """
    for folder in os.listdir(root_folder):
        species_folder_path = os.path.join(root_folder, folder)

        if not os.path.isdir(species_folder_path):
            continue

        years = gather_all_years_in_folder(species_folder_path, start_year, end_year)
        if not years:
            logging.info(f"No valid years found for folder: {folder}")
            continue

        if is_folder_within_period(years, start_year, end_year):
            logging.info(f"Processing folder {folder} with years: {years}")
            try:
                data_batch = process_species_folder(
                    species_folder_path, start_year, end_year
                )
                if not data_batch.empty:
                    save_incrementally(data_batch, filepath)
            except Exception as e:
                logging.error(f"Error processing folder {species_folder_path}: {e}")


def process_species_folder(
    species_folder_path: str, start_year: int, end_year: int
) -> pd.DataFrame:
    """
    Processes metadata and data files within a species folder, creating rows with combined information
    from image, audio, eDNA, description, and distribution data, filtered by year range.

    Args:
        species_folder_path (str): Path to a species folder.
        start_year (int): Start year for filtering.
        end_year (int): End year for filtering.

    Returns:
        pd.DataFrame: DataFrame containing processed data rows for the species.
    """
    species_name = os.path.basename(species_folder_path)
    rows = []
    max_rows_per_species = 3

    image_metadata_list, audio_metadata_list = [], []
    edna_file_path, description_file_path, distribution_file_path = None, None, None
    taxonomic_data = {}

    with os.scandir(species_folder_path) as it:
        for entry in it:
            file_path = entry.path
            try:
                if entry.name.endswith("_image.csv"):
                    metadata = extract_metadata_from_csv(entry.path, "image")
                    if isinstance(metadata, dict):
                        metadata = pd.DataFrame([metadata])
                    if "Timestamp" in metadata.columns:
                        metadata["Year"] = metadata["Timestamp"].apply(
                            lambda x: extract_year(str(x))
                        )
                        if metadata["Year"].isnull().any():
                            logging.warning(
                                f"Invalid or missing year in 'Timestamp' column for file {file_path}. "
                                f"Problematic rows: {metadata[metadata['Year'].isnull()]}"
                            )
                        metadata = metadata[
                            (metadata["Year"] >= start_year)
                            & (metadata["Year"] <= end_year)
                        ]
                    metadata["File_path"] = file_path
                    image_metadata_list.append(metadata)
                    taxonomic_data = taxonomic_data or extract_taxonomy(metadata)

                elif entry.name.endswith("_audio.csv"):
                    metadata = extract_metadata_from_csv(file_path, "audio")
                    if isinstance(metadata, dict):
                        metadata = pd.DataFrame([metadata])
                    if "Timestamp" in metadata.columns:
                        metadata["Year"] = metadata["Timestamp"].apply(
                            lambda x: extract_year(str(x))
                        )
                        if metadata["Year"].isnull().any():
                            logging.warning(
                                f"Invalid or missing year in 'Timestamp' column for file {file_path}. "
                                f"Problematic rows: {metadata[metadata['Year'].isnull()]}"
                            )

                        metadata = metadata[
                            (metadata["Year"] >= start_year)
                            & (metadata["Year"] <= end_year)
                        ]
                    metadata["File_path"] = file_path
                    audio_metadata_list.append(metadata)
                    taxonomic_data = taxonomic_data or extract_taxonomy(metadata)

                elif entry.name.endswith("_edna.csv"):
                    edna_file_path = file_path

                elif entry.name.endswith("_description.csv"):
                    description_file_path = file_path

                elif entry.name.endswith("_distribution.csv"):
                    distribution_file_path = file_path

            except IndexError as e:
                logging.warning(
                    f"IndexError reading file {file_path}: {e}. The year should be out of range."
                )
            except pd.errors.EmptyDataError as e:
                logging.warning(
                    f"EmptyDataError: File {file_path} is empty or corrupted: {e}"
                )
            except Exception as e:
                logging.warning(f"Error reading file {file_path}: {e}")

    if image_metadata_list:
        try:
            image_metadata_df = pd.concat(image_metadata_list, ignore_index=True)
        except Exception as e:
            print(f"Error concatenating image metadata: {e}")
            for i, metadata in enumerate(image_metadata_list):
                print(
                    f"Metadata {i}: Type={type(metadata)}, Shape={getattr(metadata, 'shape', None)}"
                )
            raise
    else:
        image_metadata_df = None

    if audio_metadata_list:
        try:
            audio_metadata_df = pd.concat(audio_metadata_list, ignore_index=True)
        except Exception as e:
            print(f"Error concatenating image metadata: {e}")
            for i, metadata in enumerate(audio_metadata_list):
                print(
                    f"Metadata {i}: Type={type(metadata)}, Shape={getattr(metadata, 'shape', None)}"
                )
            raise
    else:
        audio_metadata_df = None

    distribution_data = (
        pd.read_csv(distribution_file_path)
        if distribution_file_path
        else pd.DataFrame()
    )

    if not distribution_data.empty:
        year_columns = [
            col
            for col in distribution_data.columns
            if col.isdigit() and start_year <= int(col) <= end_year
        ]

        if year_columns:
            distribution_data = distribution_data[
                ["Binomial", "Latitude", "Longitude"] + year_columns
            ]
            logging.info(f"Filtered distribution data to years {year_columns}.")
        else:
            logging.warning(
                "No valid year columns found in the distribution data within the specified range."
            )
            distribution_data = pd.DataFrame()

    if image_metadata_df is not None and audio_metadata_df is not None:
        matched_data = matching_image_audios(image_metadata_df, audio_metadata_df)
        if len(matched_data) > max_rows_per_species:
            matched_data = matched_data[:max_rows_per_species]
    elif image_metadata_df is not None:
        matched_data = image_metadata_df.to_dict("records")[:max_rows_per_species]
    elif audio_metadata_df is not None:
        matched_data = audio_metadata_df.to_dict("records")[:max_rows_per_species]
    else:
        matched_data = []

    edna_data = pd.read_csv(edna_file_path) if edna_file_path else pd.DataFrame()
    description_data = (
        pd.read_csv(description_file_path) if description_file_path else pd.DataFrame()
    )

    for match in matched_data:
        row = create_row_from_match(
            match, species_name, edna_data, description_data, taxonomic_data
        )
        row["File_path"] = match.get("File_path", "")
        rows.append(row)

    if not matched_data and not edna_data.empty:
        for _, edna_row in edna_data.iterrows():
            row = create_row_from_edna(edna_row, species_name, description_data)
            row["File_path"] = edna_file_path
            rows.append(row)

    if not distribution_data.empty and not matched_data and edna_data.empty:
        rows.extend(
            create_distribution_rows(distribution_data, species_name, description_data)
        )
    elif not distribution_data.empty and not matched_data and not edna_data.empty:
        rows.extend(
            create_distribution_rows(
                distribution_data, species_name, description_data, edna_data.iloc[0]
            )
        )
    elif not distribution_data.empty and matched_data:
        rows.extend(
            create_distribution_rows(
                distribution_data, species_name, description_data, taxonomic_data
            )
        )

    species_dataset = pd.DataFrame(rows)

    return preprocess_and_normalize_species_data(species_dataset)


def create_row_from_match(
    match: Dict,
    species_name: str,
    edna_data: pd.DataFrame,
    description_data: pd.DataFrame,
    taxonomic_data: Dict,
) -> Dict:
    """
    Creates a single row by combining matched image/audio data with additional information.

    Args:
        match (dict): Matched image/audio metadata.
        species_name (str): Name of the species for this row.
        edna_data (pd.DataFrame): DataFrame containing eDNA data.
        description_data (pd.DataFrame): DataFrame containing description data.
        taxonomic_data (dict): Dictionary containing taxonomic data.

    Returns:
        dict: A dictionary representing the row, with image, audio, eDNA, description, and distribution information.
    """
    row = {
        "Species": species_name,
        "Latitude": match.get("Latitude"),
        "Longitude": match.get("Longitude"),
        "Timestamp": match.get("Timestamp"),
        "Image": None,
        "Audio": None,
        "eDNA": None,
        "Description": None,
        "Distribution": 1,
        "Redlist": None,
    }

    if "Image_path" in match:
        row["Image"] = preprocess_image(
            match["Image_path"], denoise=True, augmentation=True, normalize=True
        )["normalised_image"]
    if "Audio_path" in match:
        row["Audio"] = preprocess_audio(
            match["Audio_path"],
            red_noise=True,
            resample=True,
            normalize=True,
            mfcc=True,
        )["mfcc_features"]

    if not edna_data.empty:
        row["eDNA"] = preprocess_edna(
            edna_data.iloc[0]["Nucleotides"],
            clean=True,
            replace_ambiguous=True,
            threshold=0.5,
            extract_kmer=True,
            k=4,
            one_hot_encode=False,
            max_length=256,
            vectorize_kmers=True,
            normalise=True,
        ).get("normalised_vector", None)

    row.update(extract_biological_features(edna_data, taxonomic_data))

    if not description_data.empty:
        row.update(extract_description_features(description_data))

    return row


def create_row_from_edna(
    edna_row: pd.Series, species_name: str, description_data: pd.DataFrame
) -> Dict:
    """
    Creates a single row of data using only eDNA information if no matched image or audio metadata is available.

    Args:
        edna_row (pd.Series): A single row from the eDNA DataFrame.
        species_name (str): Name of the species for this row.
        description_data (pd.DataFrame): DataFrame containing description data for the species.

    Returns:
        Dict: A dictionary representing the row, containing eDNA features, taxonomic, description, and distribution information.
    """
    edna_vector = None
    if pd.notna(edna_row["Nucleotides"]) and isinstance(edna_row["Nucleotides"], str):
        edna_vector = preprocess_edna(
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
        ).get("normalised_vector", None)

    row = {
        "Species": species_name,
        "Latitude": edna_row["Latitude"],
        "Longitude": edna_row["Longitude"],
        "Timestamp": edna_row["Timestamp"],
        "eDNA": edna_vector,
        "Image": None,
        "Audio": None,
        "Description": None,
        "Distribution": 1,
        "Redlist": None,
    }

    row.update(extract_biological_features(edna_data=pd.DataFrame([edna_row])))

    if not description_data.empty:
        row.update(extract_description_features(description_data))

    return row


def create_distribution_rows(
    distribution_data: pd.DataFrame,
    species_name: str,
    description_data: pd.DataFrame,
    extra_data=None,
) -> List[Dict]:
    """
    Creates monthly distribution rows from distribution data, covering only the years within the specified range.

    Args:
        distribution_data (pd.DataFrame): DataFrame containing distribution information for a species.
        species_name (str): Name of the species for these distribution rows.
        description_data (pd.DataFrame): DataFrame containing description data for the species.

    Returns:
        List[Dict]: A list of dictionaries, each representing a row with monthly distribution data.
    """
    rows = []
    valid_years = [col for col in distribution_data.columns[3:] if col.isdigit()]

    if not valid_years:
        logging.warning(
            f"No valid distribution years found for species: {species_name}"
        )
        return rows

    for _, dist_row in distribution_data.iterrows():
        for year in valid_years:
            distribution = dist_row.get(year)
            if pd.isna(distribution):
                continue
            row = {
                "Species": species_name,
                "Latitude": dist_row["Latitude"],
                "Longitude": dist_row["Longitude"],
                "Timestamp": datetime(int(year), 1, 1),
                "Distribution": distribution,
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

            if extra_data is not None and "Nucleotides" in extra_data:
                row.update(
                    extract_biological_features(
                        edna_data=pd.DataFrame([extra_data]), extra_data=extra_data
                    )
                )
            elif extra_data is not None and "Nucleotides" not in extra_data:
                row.update(
                    extract_biological_features(
                        edna_data=pd.DataFrame([extra_data]),
                        taxonomic_data=extra_data,
                        extra_data=extra_data,
                    )
                )
            if not description_data.empty:
                row.update(extract_description_features(description_data))

            rows.append(row)

    return rows


def extract_description_features(description_data: pd.DataFrame) -> Dict:
    """
    Extracts text-based features from the description data, including BERT embeddings
    and Redlist status if available. If placeholders like 'No description available'
    or 'Not available' are found, it will return None.

    Args:
        description_data (pd.DataFrame): DataFrame containing species description information.

    Returns:
        Dict: Dictionary containing description embeddings and Redlist status.
              If unavailable, returns None for both fields.
    """
    if not description_data.empty and "description" in description_data.columns:
        description_text = description_data["description"].iloc[0]
        redlist_status = (
            description_data["redlist"].iloc[0]
            if "redlist" in description_data.columns
            else None
        )

        if description_text in ["No description available", "Not available"]:
            description_features = None
        else:
            description_features = preprocess_text(
                description_text, clean=True, use_bert=True, max_length=512
            ).get("bert_embeddings", None)

        if redlist_status in ["Not available", "Unknown"]:
            redlist_status = None

        return {"Description": description_features, "Redlist": redlist_status}

    return {"Description": None, "Redlist": None}


def extract_biological_features(
    edna_data: pd.DataFrame, taxonomic_data: Dict = None, extra_data=None
) -> Dict:
    """
    Extracts taxonomic features from eDNA data if available, providing a fallback to
    the existing taxonomic data if eDNA is unavailable.

    Args:
        edna_data (pd.DataFrame): DataFrame containing eDNA information, including taxonomic data.
        taxonomic_data (Dict, optional): Fallback dictionary of taxonomic data if eDNA is missing.

    Returns:
        Dict: Dictionary containing the taxonomic information (Phylum, Class, Order, Family, Genus).
              If unavailable, falls back to `taxonomic_data` or returns an empty dictionary.
    """
    if not edna_data.empty and extra_data is None:
        return {
            "Phylum": edna_data["Phylum"].iloc[0],
            "Class": edna_data["Class"].iloc[0],
            "Order": edna_data["Order"].iloc[0],
            "Family": edna_data["Family"].iloc[0],
            "Genus": edna_data["Genus"].iloc[0],
        }
    elif not edna_data.empty and extra_data is not None and "Nucleotides" in extra_data:
        if pd.notna(extra_data["Nucleotides"]) and isinstance(
            extra_data["Nucleotides"], str
        ):
            edna_vector = preprocess_edna(
                extra_data["Nucleotides"],
                clean=True,
                replace_ambiguous=True,
                threshold=0.5,
                extract_kmer=True,
                k=4,
                one_hot_encode=False,
                max_length=256,
                vectorize_kmers=True,
                normalise=True,
            ).get("normalised_vector", None)
        return {
            "Phylum": edna_data["Phylum"].iloc[0],
            "Class": edna_data["Class"].iloc[0],
            "Order": edna_data["Order"].iloc[0],
            "Family": edna_data["Family"].iloc[0],
            "Genus": edna_data["Genus"].iloc[0],
            "eDNA": edna_vector,
        }
    elif taxonomic_data == extra_data:
        return {
            "Phylum": edna_data["Phylum"].iloc[0],
            "Class": edna_data["Class"].iloc[0],
            "Order": edna_data["Order"].iloc[0],
            "Family": edna_data["Family"].iloc[0],
            "Genus": edna_data["Genus"].iloc[0],
        }

    return taxonomic_data or {}


def find_distribution(row: Dict, distribution_data: pd.DataFrame) -> Optional[float]:
    """
    Finds the closest matching distribution value based on latitude, longitude,
    and year from the timestamp in the row data.

    Args:
        row (Dict): Dictionary containing the latitude, longitude, and timestamp of the record.
        distribution_data (pd.DataFrame): DataFrame containing distribution data.

    Returns:
        Optional[float]: The distribution value for the closest match in terms of location
                         and year, or None if no matching distribution is found.
    """
    if distribution_data.empty:
        return None

    if isinstance(row["Timestamp"], str):
        try:
            row["Timestamp"] = pd.to_datetime(row["Timestamp"])
        except ValueError:
            logging.error(f"Invalid timestamp format: {row['Timestamp']}")
            return None

    closest_dist = find_closest_lat_lon(
        row["Latitude"], row["Longitude"], distribution_data
    )

    year = row["Timestamp"].year
    return closest_dist.get(str(year))


def extract_taxonomy(metadata: Dict) -> Dict:
    """
    Extracts taxonomic information from metadata if available.

    Args:
        metadata (Dict): Metadata dictionary from an image or audio file.

    Returns:
        Dict: A dictionary containing taxonomic information, if present.
    """
    taxonomy_fields = ["Phylum", "Class", "Order", "Family", "Genus"]

    if isinstance(metadata, pd.DataFrame):
        metadata = metadata.to_dict(orient="records")[0]

    return {
        field: metadata.get(field) for field in taxonomy_fields if field in metadata
    }


def debug_distribution_matching(
    parquet_file: str, distribution_file: str, folder_path: str, species_id: int
):
    """
    Debugs the matching logic between the Parquet file and the distribution file.
    Incorporates steps to inspect data directly, check precision issues, and verify timestamp formats.

    Args:
        parquet_file (str): Path to the existing Parquet file.
        distribution_file (str): Path to the distribution file.
        folder_path (str): Path to the specific folder.
        species_id (int): The species ID associated with the folder.

    Returns:
        None
    """

    try:
        species_data = pd.read_parquet(parquet_file)
        print(f"Loaded Parquet file with {len(species_data)} rows.")
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return

    try:
        dist_data = pd.read_csv(distribution_file)
        print(f"Loaded distribution file with {len(dist_data)} rows.")
    except Exception as e:
        print(f"Error loading distribution file: {e}")
        return

    species_data["Latitude"] = species_data["Latitude"].round(2)
    species_data["Longitude"] = species_data["Longitude"].round(2)
    dist_data["Latitude"] = dist_data["Latitude"].round(2)
    dist_data["Longitude"] = dist_data["Longitude"].round(2)

    species_data["Timestamp"] = pd.to_datetime(
        species_data["Timestamp"], errors="coerce"
    )

    year_columns = [col for col in dist_data.columns[3:] if col.isdigit()]

    updated_rows = 0
    for _, dist_row in dist_data.iterrows():
        latitude, longitude = dist_row["Latitude"], dist_row["Longitude"]
        for year in year_columns:
            distribution_value = dist_row.get(year)
            if pd.isna(distribution_value):
                continue

            timestamp = datetime(int(year), 1, 1)
            mask = (
                (species_data["Species"] == species_id)
                & (species_data["Latitude"] == latitude)
                & (species_data["Longitude"] == longitude)
                & (species_data["Timestamp"] == pd.Timestamp(timestamp))
            )
            match_count = mask.sum()
            if match_count > 0:
                print(
                    f"Match found: Latitude={latitude}, Longitude={longitude}, Year={year}"
                )
                species_data.loc[mask, "Distribution"] = float(distribution_value)
                updated_rows += match_count
            else:
                print(
                    f"No match: Latitude={latitude}, Longitude={longitude}, Year={year}"
                )

    print(f"\nUpdated {updated_rows} rows in the dataset.")

    species_data["Timestamp"] = species_data["Timestamp"].apply(
        lambda ts: ts.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(ts) else None
    )

    try:
        species_data.to_parquet(parquet_file, index=False, engine="pyarrow")
        print(f"Updated Parquet file saved to: {parquet_file}")
    except Exception as e:
        print(f"Error saving updated Parquet file: {e}")


import json


def process_all_folders_with_distribution(
    parquet_file: str, root_folder: str, label_mapping_file: str
):
    """
    Updates the 'Distribution' column in the Parquet file for all species folders containing a distribution file.
    Includes species ID matching based on folder name and label mapping.

    Args:
        parquet_file (str): Path to the existing Parquet file.
        root_folder (str): Path to the root directory containing species folders.
        label_mapping_file (str): Path to the JSON file containing label mappings.

    Returns:
        None
    """
    with open(label_mapping_file, "r") as f:
        label_mapping = json.load(f)

    folders = [
        f
        for f in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, f))
    ]
    total_folders = len(folders)
    print(f"Found {total_folders} folders in: {root_folder}")

    processed_count = 0
    for folder in folders:
        folder_path = os.path.join(root_folder, folder)
        distribution_file = os.path.join(folder_path, f"{folder}_distribution.csv")

        if not os.path.exists(distribution_file):
            print(f"Skipping folder without distribution file: {folder_path}")
            continue

        species_id = label_mapping["Species"].get(folder)
        if species_id is None:
            print(f"Skipping folder with unknown species name: {folder}")
            continue

        print(
            f"Processing folder {processed_count + 1}/{total_folders}: {folder_path} with species_id={species_id}"
        )
        try:
            debug_distribution_matching(
                parquet_file, distribution_file, folder_path, species_id
            )
            processed_count += 1
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")

    print(f"Processing completed: {processed_count}/{total_folders} folders processed.")

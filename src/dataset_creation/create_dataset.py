# src/data_creation/create_dataset.py

import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.config.paths import BATCHES_DATA_DIR
from src.data_preprocessing.preprocessing import (
    preprocess_audio,
    preprocess_edna,
    preprocess_image,
    preprocess_text,
)
from src.data_preprocessing.transformation.era5 import get_mean_standard_deviation
from src.dataset_creation.batch import DataBatch
from src.dataset_creation.load_data import (
    load_era5_datasets,
    load_era5_files_grouped_by_date,
    load_species_data,
)
from src.dataset_creation.metadata import BatchMetadata
from src.dataset_creation.preprocessing import (
    crop_lat_lon,
    initialize_climate_tensors,
    initialize_species_tensors,
    merge_timestamps,
    preprocess_and_normalize_species_data,
    preprocess_era5,
    rescale_sort_lat_lon,
    reset_climate_tensors,
    reset_species_tensors,
)
from src.dataset_creation.save_data import (
    save_as_parquet,
    save_batch_metadata_to_parquet,
)
from src.utils.merge_data import (
    extract_metadata_from_csv,
    folder_has_all_modalities,
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
    max_rows_per_species: int = 40
    rows = []

    for species_folder in os.listdir(root_folder):

        species_path = os.path.join(root_folder, species_folder)
        if not os.path.isdir(species_path):
            continue

        if not folder_has_all_modalities(species_path):
            continue

        image_metadata_list = []
        audio_metadata_list = []
        edna_file_path = None
        description_file_path = None

        for file_name in os.listdir(species_path):
            file_path = os.path.join(species_path, file_name)

            if file_name.endswith("_image.csv"):
                image_metadata = extract_metadata_from_csv(file_path, "image")
                image_metadata_list.append(image_metadata)
            elif file_name.endswith("_audio.csv"):
                audio_metadata = extract_metadata_from_csv(file_path, "audio")
                audio_metadata_list.append(audio_metadata)
            elif file_name.endswith("_edna.csv"):
                edna_file_path = file_path
            elif file_name.endswith("_description.csv"):
                description_file_path = file_path

        image_metadata_df = pd.DataFrame(image_metadata_list)
        audio_metadata_df = pd.DataFrame(audio_metadata_list)

        matched_data = matching_image_audios(image_metadata_df, audio_metadata_df)

        if len(matched_data) > max_rows_per_species:
            matched_data = matched_data[:max_rows_per_species]

        edna_data = pd.read_csv(edna_file_path) if edna_file_path else pd.DataFrame()
        description_data = (
            pd.read_csv(description_file_path)
            if description_file_path
            else pd.DataFrame()
        )

        for match in matched_data:

            image_file = match["Image_path"]
            audio_file = match["Audio_path"]

            image_features = preprocess_image(
                image_file,
                crop=False,
                denoise=True,
                blur=False,
                augmentation=True,
                normalize=True,
            )

            image = image_features["normalised_image"]

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

            mfcc_features = audio_features["mfcc_features"]

            all_ednas = []

            if not edna_data.empty:
                for i, edna_row in edna_data.iterrows():
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

            if all_ednas:
                combined_ednas = torch.stack(all_ednas).mean(dim=0)
            else:
                combined_ednas = torch.zeros(256)

            if not edna_data.empty:
                biological_features = {
                    "Phylum": edna_data["Phylum"].iloc[0],
                    "Class": edna_data["Class"].iloc[0],
                    "Order": edna_data["Order"].iloc[0],
                    "Family": edna_data["Family"].iloc[0],
                    "Genus": edna_data["Genus"].iloc[0],
                }
            else:
                biological_features = {
                    key: None for key in ["Phylum", "Class", "Order", "Family", "Genus"]
                }

            if not description_data.empty:
                description_features = (
                    preprocess_text(
                        description_data["description"].iloc[0],
                        clean=True,
                        use_bert=True,
                        max_length=512,
                    )
                    if "description" in description_data.columns
                    else None
                )
                bert_embeddings = (
                    description_features["bert_embeddings"]
                    if description_features is not None
                    else [0.0] * 128
                )
                redlist = (
                    description_data["redlist"].iloc[0]
                    if "redlist" in description_data.columns
                    else None
                )
            else:
                bert_embeddings = [0.0] * 128
                redlist = None

            row = {
                "Species": species_folder,
                "Image": image,
                "Audio": mfcc_features,
                "Latitude": match["Latitude"],
                "Longitude": match["Longitude"],
                "Timestamp": match["Timestamp"],
                "eDNA": combined_ednas,
                "Phylum": biological_features["Phylum"],
                "Class": biological_features["Class"],
                "Order": biological_features["Order"],
                "Family": biological_features["Family"],
                "Genus": biological_features["Genus"],
                "Description": bert_embeddings,
                "Redlist": redlist if redlist is not None else None,
            }

            rows.append(row)

    species_dataset = pd.DataFrame(rows)
    normalized_dataset = preprocess_and_normalize_species_data(species_dataset)

    save_as_parquet(normalized_dataset, filepath)


def create_batch(
    dates: list,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    surface_dataset: xr.Dataset,
    single_dataset: xr.Dataset,
    atmospheric_dataset: xr.Dataset,
    species_dataset: pd.DataFrame,
    surfaces_variables: dict,
    single_variables: dict,
    atmospheric_variables: dict,
    species_variables: dict,
) -> DataBatch:
    """
    Create a DataBatch for a specific day by merging climate and species data but by giving two timestamps
    for prediction purposes.

    Args:
        dates (list): List or tuple of two timestamps.
        lat_range (np.ndarray): Array of latitude values.
        lon_range (np.ndarray): Array of longitude values.
        surface_dataset (xarray.Dataset): Surface variables dataset.
        single_dataset (xarray.Dataset): Single-level variables dataset.
        atmospheric_dataset (xarray.Dataset): Atmospheric pressure-level dataset.
        species_dataset (pd.DataFrame): Species data containing multimodal features.
        surfaces_variables (dict): Pre-initialized surface climate variables tensors.
        single_variables (dict): Pre-initialized single-level climate variables tensors.
        atmospheric_variables (dict): Pre-initialized atmospheric pressure variables tensors.
        species_variables (dict): Pre-initialized species variables tensors.


    Returns:
        DataBatch: A DataBatch object containing both climate and species data for the given day.
    """

    locations, scales = get_mean_standard_deviation(
        surface_dataset, single_dataset, atmospheric_dataset
    )

    pressure_levels = tuple(
        int(level) for level in atmospheric_dataset.pressure_level.values
    )

    for t, current_date in enumerate(dates):

        try:
            surface_variables_by_day = surface_dataset.sel(
                valid_time=current_date[0], method="nearest"
            ).load()
            single_variables_by_day = single_dataset.sel(
                valid_time=current_date[0], method="nearest"
            ).load()
            atmospheric_variables_by_day = atmospheric_dataset.sel(
                valid_time=current_date[0], method="nearest"
            ).load()
            pressure_levels = tuple(
                int(level) for level in atmospheric_dataset.pressure_level.values
            )
            has_climate_data = True
        except KeyError:
            surface_variables_by_day = None
            has_climate_data = False
            pressure_levels = None

        if has_climate_data:
            for lat_idx, lat in enumerate(lat_range):
                for lon_idx, lon in enumerate(lon_range):

                    for var_name in ["t2m", "msl", "u10", "v10"]:
                        var_value = (
                            surface_variables_by_day[var_name]
                            .sel(latitude=lat, longitude=lon, method="nearest")
                            .values
                        )
                        surfaces_variables[var_name][
                            t, lat_idx, lon_idx
                        ] = torch.tensor(var_value.item())

                    for var_name in ["z", "lsm", "slt"]:
                        var_value = (
                            single_variables_by_day[var_name]
                            .sel(latitude=lat, longitude=lon, method="nearest")
                            .values
                        )
                        single_variables[var_name][t, lat_idx, lon_idx] = torch.tensor(
                            var_value.item()
                        )

                    for var_name in ["z", "t", "u", "v", "q"]:
                        var_value = (
                            atmospheric_variables_by_day[var_name]
                            .sel(latitude=lat, longitude=lon, method="nearest")
                            .values
                        )
                        if var_name not in atmospheric_variables:
                            atmospheric_variables[var_name][
                                t, lat_idx, lon_idx
                            ] = torch.tensor(var_value.item())

        try:
            species_variables_by_day = species_dataset[
                species_dataset["Timestamp"].apply(
                    lambda x: x[0] if x is not None else None
                )
                == pd.Timestamp(current_date[0])
            ]
            has_species_data = True
        except KeyError:
            species_variables_by_day = None
            has_species_data = False

        if has_species_data:
            for lat_idx, lat in enumerate(lat_range):
                for lon_idx, lon in enumerate(lon_range):
                    species_at_location = species_variables_by_day[
                        (species_variables_by_day["Latitude"] == lat)
                        & (species_variables_by_day["Longitude"] == lon)
                    ]

                    if not species_at_location.empty:
                        species_entry = species_at_location.iloc[0]

                        species_variables["Species"][
                            t, lat_idx, lon_idx
                        ] = torch.tensor(species_entry["Species"], dtype=torch.float16)

                        species_variables["Image"][t, lat_idx, lon_idx] = species_entry[
                            "Image"
                        ]

                        species_variables["Audio"][t, lat_idx, lon_idx] = species_entry[
                            "Audio"
                        ]

                        species_variables["eDNA"][t, lat_idx, lon_idx] = species_entry[
                            "eDNA"
                        ]

                        species_variables["Description"][
                            t, lat_idx, lon_idx
                        ] = species_entry["Description"]

                        for category in [
                            "Phylum",
                            "Class",
                            "Order",
                            "Family",
                            "Genus",
                            "Redlist",
                        ]:
                            species_variables[category][
                                t, lat_idx, lon_idx
                            ] = torch.tensor(
                                species_entry[category], dtype=torch.float16
                            )

    first_timestamp, *_ = dates[0]

    batch = DataBatch(
        surface_variables=surfaces_variables,
        single_variables=single_variables,
        atmospheric_variables=atmospheric_variables,
        species_variables=species_variables,
        batch_metadata=BatchMetadata(
            latitudes=torch.tensor(lat_range),
            longitudes=torch.tensor(lon_range),
            timestamp=(first_timestamp,),
            pressure_levels=pressure_levels,
        ),
    )

    target_dtype = torch.float32
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = preprocess_era5(
        batch,
        dtype=target_dtype,
        device=target_device,
        locations=locations,
        scales=scales,
    )

    return batch


def create_batches(
    surface_dataset: xr.Dataset,
    single_dataset: xr.Dataset,
    atmospheric_dataset: xr.Dataset,
    species_dataset: pd.DataFrame,
    load_type: str = "day-by-day",
) -> list[DataBatch]:
    """
    Create DataBatches by merging xarray-based ERA5 climate data with species data for each timestamp.

    Args:
        surface_dataset (xarray.Dataset): The surface-level ERA5 dataset.
        single_dataset (xarray.Dataset): The single-level ERA5 dataset.
        atmospheric_dataset (xarray.Dataset): The pressure-level ERA5 dataset.
        species_dataset (pd.DataFrame): DataFrame containing species data.
        load_type (str): Load type can be 'day-by-day' or 'large-file'. Default is 'day-by-day'.

    Returns:
        list[DataBatch]: A list of DataBatch objects containing both climate and species data.
    """

    species_dataset["Longitude"] = species_dataset["Longitude"].apply(
        lambda lon: (lon + 360) % 360 if lon < 0 else lon
    )

    def initialize_data(crop_lat_n=None, crop_lon_n=None):
        """
        Initialize common ranges, tensors, and return them.

        Args:
            crop_n (bool): If it is true, then crop the latitude and longitude arrays.
        """
        # TODO: Uncomment it
        # lat_ = species_dataset[["Latitude"]].drop_duplicates().dropna()
        # lon_ = species_dataset[["Longitude"]].drop_duplicates().dropna()

        # lat_range = lat_["Latitude"].values
        # lon_range = lon_["Longitude"].values

        # Testing with these coord.
        lat_range = np.array(
            [
                -27.25,
                29.75,
                -24.0,
                -27.0,
                -29.25,
                -14.25,
                -35.25,
                -30.25,
                -47.75,
                -35.25,
                -47.75,
                6.0,
                -30.5,
                41.25,
                -33.5,
                -15.25,
                34.25,
                44.25,
                -35.25,
                -15.75,
                -33.5,
                -29.25,
                -22.25,
                -47.75,
                52.25,
                -2.75,
                -16.0,
                -48.0,
                -23.75,
                -15.75,
                -21.75,
                -30.0,
                -26.75,
                -26.75,
                -2.0,
                -29.25,
                26.5,
                -43.25,
                -45.75,
                2.25,
                -1.5,
                -35.0,
                -35.0,
                -33.75,
                -33.75,
                -35.0,
                -35.0,
                -35.25,
                -35.25,
                -33.5,
                -33.5,
                -2.0,
                6.0,
                8.5,
                -47.75,
                -38.5,
                -26.25,
                -1.5,
                -7.0,
                -38.5,
                -6.5,
                -6.5,
                -25.25,
                -26.0,
                -16.75,
                -16.75,
                44.25,
                -41.5,
            ]
        )

        lon_range = np.array(
            [
                26.5,
                351.75,
                25.25,
                28.75,
                26.5,
                31.75,
                301.0,
                297.75,
                288.75,
                301.0,
                288.75,
                266.25,
                306.25,
                242.5,
                299.75,
                30.5,
                250.75,
                240.25,
                301.0,
                30.5,
                299.0,
                25.0,
                303.5,
                288.75,
                5.0,
                287.5,
                30.5,
                297.75,
                303.25,
                30.5,
                304.25,
                23.25,
                285.5,
                285.5,
                274.75,
                300.75,
                91.0,
                295.25,
                294.5,
                284.75,
                274.25,
                301.0,
                301.0,
                304.0,
                304.0,
                301.0,
                301.0,
                301.0,
                301.0,
                299.5,
                299.5,
                274.75,
                284.0,
                273.75,
                297.25,
                295.0,
                306.0,
                274.25,
                286.0,
                295.0,
                286.75,
                286.75,
                21.5,
                29.25,
                304.5,
                304.5,
                301.25,
                288.25,
            ]
        )

        lat_range = np.unique(lat_range[~np.isnan(lat_range)])
        lon_range = np.unique(lon_range[~np.isnan(lon_range)])

        if crop_lat_n is not None or crop_lon_n is not None:
            crop_lat_n = crop_lat_n if crop_lat_n is not None else len(lat_range)
            crop_lon_n = crop_lon_n if crop_lon_n is not None else len(lon_range)
            lat_range, lon_range = crop_lat_lon(
                lat_range, lon_range, crop_lat_n, crop_lon_n
            )

        lat_range, lon_range = rescale_sort_lat_lon(lat_range, lon_range)

        T = 2
        pressure_levels = 13

        climate_tensors = initialize_climate_tensors(
            lat_range, lon_range, T, pressure_levels
        )
        species_tensors = initialize_species_tensors(lat_range, lon_range, T)

        return lat_range, lon_range, climate_tensors, species_tensors

    def create_and_save_batch(
        timestamps,
        surfaces_variables,
        single_variables,
        atmospheric_variables,
        species_variables,
    ):
        """Create a batch and save it to disk."""

        date1 = timestamps[0][0].strftime("%Y-%m-%d")
        date2 = timestamps[1][0].strftime("%Y-%m-%d")
        batch_file = os.path.join(BATCHES_DATA_DIR, f"batch_{date1}_{date2}.pt")

        if os.path.exists(batch_file):
            print(f"Batch for {date1} to {date2} already exists. Skipping...")
            return None

        batch = create_batch(
            dates=timestamps,
            lat_range=lat_range,
            lon_range=lon_range,
            surface_dataset=surface_dataset,
            single_dataset=single_dataset,
            atmospheric_dataset=atmospheric_dataset,
            species_dataset=species_subset,
            surfaces_variables=surfaces_variables,
            single_variables=single_variables,
            atmospheric_variables=atmospheric_variables,
            species_variables=species_variables,
        )

        os.makedirs(os.path.dirname(batch_file), exist_ok=True)
        torch.save(batch, batch_file)
        return batch

    lat_range, lon_range, climate_tensors, species_tensors = initialize_data()
    surfaces_variables = climate_tensors["surface"]
    single_variables = climate_tensors["single"]
    atmospheric_variables = climate_tensors["atmospheric"]
    species_variables = species_tensors

    batches = []

    if load_type == "day-by-day":
        climate_timestamps = set(
            surface_dataset["valid_time"].values.astype("datetime64[s]").tolist()
        )
        min_timestamp = min(climate_timestamps)
        max_timestamp = max(climate_timestamps)

        species_subset = species_dataset[
            (
                species_dataset["Timestamp"].apply(
                    lambda x: x[0] if isinstance(x, tuple) else x
                )
                >= min_timestamp
            )
            & (
                species_dataset["Timestamp"].apply(
                    lambda x: x[0] if isinstance(x, tuple) else x
                )
                <= max_timestamp
            )
        ]

        timestamps = merge_timestamps(surface_dataset, species_subset)

        batch = create_and_save_batch(
            timestamps[:2],
            surfaces_variables,
            single_variables,
            atmospheric_variables,
            species_variables,
        )

        if batch is not None:
            return batch

    elif load_type == "large-file":
        timestamps = merge_timestamps(surface_dataset, species_dataset)

        for i in range(len(timestamps) - 1):
            timestamp_pair = timestamps[i : i + 2]

            species_subset = species_dataset[
                (species_dataset["Timestamp"] >= timestamp_pair[0])
                & (species_dataset["Timestamp"] < timestamp_pair[1])
            ]

            batch = create_and_save_batch(
                timestamp_pair,
                surfaces_variables,
                single_variables,
                atmospheric_variables,
                species_variables,
            )

            if batch is not None:
                batches.append(batch)

            reset_climate_tensors(
                surfaces_variables, single_variables, atmospheric_variables
            )
            reset_species_tensors(species_variables)

        return batches


def create_dataset(
    species_file: str,
    era5_directory: str,
    batch_metadata_file: str,
    load_type: str = "day-by-day",
    surface_file: str = None,
    single_file: str = None,
    atmospheric_file: str = None,
) -> list[DataBatch]:
    """
    Create DataBatches from the multimodal and ERA5 datasets and save the resulting batches
    and batch metadata for future use.

    Args:
        species_file (str): Path to the Parquet file for multimodal data.
        era5_directory (str): Directory containing sorted ERA5 NetCDF files (one per day).
        load_type (str): Specifies whether to load files 'day-by-day' or 'large-file'.
        surface_file (str): Path to the ERA5 surface dataset. (large file option)
        single_file (str): Path to the ERA5 single-level dataset. (large file option)
        atmospheric_file (str): Path to the ERA5 pressure-level dataset. (large file option)
        batch_metadata_file (str): Path to the Parquet file where batch metadata will be stored.

    Returns:
        None.
    """
    species_dataset = load_species_data(species_file)

    if load_type == "day-by-day":
        batches = []

        grouped_files = load_era5_files_grouped_by_date(era5_directory)

        for i in range(0, len(grouped_files) - 1, 2):
            (
                atmospheric_dataset_day1,
                single_dataset_day1,
                surface_dataset_day1,
            ) = grouped_files[i]
            (
                atmospheric_dataset_day2,
                single_dataset_day2,
                surface_dataset_day2,
            ) = grouped_files[i + 1]

            atmospheric_dataset_day1 = xr.open_dataset(atmospheric_dataset_day1)
            single_dataset_day1 = xr.open_dataset(single_dataset_day1)
            surface_dataset_day1 = xr.open_dataset(surface_dataset_day1)

            atmospheric_dataset_day2 = xr.open_dataset(atmospheric_dataset_day2)
            single_dataset_day2 = xr.open_dataset(single_dataset_day2)
            surface_dataset_day2 = xr.open_dataset(surface_dataset_day2)

            atmospheric_dataset = xr.concat(
                [atmospheric_dataset_day1, atmospheric_dataset_day2], dim="valid_time"
            )
            single_dataset = xr.concat(
                [single_dataset_day1, single_dataset_day2], dim="valid_time"
            )
            surface_dataset = xr.concat(
                [surface_dataset_day1, surface_dataset_day2], dim="valid_time"
            )

            batch = create_batches(
                surface_dataset=surface_dataset,
                single_dataset=single_dataset,
                atmospheric_dataset=atmospheric_dataset,
                species_dataset=species_dataset,
            )

            if batch is not None:
                batches.append(batch)

            atmospheric_dataset_day1.close()
            single_dataset_day1.close()
            surface_dataset_day1.close()
            atmospheric_dataset_day2.close()
            single_dataset_day2.close()
            surface_dataset_day2.close()

    elif load_type == "large-file":

        (
            surface_dataset,
            single_dataset,
            atmospheric_dataset,
        ) = load_era5_datasets(surface_file, single_file, atmospheric_file)

        batches = create_batches(
            surface_dataset=surface_dataset,
            single_dataset=single_dataset,
            atmospheric_dataset=atmospheric_dataset,
            species_dataset=species_dataset,
        )

        save_batch_metadata_to_parquet(batches, batch_metadata_file)

# src/data_creation/create_dataset.py

import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.config import paths
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
    load_world_bank_data,
)
from src.dataset_creation.metadata import BatchMetadata
from src.dataset_creation.preprocessing import (
    crop_lat_lon,
    initialize_agriculture_tensors,
    initialize_climate_tensors,
    initialize_forest_tensors,
    initialize_land_tensors,
    initialize_species_extinction_tensors,
    initialize_species_tensors,
    merge_timestamps,
    preprocess_and_normalize_species_data,
    preprocess_era5,
    rescale_sort_lat_lon,
    reset_climate_tensors,
    reset_tensors,
)
from src.dataset_creation.save_data import (
    save_as_parquet,
    save_batch_metadata_to_parquet,
)
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


def create_batch(
    dates: list,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    surface_dataset: xr.Dataset,
    single_dataset: xr.Dataset,
    atmospheric_dataset: xr.Dataset,
    species_dataset: pd.DataFrame,
    species_extinction_dataset: pd.DataFrame,
    land_dataset: pd.DataFrame,
    agriculture_dataset: pd.DataFrame,
    forest_dataset: pd.DataFrame,
    surfaces_variables: dict,
    single_variables: dict,
    atmospheric_variables: dict,
    species_variables: dict,
    species_extinction_variables: dict,
    land_variables: dict,
    agriculture_variables: dict,
    forest_variables: dict,
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
        species_extinction_dataset (pd.DataFrame): Species extinction data.
        land_dataset (pd.DataFrame): Land data.
        agriculture_dataset (pd.DataFrame): Agriculture data containing agriculture land, arable land, etc.
        forest_dataset (pd.DataFrame): Forest data.
        surfaces_variables (dict): Pre-initialized surface climate variables tensors.
        single_variables (dict): Pre-initialized single-level climate variables tensors.
        atmospheric_variables (dict): Pre-initialized atmospheric pressure variables tensors.
        species_variables (dict): Pre-initialized species variables tensors.
        species_extinction_variables (dict): Pre-initialized extinct species variables tensors.
        land_variables (dict): Pre-initialized land variables tensors.
        agriculture_variables (dict): Pre-initialized agriculture variables tensors.
        forest_variables (dict): Pre-initialized forest variables tensors.

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

                        # species_variables["Image"][t, lat_idx, lon_idx] = species_entry[
                        #     "Image"
                        # ]

                        # species_variables["Audio"][t, lat_idx, lon_idx] = species_entry[
                        #     "Audio"
                        # ]

                        # species_variables["eDNA"][t, lat_idx, lon_idx] = species_entry[
                        #     "eDNA"
                        # ]

                        species_variables["Description"][
                            t, lat_idx, lon_idx
                        ] = species_entry["Description"]

                        species_variables["Distribution"][
                            t, lat_idx, lon_idx
                        ] = torch.tensor(
                            species_entry["Distribution"], dtype=torch.float16
                        )

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

        year = pd.Timestamp(current_date[0]).year

        for lat_idx, lat in enumerate(lat_range):
            for lon_idx, lon in enumerate(lon_range):

                ndvi_at_location = land_dataset[
                    (land_dataset["Latitude"] == lat)
                    & (land_dataset["Longitude"] == lon)
                    & (land_dataset["Variable"] == "NDVI")
                ]

                if not ndvi_at_location.empty:
                    ndvi_value = ndvi_at_location.get(str(year), pd.NA).values[0]
                    land_variables["NDVI"][t, lat_idx, lon_idx] = torch.tensor(
                        ndvi_value, dtype=torch.float16
                    )

                land_at_location = land_dataset[
                    (land_dataset["Latitude"] == lat)
                    & (land_dataset["Longitude"] == lon)
                    & (land_dataset["Variable"] == "Land")
                ]
                if not land_at_location.empty:
                    land_value = land_at_location.get(str(year), pd.NA).values[0]
                    land_variables["Land"][t, lat_idx, lon_idx] = torch.tensor(
                        land_value, dtype=torch.float16
                    )

                agriculture_at_location = agriculture_dataset[
                    (agriculture_dataset["Latitude"] == lat)
                    & (agriculture_dataset["Longitude"] == lon)
                ]

                if not agriculture_at_location.empty:
                    agri_land_row = agriculture_at_location[
                        agriculture_at_location["Variable"] == "Agriculture"
                    ]
                    if not agri_land_row.empty:
                        agri_land = agri_land_row.get(f"Agri_{year}", pd.NA).values[0]
                        agriculture_variables["AgricultureLand"][
                            t, lat_idx, lon_idx
                        ] = torch.tensor(agri_land, dtype=torch.float16)

                    agri_irr_land_row = agriculture_at_location[
                        agriculture_at_location["Variable"] == "Agriculture_Irrigated"
                    ]
                    if not agri_irr_land_row.empty:
                        agri_irr_land = agri_irr_land_row.get(
                            f"Agri_{year}", pd.NA
                        ).values[0]
                        agriculture_variables["AgricultureIrrLand"][
                            t, lat_idx, lon_idx
                        ] = torch.tensor(agri_irr_land, dtype=torch.float16)

                    arable_land_row = agriculture_at_location[
                        agriculture_at_location["Variable"] == "Arable"
                    ]
                    if not arable_land_row.empty:
                        arable_land = arable_land_row.get(f"Agri_{year}", pd.NA).values[
                            0
                        ]
                        agriculture_variables["ArableLand"][
                            t, lat_idx, lon_idx
                        ] = torch.tensor(arable_land, dtype=torch.float16)

                    cropland_row = agriculture_at_location[
                        agriculture_at_location["Variable"] == "Cropland"
                    ]
                    if not cropland_row.empty:
                        cropland = cropland_row.get(f"Agri_{year}", pd.NA).values[0]
                        agriculture_variables["Cropland"][
                            t, lat_idx, lon_idx
                        ] = torch.tensor(cropland, dtype=torch.float16)

                forest_at_location = forest_dataset[
                    (forest_dataset["Latitude"] == lat)
                    & (forest_dataset["Longitude"] == lon)
                ]
                if not forest_at_location.empty:
                    forest_value = forest_at_location.get(
                        f"Forest_{year}", pd.NA
                    ).values[0]
                    forest_variables["Forest"][t, lat_idx, lon_idx] = torch.tensor(
                        forest_value, dtype=torch.float16
                    )

                extinction_at_location = species_extinction_dataset[
                    (species_extinction_dataset["Latitude"] == lat)
                    & (species_extinction_dataset["Longitude"] == lon)
                ]
                if not extinction_at_location.empty:
                    extinction_value = extinction_at_location[f"RLI_{year}"].values[0]
                    species_extinction_variables["Extinction_RLI"][
                        t, lat_idx, lon_idx
                    ] = torch.tensor(extinction_value, dtype=torch.float16)

    first_timestamp, *_ = dates[0]

    batch = DataBatch(
        surface_variables=surfaces_variables,
        single_variables=single_variables,
        atmospheric_variables=atmospheric_variables,
        species_variables=species_variables,
        species_extinction_variables=species_extinction_variables,
        land_variables=land_variables,
        agriculture_variables=agriculture_variables,
        forest_variables=forest_variables,
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
    species_extinction_dataset: pd.DataFrame,
    land_dataset: pd.DataFrame,
    agriculture_dataset: pd.DataFrame,
    forest_dataset: pd.DataFrame,
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

        min_lon, min_lat, max_lon, max_lat = -30, 34.0, 50.0, 72.0

        lat_range = np.arange(min_lat, max_lat + 0.25, 0.25)
        lon_range = np.arange(min_lon, max_lon + 0.25, 0.25)

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
        species_extinction_tensors = initialize_species_extinction_tensors(
            lat_range, lon_range, T
        )

        land_tensors = initialize_land_tensors(lat_range, lon_range, T)
        agriculture_tensors = initialize_agriculture_tensors(lat_range, lon_range, T)
        forest_tensors = initialize_forest_tensors(lat_range, lon_range, T)

        return (
            lat_range,
            lon_range,
            climate_tensors,
            species_tensors,
            species_extinction_tensors,
            land_tensors,
            agriculture_tensors,
            forest_tensors,
        )

    def create_and_save_batch(
        timestamps,
        surfaces_variables,
        single_variables,
        atmospheric_variables,
        species_variables,
        species_extinction_variables,
        land_variables,
        agriculture_variables,
        forest_variables,
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
            species_extinction_dataset=species_extinction_dataset,
            land_dataset=land_dataset,
            forest_dataset=forest_dataset,
            agriculture_dataset=agriculture_dataset,
            surfaces_variables=surfaces_variables,
            single_variables=single_variables,
            atmospheric_variables=atmospheric_variables,
            species_variables=species_variables,
            species_extinction_variables=species_extinction_variables,
            land_variables=land_variables,
            agriculture_variables=agriculture_variables,
            forest_variables=forest_variables,
        )

        os.makedirs(os.path.dirname(batch_file), exist_ok=True)
        torch.save(batch, batch_file)
        return batch

    (
        lat_range,
        lon_range,
        climate_tensors,
        species_tensors,
        species_extinction_tensors,
        land_tensors,
        agriculture_tensors,
        forest_tensors,
    ) = initialize_data()
    surfaces_variables = climate_tensors["surface"]
    single_variables = climate_tensors["single"]
    atmospheric_variables = climate_tensors["atmospheric"]
    species_variables = species_tensors
    species_extinction_variables = species_extinction_tensors
    land_variables = land_tensors
    agriculture_variables = agriculture_tensors
    forest_variables = forest_tensors

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
            species_extinction_variables,
            land_variables,
            agriculture_variables,
            forest_variables,
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
                species_extinction_variables,
                land_variables,
                agriculture_variables,
                forest_variables,
            )

            if batch is not None:
                batches.append(batch)

            reset_climate_tensors(
                surfaces_variables, single_variables, atmospheric_variables
            )
            reset_tensors(species_variables)
            reset_tensors(species_extinction_variables)
            reset_tensors(land_variables)
            reset_tensors(agriculture_variables)
            reset_tensors(forest_variables)

        return batches


def create_dataset(
    species_file: str,
    era5_directory: str,
    batch_metadata_file: str,
    load_type: str = "day-by-day",
    surface_file: str = None,
    single_file: str = None,
    atmospheric_file: str = None,
    agriculture_file: str = None,
    land_file: str = None,
    forest_file: str = None,
    species_extinction_file: str = None,
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
        agriculture_file (str): Path to the csv file for agriculture data.
        land_file (str): Path to the csv file for land data.
        forest_file (str): Path to the csv file for forest data.
        species_extinction_file (str): Path to the csv file for species extinction data.
        batch_metadata_file (str): Path to the Parquet file where batch metadata will be stored.

    Returns:
        None.
    """
    species_dataset = load_species_data(species_file)
    agriculture_dataset = load_world_bank_data(agriculture_file)
    land_dataset = load_world_bank_data(land_file)
    forest_dataset = load_world_bank_data(forest_file)
    species_extinction_dataset = load_world_bank_data(species_extinction_file)

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
                agriculture_dataset=agriculture_dataset,
                forest_dataset=forest_dataset,
                land_dataset=land_dataset,
                species_extinction_dataset=species_extinction_dataset,
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
            agriculture_dataset=agriculture_dataset,
            forest_dataset=forest_dataset,
            land_dataset=land_dataset,
            species_extinction_dataset=species_extinction_dataset,
        )

        save_batch_metadata_to_parquet(batches, batch_metadata_file)

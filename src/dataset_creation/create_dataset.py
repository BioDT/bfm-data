# src/data_creation/create_dataset.py

import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.config.paths import BATCHES_DATA_DIR
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
    preprocess_era5,
    rescale_sort_lat_lon,
    reset_climate_tensors,
    reset_tensors,
)


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

                        species_variables["eDNA"][t, lat_idx, lon_idx] = species_entry[
                            "eDNA"
                        ]

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
                    ndvi_value = ndvi_at_location.get(str(year), pd.NA)
                    if ndvi_value is not pd.NA:
                        land_variables["NDVI"][t, lat_idx, lon_idx] = torch.tensor(
                            ndvi_value.values[0], dtype=torch.float16
                        )

                land_at_location = land_dataset[
                    (land_dataset["Latitude"] == lat)
                    & (land_dataset["Longitude"] == lon)
                    & (land_dataset["Variable"] == "Land")
                ]
                if not land_at_location.empty:
                    land_value = land_at_location.get(str(year), pd.NA)
                    if land_value is not pd.NA:
                        land_variables["Land"][t, lat_idx, lon_idx] = torch.tensor(
                            land_value.values[0], dtype=torch.float16
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
                        for var, field in [
                            ("Agriculture", "AgricultureLand"),
                            ("Agriculture_Irrigated", "AgricultureIrrLand"),
                            ("Arable", "ArableLand"),
                            ("Cropland", "Cropland"),
                        ]:
                            agri_row = agriculture_at_location[
                                agriculture_at_location["Variable"] == var
                            ]
                            if not agri_row.empty:
                                agri_value = agri_row.get(f"Agri_{year}", pd.NA)
                                if agri_value is not pd.NA:
                                    agriculture_variables[field][
                                        t, lat_idx, lon_idx
                                    ] = torch.tensor(
                                        agri_value.values[0], dtype=torch.float16
                                    )

                forest_at_location = forest_dataset[
                    (forest_dataset["Latitude"] == lat)
                    & (forest_dataset["Longitude"] == lon)
                ]

                if not forest_at_location.empty:
                    forest_value = forest_at_location.get(f"Forest_{year}", pd.NA)
                    if forest_value is not pd.NA:
                        forest_variables["Forest"][t, lat_idx, lon_idx] = torch.tensor(
                            forest_value.values[0], dtype=torch.float16
                        )

                extinction_at_location = species_extinction_dataset[
                    (species_extinction_dataset["Latitude"] == lat)
                    & (species_extinction_dataset["Longitude"] == lon)
                ]
                if not extinction_at_location.empty:
                    extinction_value = extinction_at_location.get(f"RLI_{year}", pd.NA)
                    if extinction_value is not pd.NA:
                        species_extinction_variables["ExtinctionValue"][
                            t, lat_idx, lon_idx
                        ] = torch.tensor(
                            extinction_value.values[0], dtype=torch.float16
                        )

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

        print(batch_file)

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
        start_date = np.datetime64("2000-01-01T00:00:00", "s")

        climate_timestamps = set(
            surface_dataset["valid_time"].values.astype("datetime64[s]").tolist()
        )

        climate_timestamps = {t for t in climate_timestamps if t >= start_date}
        print(climate_timestamps)
        if climate_timestamps:

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
            print(timestamps)
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

            create_batch_for_pair_of_days(
                atmospheric_dataset_day1,
                single_dataset_day1,
                surface_dataset_day1,
                atmospheric_dataset_day2,
                single_dataset_day2,
                surface_dataset_day2,
                species_dataset,
                agriculture_dataset,
                forest_dataset,
                land_dataset,
                species_extinction_dataset,
            )

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


def create_batch_for_pair_of_days(
    atmospheric_dataset_day1,
    single_dataset_day1,
    surface_dataset_day1,
    atmospheric_dataset_day2,
    single_dataset_day2,
    surface_dataset_day2,
    species_dataset,
    agriculture_dataset,
    forest_dataset,
    land_dataset,
    species_extinction_dataset,
):
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

    atmospheric_dataset_day1.close()
    single_dataset_day1.close()
    surface_dataset_day1.close()
    atmospheric_dataset_day2.close()
    single_dataset_day2.close()
    surface_dataset_day2.close()

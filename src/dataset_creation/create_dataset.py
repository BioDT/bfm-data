# src/data_creation/create_dataset.py

import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.config import paths
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
    process_netcdf_files,
    rescale_sort_lat_lon,
    reset_climate_tensors,
    reset_tensors,
)


def safe_tensor_conversion(value, dtype=torch.float32, default=0.0):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return torch.tensor(default, dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def safe_int_conversion(value, default=0):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return int(value)


def add_sparse_value(tensor: torch.Tensor, index: list, value) -> torch.Tensor:
    """
    Add a value to a sparse tensor at the specified index.

    Args:
        tensor (torch.Tensor): Sparse tensor.
        index (list): List of indices where the value should be added.
        value: Scalar, torch.Tensor, or numpy array containing the value to add at the specified index.

    Returns:
        torch.Tensor: Updated sparse tensor.
    """
    # Handle torch.Tensor input
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:  # Ensure it's a single element
            raise ValueError(
                f"Value must contain a single element, but got {value.numel()} elements."
            )
        value = value.item()  # Convert single-element tensor to scalar

    # Handle numpy array input
    if isinstance(value, np.ndarray):
        if value.size != 1:  # Ensure it's a single element
            raise ValueError(
                f"Value must contain a single element, but got {value.size} elements."
            )
        value = value.item()  # Convert single-element numpy array to scalar

    # Ensure the value is a scalar
    if not isinstance(value, (float, int)):
        raise TypeError(
            f"Value must be a scalar (float or int), but got {type(value)}."
        )

    # Extract current indices and values
    current_indices = tensor._indices()
    current_values = tensor._values()

    # Check if the tensor is empty
    if current_indices.numel() == 0:  # No non-zero elements
        new_indices = torch.tensor([index], dtype=torch.int64).T
        new_values = torch.tensor([value], dtype=tensor.dtype)
    else:
        # Concatenate the new index and value
        new_indices = torch.cat(
            [current_indices, torch.tensor([index], dtype=torch.int64).T], dim=1
        )
        new_values = torch.cat(
            [current_values, torch.tensor([value], dtype=tensor.dtype)]
        )

    # Create a new sparse tensor with updated indices and values
    updated_tensor = torch.sparse_coo_tensor(
        new_indices, new_values, tensor.size(), dtype=tensor.dtype
    )

    return updated_tensor


def check_latlon_ranges(
    target_lat_range: np.ndarray,
    target_lon_range: np.ndarray,
    current_lat_range: np.ndarray,
    current_lon_range: np.ndarray,
):
    # latitude check
    assert (
        target_lat_range.min() == current_lat_range.min()
    ), f"min does not match: target_lat_range.min()={target_lat_range.min()} current_lat_range.min()={current_lat_range.min()}"
    assert (
        target_lat_range.max() == current_lat_range.max()
    ), f"max does not match: target_lat_range.max()={target_lat_range.max()} current_lat_range.max()={current_lat_range.max()}"
    assert (
        target_lat_range.shape == current_lat_range.shape
    ), f"shape does not match: target_lat_range.shape[0]={target_lat_range.shape[0]} current_lat_range.shape[0]={current_lat_range.shape[0]}"
    assert (
        target_lat_range[0] == current_lat_range[0]
    ), f"first element does not match: target_lat_range[0]={target_lat_range[0]} current_lat_range[0]={current_lat_range[0]}"
    # longitude check
    assert (
        target_lon_range.min() == current_lon_range.min()
    ), f"min does not match: target_lon_range.min()={target_lon_range.min()} current_lon_range.min()={current_lon_range.min()}"
    assert (
        target_lon_range.max() == current_lon_range.max()
    ), f"max does not match: target_lon_range.max()={target_lon_range.max()} current_lon_range.max()={current_lon_range.max()}"
    assert (
        target_lon_range.shape == current_lon_range.shape
    ), f"shape does not match: target_lon_range.shape[0]={target_lon_range.shape[0]} current_lon_range.shape[0]={current_lon_range.shape[0]}"
    assert (
        target_lon_range[0] == current_lon_range[0]
    ), f"first element does not match: target_lon_range[0]={target_lon_range[0]} current_lon_range[0]={current_lon_range[0]}"


def get_tensor_from_xarray_dataarray(
    dataarray: xr.DataArray, replace_nan_value: float | None = None
) -> torch.Tensor:
    """
    Convert an xarray DataArray to a PyTorch tensor.

    Args:
        dataarray (xr.DataArray): xarray DataArray to convert.
        replace_nan_value (float | None): Value to fill NaN values with. If None or False, NaN values are not replaced.

    Returns:
        torch.Tensor: PyTorch tensor with the same data as the input DataArray.
    """
    tensor = torch.tensor(dataarray.to_numpy())
    m = dataarray.to_numpy()
    if replace_nan_value is not None and replace_nan_value is not False:
        m = np.nan_to_num(m, nan=replace_nan_value)
    tensor = torch.tensor(m)
    return tensor


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
    species_set: set,
    species_extinction_variables: dict,
    land_variables: dict,
    agriculture_variables: dict,
    forest_variables: dict,
) -> dict:
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
    start_time_all_dates = datetime.now()

    locations, scales = get_mean_standard_deviation(
        surface_dataset, single_dataset, atmospheric_dataset
    )

    pressure_levels = (50, 500, 1000)
    # pressure_levels = tuple(
    #     int(level) for level in atmospheric_dataset.pressure_level.values
    # )

    for t, current_date in enumerate(dates):
        start_time_current_date = datetime.now()
        print(t)
        print(current_date)

        try:
            surface_variables_by_day = surface_dataset.sel(
                valid_time=current_date, method="nearest"
            ).load()
            single_variables_by_day = single_dataset.sel(
                valid_time=current_date, method="nearest"
            ).load()
            atmospheric_variables_by_day = atmospheric_dataset.sel(
                valid_time=current_date, method="nearest"
            ).load()
            pressure_levels = (50, 500, 1000)
            # pressure_levels = tuple(
            #     int(level) for level in atmospheric_dataset.pressure_level.values
            # )
            has_climate_data = True
        except KeyError:
            surface_variables_by_day = None
            has_climate_data = False
            pressure_levels = []
            single_variables_by_day = None
            atmospheric_variables_by_day = None

        if has_climate_data:
            assert surface_variables_by_day, "surface_variables_by_day is None"
            assert single_variables_by_day, "single_variables_by_day is None"
            assert atmospheric_variables_by_day, "atmospheric_variables_by_day is None"
            check_latlon_ranges(
                lat_range,
                lon_range,
                surface_variables_by_day.latitude.to_numpy(),
                surface_variables_by_day.longitude.to_numpy(),
            )

            check_latlon_ranges(
                lat_range,
                lon_range,
                single_variables_by_day.latitude.to_numpy(),
                single_variables_by_day.longitude.to_numpy(),
            )

            check_latlon_ranges(
                lat_range,
                lon_range,
                atmospheric_variables_by_day.latitude.to_numpy(),
                atmospheric_variables_by_day.longitude.to_numpy(),
            )
            start_time = datetime.now()
            for var_name in ["t2m", "msl"]:
                # for var_name in ["t2m", "msl", "u10", "v10"]:
                variable = surface_variables_by_day[var_name]
                tensor = get_tensor_from_xarray_dataarray(variable)
                surfaces_variables[var_name][t, :, :] = tensor

            for var_name in ["lsm"]:
                # for var_name in ["z", "lsm", "slt"]:
                variable = single_variables_by_day[var_name]
                tensor = get_tensor_from_xarray_dataarray(variable)
                single_variables[var_name][t, :, :] = tensor

            for var_name in ["z", "t"]:
                # for var_name in ["z", "t", "u", "v", "q"]:
                variable = atmospheric_variables_by_day[var_name]
                # m = variable.to_numpy()  # shape: (13, 153, 321)
                # only selecte wanted pressure levels (pressure_levels)
                all_pressure_levels = (
                    atmospheric_variables_by_day.pressure_level.to_numpy().tolist()
                )
                wanted_pressure_levels_indexes = [
                    all_pressure_levels.index(p) for p in pressure_levels
                ]
                # m = m[wanted_pressure_levels_indexes, :, :]
                # # print(m.shape)  # (3, 153, 321)
                # m_safe = np.nan_to_num(m, nan=0.0)
                # tensor = torch.tensor(m_safe)
                tensor = get_tensor_from_xarray_dataarray(variable)
                # only get the wanted pressure levels
                tensor = tensor[wanted_pressure_levels_indexes, :, :]
                # print(tensor.shape)  # (3, 153, 321)
                atmospheric_variables[var_name][t, :, :, :] = tensor

            end_time = datetime.now()
            print(
                "TIME: climate_data_loops:",
                end_time - start_time,
                "(from start current timestep):",
                end_time - start_time_current_date,
            )
        try:
            start_time = datetime.now()
            species_dataset["Timestamp"] = pd.to_datetime(
                species_dataset["Timestamp"], errors="coerce"
            )

            species_variables_by_day = species_dataset[
                species_dataset["Timestamp"] == pd.Timestamp(current_date)
            ]

            has_species_data = True
            print("Filtered species_variables_by_day:", species_variables_by_day)
            end_time = datetime.now()
            print(
                "TIME: filtered_species:",
                end_time - start_time,
                "(from start current timestep):",
                end_time - start_time_current_date,
            )
        except KeyError:
            species_variables_by_day = None
            has_species_data = False
            print("No species data found for the current date.")

        initial_species_ids = {
            11824,
            2082,
            9783,
            16067,
            16348,
            5997,
            10261,
            327,
            13833,
            9319,
            18673,
            16870,
            10265,
            15761,
            9060,
            10200,
            2393,
            511,
            20832,
            17663,
            15861,
        }
        max_species = 22
        extra_species_limit = 1

        if has_species_data:
            start_time = datetime.now()
            transformed_species_data = species_variables_by_day.copy()
            # no need, now we are in [-180,+180] range with longitudes
            # transformed_species_data["Longitude"] = transformed_species_data[
            #     "Longitude"
            # ].apply(lambda x: x + 360 if x < 0 else x)

            for lat_idx, lat in enumerate(lat_range):
                for lon_idx, lon in enumerate(lon_range):
                    species_at_location = transformed_species_data[
                        (transformed_species_data["Latitude"] == lat)
                        & (transformed_species_data["Longitude"] == lon)
                    ]

                    for _, species_entry in species_at_location.iterrows():

                        species_id = int(species_entry["Species"])

                        if species_id in initial_species_ids:
                            if species_id not in species_set:
                                species_set.add(species_id)
                                print(
                                    f"Added species_id {species_id} from initial_species_ids to species_set:",
                                    species_set,
                                )
                        else:
                            if (
                                len(species_set - initial_species_ids)
                                < extra_species_limit
                            ):
                                if len(species_set) < max_species:
                                    species_set.add(species_id)
                                    print(
                                        f"Added extra species_id {species_id} to species_set:",
                                        species_set,
                                    )
                                else:
                                    print(
                                        f"Skipping species_id {species_id}, max_species limit reached."
                                    )
                                    continue
                            else:
                                print(
                                    f"Skipping species_id {species_id}, extra species limit reached."
                                )
                                continue

                        species_idx = list(species_set).index(species_id) + 1

                        try:

                            # for var_name in ["Description", "eDNA", "Distribution"]:
                            #     if var_name in species_entry and pd.notna(species_entry[var_name]):
                            #         value = species_entry[var_name]
                            #         if isinstance(value, torch.Tensor) and value.numel() > 1:
                            #             # Store the full tensor for dynamic variables
                            #             species_variables["dynamic"][var_name][t, lat_idx, lon_idx, species_idx] = value
                            #         elif isinstance(value, torch.Tensor):
                            #             species_variables["dynamic"][var_name][t, lat_idx, lon_idx, species_idx] = value.item()
                            #         else:
                            #             species_variables["dynamic"][var_name][t, lat_idx, lon_idx, species_idx] = value

                            # for meta_name in ["Phylum", "Class", "Order", "Family", "Genus", "Redlist"]:
                            #     if meta_name in species_entry and pd.notna(species_entry[meta_name]):
                            #         value = species_entry[meta_name]
                            #         if isinstance(value, torch.Tensor):
                            #             if value.numel() != 1:
                            #                 raise ValueError(f"Tensor value for {meta_name} must contain a single element, got {value.numel()} elements.")
                            #             value = value.item()
                            #         species_variables["metadata"][meta_name] = add_sparse_value(
                            #             species_variables["metadata"][meta_name],
                            #             [t, lat_idx, lon_idx, species_idx],
                            #             value,
                            #         )

                            # species_variables["dynamic"]["Image"][t, lat_idx, lon_idx, species_idx] = safe_tensor_conversion(
                            # species_entry["Image"], dtype=torch.float32
                            # )

                            # species_variables["dynamic"]["Description"][
                            #     t, lat_idx, lon_idx, species_idx
                            # ] = safe_tensor_conversion(
                            #     species_entry["Description"], dtype=torch.float32
                            # )

                            # species_variables["dynamic"]["Audio"][t, lat_idx, lon_idx, species_idx] = safe_tensor_conversion(
                            #     species_entry["Audio"], dtype=torch.float32
                            # )

                            # species_variables["dynamic"]["eDNA"][
                            #     t, lat_idx, lon_idx, species_idx
                            # ] = safe_tensor_conversion(
                            #     species_entry["eDNA"], dtype=torch.float32
                            # )

                            species_variables["dynamic"]["Distribution"][
                                t, lat_idx, lon_idx, species_idx
                            ] = safe_tensor_conversion(
                                species_entry["Distribution"], dtype=torch.float32
                            )

                            species_variables["metadata"]["Phylum"][
                                t, lat_idx, lon_idx, species_idx
                            ] = safe_int_conversion(species_entry["Phylum"])

                            species_variables["metadata"]["Genus"][
                                t, lat_idx, lon_idx, species_idx
                            ] = safe_int_conversion(species_entry["Genus"])

                            species_variables["metadata"]["Class"][
                                t, lat_idx, lon_idx, species_idx
                            ] = safe_int_conversion(species_entry["Class"])

                            species_variables["metadata"]["Order"][
                                t, lat_idx, lon_idx, species_idx
                            ] = safe_int_conversion(species_entry["Order"])

                            species_variables["metadata"]["Family"][
                                t, lat_idx, lon_idx, species_idx
                            ] = safe_int_conversion(species_entry["Family"])

                            species_variables["metadata"]["Redlist"][
                                t, lat_idx, lon_idx, species_idx
                            ] = safe_int_conversion(species_entry["Redlist"])

                        except IndexError as e:
                            print(
                                f"IndexError for species_id {species_id} at (t={t}, lat_idx={lat_idx}, lon_idx={lon_idx}, species_idx={species_idx}): {e}"
                            )
            end_time = datetime.now()
            print(
                "TIME: species_data_loops:",
                end_time - start_time,
                "(from start current timestep):",
                end_time - start_time_current_date,
            )

        missing_initial_species = initial_species_ids - species_set
        for missing_species in missing_initial_species:
            if len(species_set) < max_species:
                species_set.add(missing_species)
                print(
                    f"Added missing initial species_id {missing_species} to species_set:",
                    species_set,
                )
            else:
                print(
                    f"Skipping missing initial species_id {missing_species}, max_species limit reached."
                )

        year = pd.Timestamp(current_date).year
        month_year = pd.Timestamp(current_date).strftime("%m/%Y")
        ndvi_column = f"NDVI_{month_year}"

        start_time = datetime.now()
        for lat_idx, lat in enumerate(lat_range):
            for lon_idx, lon in enumerate(lon_range):

                # land_at_location = land_dataset[
                # (land_dataset["Latitude"] == lat) & (land_dataset["Longitude"] == lon)
                # ]

                # if not land_at_location.empty:
                #     for var_name in ["Land", "NDVI"]:
                #         var_value = land_at_location.get(var_name, pd.NA)
                #         if pd.notna(var_value):
                #             land_variables[var_name] = add_sparse_value(
                #                 land_variables[var_name], [t, lat_idx, lon_idx], var_value
                #             )

                # ndvi_at_location = land_dataset[
                #     (land_dataset["Latitude"] == lat)
                #     & (land_dataset["Longitude"] == lon)
                # ]

                # if not ndvi_at_location.empty and ndvi_column in ndvi_at_location.columns:
                #     ndvi_value = ndvi_at_location.get(ndvi_column, pd.NA)
                #     if pd.notna(ndvi_value.iloc[0]):  # Use .iloc[0] to extract the first value
                #         land_variables["NDVI"] = add_sparse_value(
                #             land_variables["NDVI"], [t, lat_idx, lon_idx], ndvi_value.iloc[0]
                #         )

                # land_at_location = land_dataset[
                #     (land_dataset["Latitude"] == lat)
                #     & (land_dataset["Longitude"] == lon)
                # ]
                # if not land_at_location.empty and str(year) in land_at_location.columns:
                #     land_value = land_at_location.get(str(year), pd.NA)
                #     if pd.notna(land_value.iloc[0]):  # Use .iloc[0] to extract the first value
                #         land_variables["Land"] = add_sparse_value(
                #             land_variables["Land"], [t, lat_idx, lon_idx], land_value.iloc[0]
                #         )

                # agriculture_at_location = agriculture_dataset[
                #     (agriculture_dataset["Latitude"] == lat)
                #     & (agriculture_dataset["Longitude"] == lon)
                # ]

                # if not agriculture_at_location.empty:
                #     for var_name in ["AgricultureLand", "AgricultureIrrLand", "ArableLand", "Cropland"]:
                #         var_value = agriculture_at_location.get(f"Agri_{year}", pd.NA)
                #         if not var_value.empty and pd.notna(var_value.iloc[0]):  # Use .iloc[0] to extract the first value
                #             agriculture_variables[var_name] = add_sparse_value(
                #                 agriculture_variables[var_name], [t, lat_idx, lon_idx], var_value.iloc[0]
                #             )

                # forest_at_location = forest_dataset[
                #     (forest_dataset["Latitude"] == lat) & (forest_dataset["Longitude"] == lon)
                # ]

                # if not forest_at_location.empty:
                #     var_value = forest_at_location.get(f"Forest_{year}", pd.NA)
                #     if not var_value.empty and pd.notna(var_value.iloc[0]):  # Use .iloc[0] to extract the first value
                #         forest_variables["Forest"] = add_sparse_value(
                #             forest_variables["Forest"], [t, lat_idx, lon_idx], var_value.iloc[0]
                #         )

                # extinction_at_location = species_extinction_dataset[
                #     (species_extinction_dataset["Latitude"] == lat)
                #     & (species_extinction_dataset["Longitude"] == lon)
                # ]
                # if not extinction_at_location.empty:
                #     var_value = extinction_at_location.get(f"RLI_{year}", pd.NA)
                #     if not var_value.empty and pd.notna(var_value.iloc[0]):  # Use .iloc[0] to extract the first value
                #         species_extinction_variables["ExtinctionValue"] = add_sparse_value(
                #             species_extinction_variables["ExtinctionValue"], [t, lat_idx, lon_idx], var_value.iloc[0]
                #         )

                # ndvi_at_location = land_dataset[
                #     (land_dataset["Latitude"] == lat)
                #     & (land_dataset["Longitude"] == lon)
                # ]

                # if not ndvi_at_location.empty and ndvi_column in ndvi_at_location.columns:
                #     ndvi_value = ndvi_at_location.get(ndvi_column, pd.NA)
                #     if pd.notna(var_value):
                #         land_variables["NDVI"] = add_sparse_value(
                #             land_variables["NDVI"], [t, lat_idx, lon_idx], ndvi_value.values[0]
                #         )

                # land_at_location = land_dataset[
                #     (land_dataset["Latitude"] == lat)
                #     & (land_dataset["Longitude"] == lon)
                # ]
                # if not land_at_location.empty and str(year) in land_at_location.columns:
                #     land_value = land_at_location.get(str(year), pd.NA)
                #     if land_value is not pd.NA:
                #         land_variables["Land"] = add_sparse_value(
                #             land_variables["Land"], [t, lat_idx, lon_idx], land_value.values[0]
                #         )

                # agriculture_at_location = agriculture_dataset[
                #     (agriculture_dataset["Latitude"] == lat)
                #     & (agriculture_dataset["Longitude"] == lon)
                # ]

                # if not agriculture_at_location.empty:
                #     for var_name in ["AgricultureLand", "AgricultureIrrLand", "ArableLand", "Cropland"]:
                #         var_value = agriculture_at_location.get(f"Agri_{year}", pd.NA)
                #         if pd.notna(var_value):
                #             agriculture_variables[var_name] = add_sparse_value(
                #                 agriculture_variables[var_name], [t, lat_idx, lon_idx], var_value.values[0]
                #             )

                # forest_at_location = forest_dataset[
                # (forest_dataset["Latitude"] == lat) & (forest_dataset["Longitude"] == lon)
                # ]

                # if not forest_at_location.empty:
                #     var_value = forest_at_location.get(f"Forest_{year}", pd.NA)
                #     if pd.notna(var_value):
                #         forest_variables["Forest"] = add_sparse_value(
                #             forest_variables["Forest"], [t, lat_idx, lon_idx], var_value.values[0]
                #         )

                ndvi_at_location = land_dataset[
                    (land_dataset["Latitude"] == lat)
                    & (land_dataset["Longitude"] == lon)
                ]

                if (
                    not ndvi_at_location.empty
                    and ndvi_column in ndvi_at_location.columns
                ):
                    ndvi_value = ndvi_at_location.get(ndvi_column, pd.NA)
                    if ndvi_value is not pd.NA:
                        land_variables["NDVI"][t, lat_idx, lon_idx] = torch.tensor(
                            ndvi_value.values[0], dtype=torch.float16
                        )

                land_at_location = land_dataset[
                    (land_dataset["Latitude"] == lat)
                    & (land_dataset["Longitude"] == lon)
                ]
                if not land_at_location.empty and str(year) in land_at_location.columns:
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

                # extinction_at_location = species_extinction_dataset[
                #     (species_extinction_dataset["Latitude"] == lat)
                #     & (species_extinction_dataset["Longitude"] == lon)
                # ]
                # if not extinction_at_location.empty:
                #     var_value = extinction_at_location.get(f"RLI_{year}", pd.NA)
                #     if pd.notna(var_value):
                #         species_extinction_variables["ExtinctionValue"] = add_sparse_value(
                #             species_extinction_variables["ExtinctionValue"], [t, lat_idx, lon_idx], var_value.values[0]
                #         )

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

        end_time = datetime.now()
        print(
            "TIME: other_variables_loops:",
            end_time - start_time,
            "(from start current timestep):",
            end_time - start_time_current_date,
        )
    first_timestamp = dates[0]
    second_timestamp = dates[1]

    batch = {
        "surface_variables": surfaces_variables,
        "single_variables": single_variables,
        "atmospheric_variables": atmospheric_variables,
        "species_variables": species_variables,
        "species_extinction_variables": species_extinction_variables,
        "land_variables": land_variables,
        "agriculture_variables": agriculture_variables,
        "forest_variables": forest_variables,
        "batch_metadata": {
            "latitudes": torch.tensor(lat_range).tolist(),
            "longitudes": torch.tensor(lon_range).tolist(),
            "timestamp": [ts.isoformat() for ts in (first_timestamp, second_timestamp)],
            "pressure_levels": list(pressure_levels),
            "species_list": list(species_set),
        },
    }

    # batch = DataBatch(
    #     surface_variables=surfaces_variables,
    #     single_variables=single_variables,
    #     atmospheric_variables=atmospheric_variables,
    #     species_variables=species_variables,
    #     species_extinction_variables=species_extinction_variables,
    #     land_variables=land_variables,
    #     agriculture_variables=agriculture_variables,
    #     forest_variables=forest_variables,
    #     batch_metadata=BatchMetadata(
    #         latitudes=torch.tensor(lat_range),
    #         longitudes=torch.tensor(lon_range),
    #         timestamp=(first_timestamp, second_timestamp),
    #         pressure_levels=pressure_levels,
    #     ),
    # )

    # target_dtype = torch.float32
    # target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch = preprocess_era5(
    #     batch,
    #     dtype=target_dtype,
    #     device=target_device,
    #     locations=locations,
    #     scales=scales,
    # )

    return batch


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

    # lat_range, lon_range = rescale_sort_lat_lon(lat_range, lon_range) # Let's keep to [-180, 180]
    # reverse lat_range to go from North to South
    lat_range = lat_range[::-1]
    lat_range = lat_range.astype(float)
    lon_range = lon_range.astype(float)

    T = 2
    pressure_levels_len = 3
    num_species = 22

    climate_tensors = initialize_climate_tensors(
        lat_range, lon_range, T, pressure_levels_len
    )
    species_tensors = initialize_species_tensors(lat_range, lon_range, T, num_species)
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
    lat_range,
    lon_range,
    surface_dataset,
    single_dataset,
    atmospheric_dataset,
    species_dataset,
    species_extinction_dataset,
    land_dataset,
    agriculture_dataset,
    forest_dataset,
    species_set,
):
    """Create a batch and save it to disk."""

    # Include date and time in the batch file name
    date1 = timestamps[0].strftime("%Y-%m-%d_%H-%M-%S")
    date2 = timestamps[1].strftime("%Y-%m-%d_%H-%M-%S")
    batch_file = os.path.join(paths.BATCHES_DATA_DIR, f"batch_{date1}_to_{date2}.pt")

    print(f"Processing timestamps pair: ({timestamps[0]}, {timestamps[1]})")
    print(batch_file)

    if os.path.exists(batch_file):
        print(f"Batch for {date1} to {date2} already exists. Skipping...")
        return None

    start_time_all = datetime.now()

    batch = create_batch(
        dates=timestamps,
        lat_range=lat_range,
        lon_range=lon_range,
        surface_dataset=surface_dataset,
        single_dataset=single_dataset,
        atmospheric_dataset=atmospheric_dataset,
        species_dataset=species_dataset,
        species_extinction_dataset=species_extinction_dataset,
        land_dataset=land_dataset,
        forest_dataset=forest_dataset,
        agriculture_dataset=agriculture_dataset,
        surfaces_variables=surfaces_variables,
        single_variables=single_variables,
        atmospheric_variables=atmospheric_variables,
        species_variables=species_variables,
        species_set=species_set,
        species_extinction_variables=species_extinction_variables,
        land_variables=land_variables,
        agriculture_variables=agriculture_variables,
        forest_variables=forest_variables,
    )
    end_time = datetime.now()
    print("TIME: create_batch (total):", end_time - start_time_all)

    os.makedirs(os.path.dirname(batch_file), exist_ok=True)
    start_time = datetime.now()
    torch.save(batch, batch_file)
    end_time = datetime.now()
    print(
        "TIME: write_batch:",
        end_time - start_time,
        "(from start of batch creation):",
        end_time - start_time_all,
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
) -> int:
    """
    Create DataBatches by merging xarray-based ERA5 climate data with species data for each timestamp.

    Args:
        surface_dataset (xarray.Dataset): The surface-level ERA5 dataset.
        single_dataset (xarray.Dataset): The single-level ERA5 dataset.
        atmospheric_dataset (xarray.Dataset): The pressure-level ERA5 dataset.
        species_dataset (pd.DataFrame): DataFrame containing species data.
        load_type (str): Load type can be 'day-by-day' or 'large-file'. Default is 'day-by-day'.

    Returns:
        int: The number of batches created.
    """

    # we are in [-180, +180] range with longitudes
    # species_dataset["Longitude"] = species_dataset["Longitude"].apply(
    #     lambda lon: (lon + 360) % 360 if lon < 0 else lon
    # )

    species_set = set()

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

    count = 0

    if load_type == "day-by-day":
        start_date = np.datetime64("2000-01-01T00:00:00", "s")

        # all the timestamps (every 6 hours, for 2 files concatenated by caller = 8 timestamps)
        climate_timestamps = sorted(
            t
            for t in surface_dataset["valid_time"]
            .values.astype("datetime64[s]")
            .tolist()
            if t >= start_date
        )

        paired_timestamps = [
            (climate_timestamps[i], climate_timestamps[i + 1])
            # --> [0,1], [2,3], [4,5]
            for i in range(0, len(climate_timestamps) - 1, 2)
            # instead if we want all the transitions: [0,1], [1,2], [2,3] ...
            # for i in range(len(climate_timestamps) - 1)
            # TODO: need to avoid writing the same timestamp to multiple files, it's a waste of compute and space
        ]

        for timestamps in paired_timestamps:
            print(f"Processing timestamps pair: {timestamps}")

            batch = create_and_save_batch(
                timestamps=timestamps,
                surfaces_variables=surfaces_variables,
                single_variables=single_variables,
                atmospheric_variables=atmospheric_variables,
                species_variables=species_variables,
                species_extinction_variables=species_extinction_variables,
                land_variables=land_variables,
                agriculture_variables=agriculture_variables,
                forest_variables=forest_variables,
                lat_range=lat_range,
                lon_range=lon_range,
                surface_dataset=surface_dataset,
                single_dataset=single_dataset,
                atmospheric_dataset=atmospheric_dataset,
                species_dataset=species_dataset,
                species_extinction_dataset=species_extinction_dataset,
                land_dataset=land_dataset,
                agriculture_dataset=agriculture_dataset,
                forest_dataset=forest_dataset,
                species_set=species_set,
            )

            if batch is not None:
                count += 1

    elif load_type == "large-file":
        timestamps = merge_timestamps(surface_dataset, species_dataset)

        for i in range(len(timestamps) - 1):
            timestamp_pair = timestamps[i : i + 2]

            batch = create_and_save_batch(
                timestamps=timestamp_pair,
                surfaces_variables=surfaces_variables,
                single_variables=single_variables,
                atmospheric_variables=atmospheric_variables,
                species_variables=species_variables,
                species_extinction_variables=species_extinction_variables,
                land_variables=land_variables,
                agriculture_variables=agriculture_variables,
                forest_variables=forest_variables,
                lat_range=lat_range,
                lon_range=lon_range,
                surface_dataset=surface_dataset,
                single_dataset=single_dataset,
                atmospheric_dataset=atmospheric_dataset,
                species_dataset=species_dataset,
                species_extinction_dataset=species_extinction_dataset,
                land_dataset=land_dataset,
                agriculture_dataset=agriculture_dataset,
                forest_dataset=forest_dataset,
                species_set=species_set,
            )

            count += 1

            reset_climate_tensors(
                surfaces_variables, single_variables, atmospheric_variables
            )
            reset_tensors(species_variables)
            reset_tensors(species_extinction_variables)
            reset_tensors(land_variables)
            reset_tensors(agriculture_variables)
            reset_tensors(forest_variables)

    return count


def get_paths_for_files_pairs_of_days(era5_directory: str) -> List[Tuple[Dict, Dict]]:
    """
    Get candidate pairs of days for creating batches.
    """
    grouped_files = load_era5_files_grouped_by_date(era5_directory)
    all_values = []
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

        all_values.append(
            [
                {
                    "atmospheric": atmospheric_dataset_day1,
                    "single": single_dataset_day1,
                    "surface": surface_dataset_day1,
                },
                {
                    "atmospheric": atmospheric_dataset_day2,
                    "single": single_dataset_day2,
                    "surface": surface_dataset_day2,
                },
            ]
        )

    return all_values


def create_batches_for_pair_of_days(
    atmospheric_dataset_day1_path: str,
    single_dataset_day1_path: str,
    surface_dataset_day1_path: str,
    atmospheric_dataset_day2_path: str,
    single_dataset_day2_path: str,
    surface_dataset_day2_path: str,
    species_dataset: pd.DataFrame,
    agriculture_dataset: pd.DataFrame,
    forest_dataset: pd.DataFrame,
    land_dataset: pd.DataFrame,
    species_extinction_dataset: pd.DataFrame,
) -> int:
    all_files = [
        atmospheric_dataset_day1_path,
        single_dataset_day1_path,
        surface_dataset_day1_path,
        atmospheric_dataset_day2_path,
        single_dataset_day2_path,
        surface_dataset_day2_path,
    ]
    valid_files = process_netcdf_files(all_files)

    if len(valid_files) < 6:
        non_valid_files = set(all_files) - set(valid_files)
        print(
            f"Skipping date due to missing or invalid files: {non_valid_files}.",
        )
        return 0

    atmospheric_dataset_day1 = xr.open_dataset(atmospheric_dataset_day1_path)
    single_dataset_day1 = xr.open_dataset(single_dataset_day1_path)
    surface_dataset_day1 = xr.open_dataset(surface_dataset_day1_path)

    atmospheric_dataset_day2 = xr.open_dataset(atmospheric_dataset_day2_path)
    single_dataset_day2 = xr.open_dataset(single_dataset_day2_path)
    surface_dataset_day2 = xr.open_dataset(surface_dataset_day2_path)

    # here putting together the data for 2 days (2 different .nc files)
    atmospheric_dataset = xr.concat(
        [atmospheric_dataset_day1, atmospheric_dataset_day2], dim="valid_time"
    )
    single_dataset = xr.concat(
        [single_dataset_day1, single_dataset_day2], dim="valid_time"
    )
    surface_dataset = xr.concat(
        [surface_dataset_day1, surface_dataset_day2], dim="valid_time"
    )

    count = create_batches(
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

    return count


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
):
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
        pair_of_days_paths = get_paths_for_files_pairs_of_days(era5_directory)

        for i, (day_1_path, day_2_paths) in enumerate(pair_of_days_paths):
            atmospheric_dataset_day1_path = day_1_path["atmospheric"]
            single_dataset_day1_path = day_1_path["single"]
            surface_dataset_day1_path = day_1_path["surface"]
            atmospheric_dataset_day2_path = day_2_paths["atmospheric"]
            single_dataset_day2_path = day_2_paths["single"]
            surface_dataset_day2_path = day_2_paths["surface"]

            count = create_batches_for_pair_of_days(
                atmospheric_dataset_day1_path=atmospheric_dataset_day1_path,
                single_dataset_day1_path=single_dataset_day1_path,
                surface_dataset_day1_path=surface_dataset_day1_path,
                atmospheric_dataset_day2_path=atmospheric_dataset_day2_path,
                single_dataset_day2_path=single_dataset_day2_path,
                surface_dataset_day2_path=surface_dataset_day2_path,
                species_dataset=species_dataset,
                agriculture_dataset=agriculture_dataset,
                forest_dataset=forest_dataset,
                land_dataset=land_dataset,
                species_extinction_dataset=species_extinction_dataset,
            )
            print(f"successfully created {count} batches for pair of days")

    elif load_type == "large-file":

        (
            surface_dataset,
            single_dataset,
            atmospheric_dataset,
        ) = load_era5_datasets(surface_file, single_file, atmospheric_file)

        count = create_batches(
            surface_dataset=surface_dataset,
            single_dataset=single_dataset,
            atmospheric_dataset=atmospheric_dataset,
            species_dataset=species_dataset,
            agriculture_dataset=agriculture_dataset,
            forest_dataset=forest_dataset,
            land_dataset=land_dataset,
            species_extinction_dataset=species_extinction_dataset,
        )
        print(f"successfully created {count} batches for large-file")

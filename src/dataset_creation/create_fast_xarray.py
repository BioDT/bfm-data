"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.config.paths import *
from src.dataset_creation.load_data import load_species_data, load_world_bank_data


def get_latlon_matrix_for_columns(
    df: pd.DataFrame,
    column_names: List[str],
    lat_range: np.ndarray | List[float],
    lon_range: np.ndarray | List[float],
    default_value=np.nan,
) -> Dict[str, np.ndarray]:
    results = {
        column_name: np.full((len(lat_range), len(lon_range)), default_value)
        for column_name in column_names
    }
    for lat_idx, lat in enumerate(
        tqdm(lat_range, desc="converting df into np matrix (lat)")
    ):
        for lon_idx, lon in enumerate(lon_range):
            row_at_location = df[(df["Latitude"] == lat) & (df["Longitude"] == lon)]
            if not row_at_location.empty:
                for column_name in column_names:
                    try:
                        results[column_name][lat_idx, lon_idx] = row_at_location[
                            column_name
                        ].values[
                            0
                        ]  # TODO: there could be more rows, now just take the first one found
                    except Exception as e:
                        print(
                            f"Error at lat: {lat}, lon: {lon}, column: {column_name}, {e}"
                        )
    # print some stats
    for column_name in column_names:
        min = np.nanmin(results[column_name])
        max = np.nanmax(results[column_name])
        nan_count = np.isnan(results[column_name]).sum()
        nan_perc = nan_count / results[column_name].size
        print(f"Column: {column_name}, min: {min}, max: {max}, nan: {nan_perc:.2%}")
    return results


def convert_csv_to_netcdf(
    input_file_path: str,
    output_file_path: str,
    lat_range: np.ndarray | List[float],
    lon_range: np.ndarray | List[float],
    type_file: str,
):
    # generic columns
    possible_coordinates_columns = ["Latitude", "Longitude"]
    possible_ignore_columns = [
        "Country",
        "Image",
        "Audio",
        "Timestamp",
        "unique_id",
        "Description",
        "File_path",
    ]

    if type_file == "species":
        df = load_species_data(input_file_path)
    else:
        df = load_world_bank_data(str(input_file_path))

    if type_file == "species":
        raise NotImplementedError("Species dataset not implemented yet")
        # TODO: how to get distributions of different species?
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        for species_id in df["Species"].unique():
            # species_id = 13833
            species_df = df[df["Species"] == species_id]
            column_name = f"Species_{species_id}"

    elif type_file == "agriculture":
        # The column "Variable" says what the values in the columns represent (names "Agri_{YEAR}")
        # Let's build columns that have {Variable}_{YEAR}
        columns_with_values = [col for col in df.columns.tolist() if "Agri_" in col]
        new_columns = []
        for col in tqdm(columns_with_values):
            for index, row in df.iterrows():
                variable = row["Variable"]
                year = col.split("_")[1]
                new_column_name = f"{variable}_{year}"
                df[new_column_name] = df[col]
                new_columns.append(new_column_name)
        # only keep new columns
        df = df[new_columns]

    elif type_file == "land":
        # columns with only year need to be renamed with "Land_{YEAR}", the others are already "NDVI_{MONTH}_{YEAR}"
        all_columns = df.columns.tolist()
        ndvi_columns = [col for col in all_columns if "NDVI_" in col]
        land_columns = [
            col
            for col in all_columns
            if col not in ndvi_columns
            and col not in possible_coordinates_columns
            and col not in possible_ignore_columns
        ]
        print(df.columns.tolist())
        df = df.rename(columns={col: f"Land_{col}" for col in land_columns})
        print(df.columns.tolist())

    # in this dataset columns
    all_columns = df.columns.tolist()
    ignore_columns = [c for c in possible_ignore_columns if c in all_columns]
    coordinates_columns = [c for c in possible_coordinates_columns if c in all_columns]
    assert (
        coordinates_columns
    ), "Coordinates columns in the dataset not found: Latitude, Longitude"
    numerical_columns = sorted(
        set(all_columns) - set(ignore_columns) - set(coordinates_columns)
    )
    print(f"Numerical columns: {numerical_columns}")

    matrices_by_name = get_latlon_matrix_for_columns(
        df, numerical_columns, lat_range, lon_range
    )
    # TODO: investigate how to keep string columns (e.g. Country)

    variables = {
        # no forward slash in variable name in HDF5-based files
        var_name.replace("/", "_"): xr.DataArray(
            data=matrices_by_name[
                var_name
            ],  # np.random.random((len(lat_range), len(lon_range))),  # enter data here
            dims=["latitude", "longitude"],
            coords={"latitude": lat_range, "longitude": lon_range},
            attrs={
                # "_FillValue": np.nan,
                # "units": "W/m2",
            },
        )
        for var_name in numerical_columns
    }

    # TODO: add non-geo variables

    ds = xr.Dataset(
        variables,
        # attrs={"example_attr": "this is a global attribute"},
    )
    # test if values is there
    # print(
    #     ds["NDVI_12_2017"].sel(latitude=39.75, longitude=19.75, method="nearest").values
    # )  # 0.54

    # remove file if exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # write to file
    ds.to_netcdf(output_file_path)
    # xr.open_dataset(output_file_path)


if __name__ == "__main__":
    # lat_lon ranges
    min_lon, min_lat, max_lon, max_lat = -30, 34.0, 50.0, 72.0
    lat_range = np.arange(min_lat, max_lat + 0.25, 0.25)
    lon_range = np.arange(min_lon, max_lon + 0.25, 0.25)
    lat_range = lat_range[::-1]
    lat_range = lat_range.astype(float)
    lon_range = lon_range.astype(float)

    files_to_convert = [
        # {
        #     "input_file_path": SPECIES_DATASET,
        #     "output_file_path": SPECIES_DATASET_NC,
        #     "type_file": "species",
        # }, # TODO: this is kept with batch creation slow script (16 secs)
        {
            "input_file_path": LAND_COMBINED_FILE,
            "output_file_path": LAND_COMBINED_FILE_NC,
            "type_file": "land",
        },  # 2 min 50 secs
        # {
        #     "input_file_path": AGRICULTURE_COMBINED_FILE,
        #     "output_file_path": AGRICULTURE_COMBINED_FILE_NC,
        #     "type_file": "agriculture",
        # }, # 20 minutes --> TODO: FAIL, for now don't generate
        {
            "input_file_path": FOREST_FILE,
            "output_file_path": FOREST_FILE_NC,
            "type_file": "forest",
        },
        {
            "input_file_path": SPECIES_EXTINCTION_FILE,
            "output_file_path": SPECIES_EXTINCTION_FILE_NC,
            "type_file": "species_extinction",
        },
    ]
    for cfg in files_to_convert:
        input_file_path = cfg["input_file_path"]
        output_file_path = cfg["output_file_path"]
        type_file = cfg["type_file"]
        print(f"Converting {input_file_path} to {output_file_path}")
        convert_csv_to_netcdf(
            str(input_file_path),
            str(output_file_path),
            lat_range,
            lon_range,
            type_file=type_file,
        )
        print(f"DONE {input_file_path} to {output_file_path}")

    # convert_csv_to_netcdf(
    #     str(AGRICULTURE_COMBINED_FILE),
    #     str(AGRICULTURE_COMBINED_FILE_NC),
    #     lat_range,
    #     lon_range,
    #     is_agriculture=True,
    # )

    # convert_csv_to_netcdf(
    #     str(SPECIES_DATASET),
    #     str(SPECIES_DATASET_NC),
    #     lat_range,
    #     lon_range,
    #     is_species=True,
    # )

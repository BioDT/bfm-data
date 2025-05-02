"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import pandas as pd

from bfm_data.config import paths
from bfm_data.data_ingestion.ingestion_scripts.worldbank import WorldBankDataProcessor


def run_land_data_processing(region: str = None, global_mode: bool = True):
    """
    Main function to process Land Area data. Processes globally if no region specified.

    Args:
        region (str): The region to process (e.g., 'Europe').
        global_mode (bool): Flag to run in global mode (default).
    """
    data_dir = paths.LAND_DIR
    data_file = paths.LAND_FILE

    if region:
        output_csv = f"{data_dir}/{region}_land_data.csv"
    else:
        output_csv = f"{data_dir}/global_land_data.csv"

    processor = WorldBankDataProcessor(
        data_file=data_file,
        output_csv=output_csv,
        region=region,
        global_mode=global_mode,
    )

    processor.process_data(value_key="Land")


def merge_land_data(
    land_file: str,
    ndvi_file: str,
    output_path: str,
    start_year=1961,
    end_year=2021,
):
    """
    Transform datasets for Land and NDVI to have unified year columns from 1961 to 2021.
    Each year will have values for Land and NDVI data without prefixes in the year columns.

    Args:
        land_file (str): Paths to the land CSV file.
        ndvi_file (str): Paths to the ndvi CSV file.
        output_path (str): Path to save the transformed output CSV file.
        start_year (int): Starting year for the columns (default is 1961).
        end_year (int): Ending year for the columns (default is 2021).
    """
    land_df = pd.read_csv(land_file)
    land_df_melted = land_df.melt(
        id_vars=["Country", "Latitude", "Longitude"],
        var_name="Year",
        value_name="Land_Value",
    )
    land_df_melted["Year"] = land_df_melted["Year"].str.extract(r"(\d+)$").astype(int)
    land_df_melted = land_df_melted[
        (land_df_melted["Year"] >= start_year) & (land_df_melted["Year"] <= end_year)
    ]

    land_pivoted = land_df_melted.pivot_table(
        index=["Country", "Latitude", "Longitude"],
        columns="Year",
        values="Land_Value",
        aggfunc="first",
    ).reset_index()

    ndvi_df = pd.read_csv(ndvi_file)
    ndvi_df_melted = ndvi_df.melt(
        id_vars=["Country", "Latitude", "Longitude"],
        var_name="Month_Year",
        value_name="NDVI_Value",
    )
    ndvi_df_melted = ndvi_df_melted.dropna(subset=["NDVI_Value"])

    ndvi_pivoted = ndvi_df_melted.pivot_table(
        index=["Country", "Latitude", "Longitude"],
        columns="Month_Year",
        values="NDVI_Value",
        aggfunc="first",
    ).reset_index()

    merged_df = pd.merge(
        land_pivoted,
        ndvi_pivoted,
        on=["Country", "Latitude", "Longitude"],
        how="outer",
    )

    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")


def run_land_merging():
    """
    Runs the merging function with specified paths, variable names, and output path.
    """

    land_file = paths.LAND_DIR / "Europe_land_data.csv"
    ndvi_file = paths.LAND_DIR / "Europe_ndvi_monthly_data.csv"

    output_path = paths.LAND_COMBINED_FILE

    merge_land_data(str(land_file), str(ndvi_file), str(output_path))

# src/data_ingestion/ingestion_scripts/land.py

import pandas as pd

from src.config import paths
from src.data_ingestion.ingestion_scripts.worldbank import WorldBankDataProcessor


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
    file_paths: list,
    variable_names: list,
    output_path: str,
    start_year=1961,
    end_year=2021,
):
    """
    Transform datasets for Land and NDVI to have unified year columns from 1961 to 2021.
    Each year will have values for Land and NDVI data without prefixes in the year columns.

    Args:
        file_paths (list of str): Paths to the CSV files.
        variable_names (list of str): Names of the variables for each file.
        output_path (str): Path to save the transformed output CSV file.
        start_year (int): Starting year for the columns (default is 1961).
        end_year (int): Ending year for the columns (default is 2021).
    """
    data_frames = []

    for file_path, variable_name in zip(file_paths, variable_names):
        df = pd.read_csv(file_path)

        df_melted = df.melt(
            id_vars=["Country", "Latitude", "Longitude"],
            var_name="Year",
            value_name="Value",
        )
        df_melted["Year"] = df_melted["Year"].str.extract(r"(\d+)$").astype(int)
        df_melted = df_melted[
            (df_melted["Year"] >= start_year) & (df_melted["Year"] <= end_year)
        ]
        df_melted["Variable"] = variable_name

        data_frames.append(df_melted)

    combined_df = pd.concat(data_frames, ignore_index=True)

    pivoted_df = combined_df.pivot_table(
        index=["Country", "Latitude", "Longitude", "Variable"],
        columns="Year",
        values="Value",
        aggfunc="first",
    ).reset_index()

    pivoted_df.columns.name = None
    pivoted_df.columns = [
        str(col) if isinstance(col, int) else col for col in pivoted_df.columns
    ]

    pivoted_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")


def run_land_merging():
    """
    Runs the merging function with specified paths, variable names, and output path.
    """
    file_paths = [
        "/data/projects/biodt/storage/data/Land/Europe_land_data.csv",
        "/data/projects/biodt/storage/data/Land/Europe_ndvi.csv",
    ]
    variable_names = ["Land", "NDVI"]
    output_path = paths.LAND_DIR / "Europe_combined_land_data.csv"

    merge_land_data(file_paths, variable_names, output_path)

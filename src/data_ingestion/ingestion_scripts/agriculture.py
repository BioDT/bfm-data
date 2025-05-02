"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import pandas as pd

from src.config import paths
from src.data_ingestion.ingestion_scripts.worldbank import WorldBankDataProcessor
from src.utils.merge_data import merge_world_bank_data


def run_agriculture_data_processing(
    region: str = None,
    global_mode: bool = True,
    irrigated: bool = False,
    arable: bool = False,
    cropland: bool = False,
):
    """
    Main function to process Agricultural Land data. Processes globally if no region specified.

    Args:
        region (str): The region to process (e.g., 'Europe').
        global_mode (bool): Flag to run in global mode (default).
        irrigated (bool): Flag to run the irrigated file.
        arable (bool): Flag to run the arable file.
        cropland (bool): Flag to run the permanent croppland file.
    """
    data_dir = paths.AGRICULTURE_DIR
    global_mode = not bool(region)

    base_name = "agriculture"
    if irrigated:
        base_name = "agriculture_irrigated"
        data_file = paths.AGRICULTURE_IRR_LAND_FILE
    elif arable:
        base_name = "arable"
        data_file = paths.ARABLE_LAND_FILE
    elif cropland:
        base_name = "cropland"
        data_file = paths.PERMANENT_CROPLAND_FILE
    else:
        data_file = paths.AGRICULTURE_LAND_FILE

    if region:
        output_csv = f"{data_dir}/{region}_{base_name}_data.csv"
    else:
        output_csv = f"{data_dir}/{base_name}_data.csv"

    processor = WorldBankDataProcessor(
        data_file=data_file,
        output_csv=output_csv,
        region=region,
        global_mode=global_mode,
    )

    processor.process_data(value_key="Agri")


def run_agriculture_merging():
    """
    Runs the merging function with specified paths, variable names, and output path.
    """
    file_paths = [
        "/data/projects/biodt/storage/data/Agriculture/Europe_agriculture_data.csv",
        "/data/projects/biodt/storage/data/Agriculture/Europe_agriculture_irrigated_data.csv",
        "/data/projects/biodt/storage/data/Agriculture/Europe_arable_data.csv",
        "/data/projects/biodt/storage/data/Agriculture/Europe_cropland_data.csv",
    ]

    variable_names = ["Agriculture", "Agriculture_Irrigated", "Arable", "Cropland"]

    output_path = paths.AGRICULTURE_DIR / "Europe_combined_agriculture_data.csv"

    merge_world_bank_data(file_paths, variable_names, output_path)

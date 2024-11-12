# src/data_ingestion/ingestion_scripts/forest.py

from src.config import paths
from src.data_ingestion.ingestion_scripts.worldbank import WorldBankDataProcessor


def run_forest_data_processing(region: str = None, global_mode: bool = True):
    """
    Main function to process Forest Area data. Processes globally if no region specified.

    Args:
        region (str): The region to process (e.g., 'Europe').
        global_mode (bool): Flag to run in global mode (default).
    """
    data_dir = paths.FOREST_DIR
    data_file = paths.FOREST_LAND_FILE

    if region:
        output_csv = f"{data_dir}/{region}_forest_data.csv"
    else:
        output_csv = f"{data_dir}/global_forest_data.csv"

    processor = WorldBankDataProcessor(
        data_file=data_file,
        output_csv=output_csv,
        region=region,
        global_mode=global_mode,
    )

    processor.process_data(value_key="Forest")

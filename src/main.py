# src/main.py

import argparse

from src.data_ingestion.api_clients.bold import bold
from src.data_ingestion.api_clients.era5 import era5
from src.data_ingestion.api_clients.inaturalist import inaturalist
from src.data_ingestion.api_clients.mapoflife import mop
from src.data_ingestion.api_clients.xenocanto import xeno_canto
from src.data_ingestion.ingestion_scripts.livingplanetindex import livingplanextindex
from src.data_ingestion.ingestion_scripts.species_occurences import (
    gbif_species_occurrence,
)
from src.dataset_creation.create_dataset import create_dataset


def run_era5(args):
    """Run the ERA5 climate data downloader."""
    era5(
        mode=args.mode,
        start_year=args.start_year,
        end_year=args.end_year,
        batch_size=args.batch_size,
        time=args.time,
    )


def run_xeno_canto(args):
    """Run the XenoCanto bird audio data downloader."""
    xeno_canto(
        species=args.species,
        download_all=args.download_all,
        audio_sample_rate=args.audio_sample_rate,
    )


def run_bold(args):
    """Run the BOLD barcode of life data downloader."""
    bold(txt=args.txt, scientific_name=args.scientific_name, name=args.name)


def run_inaturalist():
    """Run the iNaturalist species observation data downloader."""
    inaturalist()


def run_mapoflife():
    """Run the Map of Life data downloader."""
    mop()


def run_liningplanetindex():
    """Run the Living Planet Index data downloader"""
    livingplanextindex()


def run_gbif_species_occurrence():
    """Run the GBIF Species Occurrence data downloader"""
    gbif_species_occurrence()


def run_create_dataset(args):
    """Run the function to create data batches from ERA5 and species datasets."""
    create_dataset(
        species_file=args.species_file,
        era5_directory=args.era5_directory,
        batch_metadata_file=args.batch_metadata_file,
        load_type=args.load_type,
        surface_file=args.surface_file,
        single_file=args.single_file,
        atmospheric_file=args.atmospheric_file,
    )


def main():
    # parser = argparse.ArgumentParser(description="Data Downloader")
    # subparsers = parser.add_subparsers(title="commands", dest="command")

    # era5_parser = subparsers.add_parser("era5", help="Download ERA5 climate data")
    # era5_parser.add_argument(
    #     "--mode",
    #     type=str,
    #     required=True,
    #     choices=["range", "timestamps"],
    #     help="Mode of operation: 'range' or 'timestamps'",
    # )
    # era5_parser.add_argument(
    #     "--start_year",
    #     type=int,
    #     help="Start year for data download (required for 'range' mode)",
    # )
    # era5_parser.add_argument(
    #     "--end_year",
    #     type=int,
    #     help="End year for data download (required for 'range' mode)",
    # )
    # era5_parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=5,
    #     help="Batch size for timestamp mode (default: 5)",
    # )
    # era5_parser.add_argument(
    #     "--time",
    #     type=str,
    #     default="00/to/23/by/6",
    #     help="Time range for data download (default: '00/to/23/by/6')",
    # )
    # era5_parser.set_defaults(func=run_era5)

    # xeno_canto_parser = subparsers.add_parser(
    #     "xeno_canto", help="Download XenoCanto bird audio data"
    # )
    # xeno_canto_parser.add_argument(
    #     "--species",
    #     type=str,
    #     default="",
    #     help="Scientific name of the bird species to download",
    # )
    # xeno_canto_parser.add_argument(
    #     "--download_all", action="store_true", help="Flag to download all species"
    # )
    # xeno_canto_parser.add_argument(
    #     "--audio_sample_rate",
    #     type=int,
    #     default=16000,
    #     help="Sample rate for the downloaded audio",
    # )
    # xeno_canto_parser.set_defaults(func=run_xeno_canto)

    # bold_parser = subparsers.add_parser(
    #     "bold", help="Download BOLD barcode of life data"
    # )
    # bold_parser.add_argument(
    #     "--txt",
    #     action="store_true",
    #     help="Use text file for species names in BOLD download",
    # )
    # bold_parser.add_argument(
    #     "--scientific_name",
    #     type=str,
    #     help="Scientific name of the species for BOLD download",
    # )
    # bold_parser.add_argument(
    #     "--name",
    #     action="store_true",
    #     help="Download based on the provided scientific name in BOLD",
    # )
    # bold_parser.set_defaults(func=run_bold)

    # mapoflife_parser = subparsers.add_parser(
    #     "mapoflife", help="Download Map of Life data"
    # )
    # mapoflife_parser.set_defaults(func=run_mapoflife)

    # inaturalist_parser = subparsers.add_parser(
    #     "inaturalist", help="Download iNaturalist species observation data"
    # )
    # inaturalist_parser.set_defaults(func=run_inaturalist)

    # dataset_parser = subparsers.add_parser(
    #     "create_dataset", help="Create Data Batches from ERA5 and species datasets"
    # )
    # dataset_parser.add_argument(
    #     "--species_file",
    #     type=str,
    #     required=True,
    #     help="Path to the species file (Parquet)",
    # )
    # dataset_parser.add_argument(
    #     "--era5_directory",
    #     type=str,
    #     required=False,
    #     help="Directory for ERA5 files (day-by-day)",
    # )
    # dataset_parser.add_argument(
    #     "--batch_metadata_file",
    #     type=str,
    #     required=True,
    #     help="Path to the Parquet file for batch metadata",
    # )
    # dataset_parser.add_argument(
    #     "--load_type",
    #     type=str,
    #     required=True,
    #     choices=["day-by-day", "large-file"],
    #     help="Loading strategy for ERA5 files",
    # )
    # dataset_parser.add_argument(
    #     "--surface_file",
    #     type=str,
    #     required=False,
    #     help="Path to the surface dataset for large file loading",
    # )
    # dataset_parser.add_argument(
    #     "--single_file",
    #     type=str,
    #     required=False,
    #     help="Path to the single-level dataset for large file loading",
    # )
    # dataset_parser.add_argument(
    #     "--atmospheric_file",
    #     type=str,
    #     required=False,
    #     help="Path to the atmospheric dataset for large file loading",
    # )
    # dataset_parser.set_defaults(func=run_create_dataset)

    # args = parser.parse_args()

    # if args.command is None:
    #     parser.print_help()
    # else:
    #     args.func(args)

    from src.config import paths
    from src.data_ingestion.api_clients.era5 import era5
    from src.dataset_creation.create_dataset import create_dataset
    from src.utils.merge_data import save_sorted_timestamps

    create_dataset(
        species_file=paths.SPECIES_DATASET,
        era5_directory=paths.ERA5_DIR,
        agriculture_file=paths.AGRICULTURE_COMBINED_FILE,
        land_file=paths.LAND_COMBINED_FILE,
        forest_file=paths.FOREST_FILE,
        species_extinction_file=paths.SPECIES_EXTINCTION_FILE,
        load_type="day-by-day",
    )
    # create_species_dataset(paths.LIFE_DIR, '/data/projects/biodt/storage/processed_data/species_dataset3.parquet', start_year=2000, end_year=2005)

    # import torch

    # # Path to the saved batch file
    # batch_file_path = "/data/projects/biodt/storage/batches/batch_2000-01-01_2000-01-01.pt"

    # # Load the batch
    # batch = torch.load(batch_file_path, map_location='cpu')

    # print("Type of loaded batch:", type(batch))

    # for var_name, tensor in batch.surface_variables.items():
    #     print(f"Surface variable '{var_name}' shape: {tensor.shape}")

    # for var_name, tensor in batch.single_variables.items():
    #     print(f"Single variable '{var_name}' shape: {tensor.shape}")

    # for var_name, tensor in batch.atmospheric_variables.items():
    #     print(f"Atmospheric variable '{var_name}' shape: {tensor.shape}")
    #     num_nans = torch.isnan(tensor).sum().item()  # Count the number of NaN values
    #     print(f"Atmospheric variable '{var_name}' contains {num_nans} NaN values.")

    # for var_name, tensor in batch.species_variables.items():
    #     print(f"Species variable '{var_name}' shape: {tensor.shape}")

    #     # Count the number of NaN values
    #     num_nans = torch.isnan(tensor).sum().item()
    #     print(f"Species variable '{var_name}' contains {num_nans} NaN values.")

    #     # Count the number of non-NaN values
    #     num_non_nans = tensor.numel() - num_nans  # Total elements minus NaNs
    #     print(f"Species variable '{var_name}' contains {num_non_nans} non-NaN values.")

    # print("Metadata:", batch.batch_metadata)

    # import pandas as pd

    # # Load the parquet file
    # file_path = '/data/projects/biodt/storage/processed_data/species_dataset1.parquet'
    # df = pd.read_parquet(file_path)

    # # Total number of rows in the DataFrame
    # total_rows = len(df)
    # print(f"Total rows in the parquet file: {total_rows}")

    # # Count rows where 'Timestamp' is empty or null
    # # empty_timestamps_count = df['Timestamp'].isna().sum()
    # # print(f"Number of rows with empty timestamps: {empty_timestamps_count}")

    # # Count rows where any column is empty
    # empty_columns_count = df.isna().any(axis=1).sum()
    # print(f"Number of rows with at least one empty column: {empty_columns_count}")

    # # Count the number of empty cells per column
    # empty_cells_per_column = df.isna().sum()
    # print("Number of empty cells per column:")
    # print(empty_cells_per_column)

    # # Find rows where all columns (except Timestamp) are empty
    # empty_rows = df.drop(columns=['Timestamp']).isna().all(axis=1)

    # # Count unique timestamps for which the row is empty
    # empty_timestamps_count = df.loc[empty_rows, 'Timestamp'].nunique()

    # # Display the count
    # print(f"Number of unique timestamps with empty rows: {empty_timestamps_count}")

    # import pandas as pd

    # # Load the parquet file
    # file_path = '/data/projects/biodt/storage/processed_data/species_dataset1.parquet'
    # df = pd.read_parquet(file_path)

    # # Specify the timestamp to filter
    # target_timestamp = '2014-05-30T12:00:00.000000'

    # # Filter rows with the specific timestamp
    # filtered_rows = df[df['Timestamp'].apply(lambda x: target_timestamp in x)]

    # # Display the filtered rows
    # print(filtered_rows)

    # # Check if the target timestamp exists
    # timestamp_exists = target_timestamp in df['Timestamp'].values
    # print(f"Does the target timestamp exist? {timestamp_exists}")

    # filtered_rows = df[df['Timestamp'] == target_timestamp]
    # print(f"Rows with the target timestamp '{target_timestamp}':")
    # print(filtered_rows)

    # rows_with_target_timestamp = df[df['Timestamp'] == target_timestamp]

    # # Check if all columns except 'Timestamp' are empty
    # all_empty_except_timestamp = rows_with_target_timestamp.drop(columns=['Timestamp']).isna().all(axis=1)

    # print(f"Number of rows where all columns except 'Timestamp' are empty for '{target_timestamp}': {all_empty_except_timestamp.sum()}")

    # import pandas as pd

    # import pandas as pd

    # import pandas as pd

    # import pandas as pd

    # def round_to_nearest_025(value):
    #     """Round a value to the nearest 0.25, handling NaN."""
    #     if pd.isna(value):
    #         return value  # Keep NaN as is
    #     return round(value * 4) / 4

    # def process_species_data(input_csv, output_csv):
    #     # Read the CSV file and handle encoding issues
    #     try:
    #         df = pd.read_csv(input_csv, encoding='utf-8', low_memory=False)
    #     except UnicodeDecodeError:
    #         print("UTF-8 decoding failed. Trying with 'latin1'.")
    #         df = pd.read_csv(input_csv, encoding='latin1', low_memory=False)

    #     # Rename and select relevant columns
    #     df = df.rename(columns={'Binomial': 'species', 'Latitude': 'lat', 'Longitude': 'lon'})
    #     years = [str(year) for year in range(1950, 2021)]  # Adjust range as necessary

    #     # Filter the dataframe to keep only species, lat, lon, and years
    #     selected_columns = ['species', 'lat', 'lon'] + years
    #     df = df[selected_columns]

    #     # Transform latitude and longitude to 0.25-degree resolution
    #     df['lat'] = df['lat'].apply(round_to_nearest_025)
    #     df['lon'] = df['lon'].apply(round_to_nearest_025).apply(lambda x: round(x % 360, 6) if pd.notna(x) else x)

    #     # Save the processed data
    #     df.to_csv(output_csv, index=False)
    #     print(f"Processed data saved to {output_csv}")

    # # Example usage
    # input_csv = '/data/projects/biodt/storage/dataset_files/Living_Planet_Index/LPD2022_public.csv'  # Replace with your input file path
    # output_csv = '/data/projects/biodt/storage/data/Distribution/species_distribution.csv'  # Replace with your output file path
    # process_species_data(input_csv, output_csv)


if __name__ == "__main__":
    main()

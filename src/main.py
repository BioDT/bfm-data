# src/main.py

import argparse

from src.data_ingestion.api_clients.bold import bold
from src.data_ingestion.api_clients.era5 import era5
from src.data_ingestion.api_clients.inaturalist import inaturalist
from src.data_ingestion.api_clients.mapoflife import mop
from src.data_ingestion.api_clients.xenocanto import xeno_canto
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
    parser = argparse.ArgumentParser(description="Data Downloader")
    subparsers = parser.add_subparsers(title="commands", dest="command")

    era5_parser = subparsers.add_parser("era5", help="Download ERA5 climate data")
    era5_parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["range", "timestamps"],
        help="Mode of operation: 'range' or 'timestamps'",
    )
    era5_parser.add_argument(
        "--start_year",
        type=int,
        help="Start year for data download (required for 'range' mode)",
    )
    era5_parser.add_argument(
        "--end_year",
        type=int,
        help="End year for data download (required for 'range' mode)",
    )
    era5_parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for timestamp mode (default: 5)",
    )
    era5_parser.add_argument(
        "--time",
        type=str,
        default="00/to/23/by/6",
        help="Time range for data download (default: '00/to/23/by/6')",
    )
    era5_parser.set_defaults(func=run_era5)

    xeno_canto_parser = subparsers.add_parser(
        "xeno_canto", help="Download XenoCanto bird audio data"
    )
    xeno_canto_parser.add_argument(
        "--species",
        type=str,
        default="",
        help="Scientific name of the bird species to download",
    )
    xeno_canto_parser.add_argument(
        "--download_all", action="store_true", help="Flag to download all species"
    )
    xeno_canto_parser.add_argument(
        "--audio_sample_rate",
        type=int,
        default=16000,
        help="Sample rate for the downloaded audio",
    )
    xeno_canto_parser.set_defaults(func=run_xeno_canto)

    bold_parser = subparsers.add_parser(
        "bold", help="Download BOLD barcode of life data"
    )
    bold_parser.add_argument(
        "--txt",
        action="store_true",
        help="Use text file for species names in BOLD download",
    )
    bold_parser.add_argument(
        "--scientific_name",
        type=str,
        help="Scientific name of the species for BOLD download",
    )
    bold_parser.add_argument(
        "--name",
        action="store_true",
        help="Download based on the provided scientific name in BOLD",
    )
    bold_parser.set_defaults(func=run_bold)

    mapoflife_parser = subparsers.add_parser(
        "mapoflife", help="Download Map of Life data"
    )
    mapoflife_parser.set_defaults(func=run_mapoflife)

    inaturalist_parser = subparsers.add_parser(
        "inaturalist", help="Download iNaturalist species observation data"
    )
    inaturalist_parser.set_defaults(func=run_inaturalist)

    dataset_parser = subparsers.add_parser(
        "create_dataset", help="Create Data Batches from ERA5 and species datasets"
    )
    dataset_parser.add_argument(
        "--species_file",
        type=str,
        required=True,
        help="Path to the species file (Parquet)",
    )
    dataset_parser.add_argument(
        "--era5_directory",
        type=str,
        required=False,
        help="Directory for ERA5 files (day-by-day)",
    )
    dataset_parser.add_argument(
        "--batch_metadata_file",
        type=str,
        required=True,
        help="Path to the Parquet file for batch metadata",
    )
    dataset_parser.add_argument(
        "--load_type",
        type=str,
        required=True,
        choices=["day-by-day", "large-file"],
        help="Loading strategy for ERA5 files",
    )
    dataset_parser.add_argument(
        "--surface_file",
        type=str,
        required=False,
        help="Path to the surface dataset for large file loading",
    )
    dataset_parser.add_argument(
        "--single_file",
        type=str,
        required=False,
        help="Path to the single-level dataset for large file loading",
    )
    dataset_parser.add_argument(
        "--atmospheric_file",
        type=str,
        required=False,
        help="Path to the atmospheric dataset for large file loading",
    )
    dataset_parser.set_defaults(func=run_create_dataset)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()

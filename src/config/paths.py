# src/config/paths.py

from pathlib import Path

HOME_DIR = Path.home()
DATA_DIR = Path("/data/projects/biodt/storage/data")
STORAGE_DIR = Path("/data/projects/biodt/storage")

LIFE_DIR = DATA_DIR / "Life"
ERA5_DIR = DATA_DIR / "Copernicus/ERA5"

TEST_DATA_DIR = STORAGE_DIR / "test_data"
BATCHES_DATA_DIR = STORAGE_DIR / "batches"

PROCESSED_DATA_DIR = STORAGE_DIR / "processed_data"

ERA5_CONFIG_PATH = HOME_DIR / "bfm-data/src/config/era5_config/cds_api.txt"
CDSAPI_CONFIG_PATH = HOME_DIR / ".cdsapirc"

INAT_2021_MINI_DIR = STORAGE_DIR / "iNaturalist_2021_mini_dataset"
INAT_2021_MINI_JSON = (
    STORAGE_DIR
    / "dataset_files"
    / "iNaturalist_2021"
    / "iNaturalist_2021_mini_dataset.json"
)

XENO_CANTO_DIR = DATA_DIR / "Xeno_Canto"
XENO_CANTO_TXT = (
    STORAGE_DIR / "dataset_files" / "Xeno_Canto_GBIF" / "Xeno_Canto_dataset_gbif.txt"
)
XENO_CANTO_PROCESSED_LOG_FILE = (
    STORAGE_DIR / "dataset_files" / "xeno_canto_processed_audios.txt"
)

LPI_FILE = STORAGE_DIR / "dataset_files" / "Living_Planet_Index" / "LPD2022_public.csv"

ONLY_IMGS_PATHS = STORAGE_DIR / "folders_with_only_jpg.txt"
ALL_MOD_PATHS = STORAGE_DIR / "matching_directories.txt"

TIMESTAMPS = STORAGE_DIR / "sorted_timestamps.csv"

STATISTICS_DIR = STORAGE_DIR / "statistics"

SPECIES_OCCURRENCES_FILE = (
    STORAGE_DIR
    / "dataset_files"
    / "GBIF"
    / "Species_Occurrences"
    / "Netherlands_France_2017-2020.csv"
)

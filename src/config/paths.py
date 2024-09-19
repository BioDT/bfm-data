# src/config/paths.py

from pathlib import Path

DATA_DIR = "../../../data/projects/biodt/storage/data/"
LIFE_DIR = "../../../data/projects/biodt/storage/data/Life"
ERA5_DIR = "../../../data/projects/biodt/storage/data/Copernicus/ERA5"

TEST_DATA_DIR = "../../../data/projects/biodt/storage/test_data"
BATCHES_DATA_DIR = "../../../data/projects/biodt/storage/batches"

PROCESSED_DATA_DIR = "../../../data/projects/biodt/storage/processed_data"

ERA5_CONFIG_PATH = Path.home() / "bfm-data/src/config/era5_config/cds_api.txt"
CDSAPI_CONFIG_PATH = Path.home() / ".cdsapirc"

INAT_2021_MINI_DIR = "../../../data/projects/biodt/storage/data//data/projects/biodt/storage/data/iNaturalist_2021_mini_dataset"
INAT_2021_MINI_JSON = "../../../data/projects/biodt/storage/data//data/projects/biodt/storage/data/iNaturalist_2021_mini_dataset.json"

XENO_CANTO_DIR = "../../../data/projects/biodt/storage/data/Xeno_Canto"

XENO_CANTO_TXT = "../../../data/projects/biodt/storage/data/Xeno_Canto_dataset_gbif.txt"
XENO_CANTO_PROCESSED_LOG_FILE = (
    "../../../data/projects/biodt/storage/data/xeno_canto_processed_audios.txt"
)

ONLY_IMGS_PATHS = "../../../data/projects/biodt/storage/folders_with_only_jpg.txt"
ALL_MOD_PATHS = "../../../data/projects/biodt/storage/matching_directories.txt"

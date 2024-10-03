# src/config/paths.py

from pathlib import Path

HOME_DIR = Path.home()
DATA_DIR = Path("/data/projects/biodt/storage/data")

LIFE_DIR = DATA_DIR / "Life"
ERA5_DIR = DATA_DIR / "Copernicus/ERA5"

TEST_DATA_DIR = Path("/data/projects/biodt/storage/test_data")
BATCHES_DATA_DIR = Path("/data/projects/biodt/storage/batches")

PROCESSED_DATA_DIR = Path("/data/projects/biodt/storage/processed_data")

ERA5_CONFIG_PATH = HOME_DIR / "bfm-data/src/config/era5_config/cds_api.txt"
CDSAPI_CONFIG_PATH = HOME_DIR / ".cdsapirc"

INAT_2021_MINI_DIR = DATA_DIR / "iNaturalist_2021_mini_dataset"
INAT_2021_MINI_JSON = DATA_DIR / "iNaturalist_2021_mini_dataset.json"

XENO_CANTO_DIR = DATA_DIR / "Xeno_Canto"
XENO_CANTO_TXT = DATA_DIR / "Xeno_Canto_dataset_gbif.txt"
XENO_CANTO_PROCESSED_LOG_FILE = DATA_DIR / "xeno_canto_processed_audios.txt"

ONLY_IMGS_PATHS = Path("/data/projects/biodt/storage/folders_with_only_jpg.txt")
ALL_MOD_PATHS = Path("/data/projects/biodt/storage/matching_directories.txt")

TIMESTAMPS = Path("/data/projects/biodt/storage/sorted_timestamps.csv")

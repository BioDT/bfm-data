# src/config/settings.py

from pathlib import Path

DATA_DIR = "../../../data/projects/biodt/storage/data/"
LIFE_DIR = "../../../data/projects/biodt/storage/data/Life"

ERA5_DIR = "../../../data/projects/biodt/storage/data/Copernicus/ERA5"

ERA5_CONFIG_PATH = Path.home() / "bfm-data/src/config/era5_config/cds_api.txt"
CDSAPI_CONFIG_PATH = Path.home() / ".cdsapirc"

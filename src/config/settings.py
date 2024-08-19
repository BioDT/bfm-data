# src/config/settings.py

from pathlib import Path

DATA_DIR = "data/"
LIFE_DIR = "data/Life"

ERA5_CONFIG_PATH = Path.home() / "bfm-data/src/config/era5_config/cds_api.txt"
CDSAPI_CONFIG_PATH = Path.home() / ".cdsapirc"

"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import os
import platform
from pathlib import Path

HOME_DIR = Path.home()

if platform.node() == "hinton":
    PLATFORM = "hinton"
elif os.path.isdir("/projects/prjs1134"):
    PLATFORM = "snellius"
else:
    raise ValueError("Unknown storage location, cannote determine platform")

if PLATFORM == "hinton":
    STORAGE_DIR = Path("/data/projects/biodt/storage")  # hinton
elif PLATFORM == "snellius":
    STORAGE_DIR = Path("/projects/prjs1134/data/projects/biodt/storage")  # snellius
else:
    raise ValueError(f"Unknown platform {PLATFORM}")

DATA_DIR = STORAGE_DIR / "data"

LIFE_DIR = DATA_DIR / "Life"
ERA5_DIR = DATA_DIR / "Copernicus/ERA5"
LAND_DIR = DATA_DIR / "Land"
NDVI_DIR = DATA_DIR / "Copernicus/NDVI"
REDLIST_DIR = DATA_DIR / "RedList"
AGRICULTURE_DIR = DATA_DIR / "Agriculture"
FOREST_DIR = DATA_DIR / "Forest"
DATASET_FILES_DIR = STORAGE_DIR / "dataset_files"

TEST_DATA_DIR = STORAGE_DIR / "test_data"
BATCHES_DATA_DIR = STORAGE_DIR / "batches"

PROCESSED_DATA_DIR = STORAGE_DIR / "processed_data"

ERA5_CONFIG_PATH = HOME_DIR / "bfm-data/src/config/era5_config/cds_api.txt"
CDSAPI_CONFIG_PATH = HOME_DIR / ".cdsapirc"

INAT_2021_MINI_DIR = STORAGE_DIR / "iNaturalist_2021_mini_dataset"
INAT_2021_MINI_JSON = (
    DATASET_FILES_DIR / "iNaturalist_2021" / "iNaturalist_2021_mini_dataset.json"
)

XENO_CANTO_DIR = DATA_DIR / "Xeno_Canto"
XENO_CANTO_TXT = DATASET_FILES_DIR / "Xeno_Canto_GBIF" / "Xeno_Canto_dataset_gbif.txt"
XENO_CANTO_PROCESSED_LOG_FILE = DATASET_FILES_DIR / "xeno_canto_processed_audios.txt"

LPI_FILE = DATASET_FILES_DIR / "Living_Planet_Index" / "LPD2022_public.csv"

ONLY_IMGS_PATHS = (
    STORAGE_DIR / "folders_with_jpg_no_wav_no_edna_no_distribution_no_description.txt"
)
MODALITY_FOLDER_DIR = STORAGE_DIR / "modality_folder_lists"

TIMESTAMPS = STORAGE_DIR / "processed_data" / "timestamps" / "dates_2000_2024.csv"

STATISTICS_DIR = STORAGE_DIR / "statistics"

SPECIES_OCCURRENCES_FILE = DATASET_FILES_DIR / "GBIF" / "Species_Occurrences" / ".csv"

RED_LIST_FILE = DATASET_FILES_DIR / "Red_List" / "red_list_index.csv"

AGRICULTURE_LAND_FILE = (
    DATASET_FILES_DIR / "World_Bank" / "agricultural_land_1961_2021.csv"
)
AGRICULTURE_IRR_LAND_FILE = (
    DATASET_FILES_DIR / "World_Bank" / "agricultural_irrigated_land_2001_2021.csv"
)
ARABLE_LAND_FILE = DATASET_FILES_DIR / "World_Bank" / "arable_land_1961_2021.csv"
FOREST_LAND_FILE = DATASET_FILES_DIR / "World_Bank" / "forest_land_1990_2021.csv"
PERMANENT_CROPLAND_FILE = (
    DATASET_FILES_DIR / "World_Bank" / "permanent_cropland_1961_2021.csv"
)
LAND_FILE = DATASET_FILES_DIR / "World_Bank" / "land_area_1961_2021.csv"

SPECIES_DATASET = PROCESSED_DATA_DIR / "species_dataset.parquet"
SPECIES_DATASET_NC = PROCESSED_DATA_DIR / "species_dataset.nc"

AGRICULTURE_COMBINED_FILE = AGRICULTURE_DIR / "Europe_combined_agriculture_data.csv"
AGRICULTURE_COMBINED_FILE_NC = AGRICULTURE_DIR / "Europe_combined_agriculture_data.nc"

SPECIES_EXTINCTION_FILE = REDLIST_DIR / "Europe_red_list_index.csv"
SPECIES_EXTINCTION_FILE_NC = REDLIST_DIR / "Europe_red_list_index.nc"

FOREST_FILE = FOREST_DIR / "Europe_forest_data.csv"
FOREST_FILE_NC = FOREST_DIR / "Europe_forest_data.nc"

LAND_COMBINED_FILE = LAND_DIR / "Europe_combined_land_data.csv"
LAND_COMBINED_FILE_NC = LAND_DIR / "Europe_combined_land_data.nc"

MAPPING_FILE = PROCESSED_DATA_DIR / "labels_mapping" / "label_mappings.json"

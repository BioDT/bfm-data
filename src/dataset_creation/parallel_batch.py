# USAGE:
# python src/dataset_creation/parallel_batch.py create-list-file
# python src/dataset_creation/parallel_batch.py run-single 0

import json

import typer

from src.dataset_creation.create_dataset import create_batch_for_pair_of_days
from src.dataset_creation.load_data import (
    load_era5_datasets,
    load_era5_files_grouped_by_date,
    load_species_data,
    load_world_bank_data,
)

app = typer.Typer()

# the file to save the potential pairs of days for batch
list_file_path = "ERA5_days_pairs.json"

STORAGE_DIR = "/data/projects/biodt/storage"  # hinton
STORAGE_DIR = "/projects/prjs1134/data/projects/biodt/storage"  # snellius

DATA_DIR = f"{STORAGE_DIR}/data"

era5_directory = f"{DATA_DIR}/Copernicus/ERA5"

species_file = f"{STORAGE_DIR}/processed_data/species_dataset1.parquet"
agriculture_file = f"{DATA_DIR}/Agriculture/Europe_combined_agriculture_data.csv"
land_file = f"{DATA_DIR}/Land/Europe_combined_land_data.csv"
forest_file = f"{DATA_DIR}/Forest/Europe_forest_data.csv"
species_extinction_file = f"{DATA_DIR}/RedList/Europe_red_list_index.csv"


@app.command()
def create_list_file():
    grouped_files = load_era5_files_grouped_by_date(era5_directory)
    all_values = []

    for i in range(0, len(grouped_files) - 1, 2):
        (
            atmospheric_dataset_day1,
            single_dataset_day1,
            surface_dataset_day1,
        ) = grouped_files[i]
        (
            atmospheric_dataset_day2,
            single_dataset_day2,
            surface_dataset_day2,
        ) = grouped_files[i + 1]

        all_values.append(
            {
                "day_1": {
                    "atmospheric": atmospheric_dataset_day1,
                    "single": single_dataset_day1,
                    "surface": surface_dataset_day1,
                },
                "day_2": {
                    "atmospheric": atmospheric_dataset_day2,
                    "single": single_dataset_day2,
                    "surface": surface_dataset_day2,
                },
            }
        )

    with open(list_file_path, "w") as file:
        json.dump(all_values, file, indent=2)
    print(f"List file created with {len(all_values)} pairs of days: {list_file_path}")


@app.command()
def run_single(index: int):
    print("Running single day ingestion for index", index)
    with open(list_file_path, "r") as file:
        all_values = json.load(file)
    max_index = len(all_values)
    try:
        selected = all_values[index]
    except IndexError:
        print(f"Max index is {max_index}")
        return

    print("Selected data:", selected)
    atmospheric_dataset_day1 = selected["day_1"]["atmospheric"]
    single_dataset_day1 = selected["day_1"]["single"]
    surface_dataset_day1 = selected["day_1"]["surface"]
    atmospheric_dataset_day2 = selected["day_2"]["atmospheric"]
    single_dataset_day2 = selected["day_2"]["single"]
    surface_dataset_day2 = selected["day_2"]["surface"]

    species_dataset = load_species_data(species_file)
    agriculture_dataset = load_world_bank_data(agriculture_file)
    land_dataset = load_world_bank_data(land_file)
    forest_dataset = load_world_bank_data(forest_file)
    species_extinction_dataset = load_world_bank_data(species_extinction_file)

    create_batch_for_pair_of_days(
        atmospheric_dataset_day1,
        single_dataset_day1,
        surface_dataset_day1,
        atmospheric_dataset_day2,
        single_dataset_day2,
        surface_dataset_day2,
        species_dataset,
        agriculture_dataset,
        forest_dataset,
        land_dataset,
        species_extinction_dataset,
    )


if __name__ == "__main__":
    app()

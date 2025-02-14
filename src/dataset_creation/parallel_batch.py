# USAGE:
# python src/dataset_creation/parallel_batch.py create-list-file
# python src/dataset_creation/parallel_batch.py run-single 0

import json

import typer
import xarray as xr

from src.config.paths import *
from src.dataset_creation.create_dataset import (
    create_batches_for_pair_of_days,
    get_paths_for_files_pairs_of_days,
)
from src.dataset_creation.load_data import (
    load_era5_datasets,
    load_era5_files_grouped_by_date,
    load_species_data,
    load_world_bank_data,
)

app = typer.Typer(pretty_exceptions_enable=False)

# the file to save the potential pairs of days for batch
list_file_path = "ERA5_days_pairs.json"

era5_directory = ERA5_DIR

species_file = SPECIES_DATASET
agriculture_file = AGRICULTURE_COMBINED_FILE
land_file = LAND_COMBINED_FILE_NC
forest_file = FOREST_FILE_NC
species_extinction_file = SPECIES_EXTINCTION_FILE_NC


@app.command()
def create_list_file():
    all_pairs = get_paths_for_files_pairs_of_days(str(era5_directory))
    json_list = []
    for day_1_paths, day_2_paths in all_pairs:
        json_list.append(
            {
                "day_1": day_1_paths,
                "day_2": day_2_paths,
            }
        )

    with open(list_file_path, "w") as file:
        json.dump(json_list, file, indent=2)
    print(f"List file created with {len(json_list)} pairs of days: {list_file_path}")


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
    atmospheric_dataset_day1_path = selected["day_1"]["atmospheric"]
    single_dataset_day1_path = selected["day_1"]["single"]
    surface_dataset_day1_path = selected["day_1"]["surface"]
    atmospheric_dataset_day2_path = selected["day_2"]["atmospheric"]
    single_dataset_day2_path = selected["day_2"]["single"]
    surface_dataset_day2_path = selected["day_2"]["surface"]

    species_dataset = load_species_data(str(species_file))
    agriculture_dataset = load_world_bank_data(str(agriculture_file))
    land_dataset = xr.open_dataset(str(land_file))
    forest_dataset = xr.open_dataset(forest_file)
    species_extinction_dataset = xr.open_dataset(species_extinction_file)

    count = create_batches_for_pair_of_days(
        atmospheric_dataset_day1_path=atmospheric_dataset_day1_path,
        single_dataset_day1_path=single_dataset_day1_path,
        surface_dataset_day1_path=surface_dataset_day1_path,
        atmospheric_dataset_day2_path=atmospheric_dataset_day2_path,
        single_dataset_day2_path=single_dataset_day2_path,
        surface_dataset_day2_path=surface_dataset_day2_path,
        species_dataset=species_dataset,
        agriculture_dataset=agriculture_dataset,
        forest_dataset=forest_dataset,
        land_dataset=land_dataset,
        species_extinction_dataset=species_extinction_dataset,
    )
    print(f"successfully created {count} batches")
    print("FINISHED")


if __name__ == "__main__":
    app()

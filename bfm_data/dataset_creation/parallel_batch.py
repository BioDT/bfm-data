"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

# USAGE:
# python src/dataset_creation/parallel_batch.py create-list-file
# python src/dataset_creation/parallel_batch.py run-single 0


import typer
import xarray as xr

from bfm_data.config.paths import *
from bfm_data.dataset_creation.create_dataset import (  # create_batches_for_pair_of_days,; get_paths_for_files_pairs_of_days,
    create_batches,
    create_era5_range,
    get_chunk_parameters,
)
from bfm_data.dataset_creation.load_data import (
    load_era5_datasets,
    load_era5_files_grouped_by_date,
    load_species_data,
    load_world_bank_data,
)

app = typer.Typer(pretty_exceptions_enable=False)

# the file to save the potential pairs of days for batch
# list_file_path = "ERA5_days_pairs.json"

era5_directory = str(ERA5_DIR)

species_file = SPECIES_DATASET
agriculture_file = AGRICULTURE_COMBINED_FILE
land_file = LAND_COMBINED_FILE_NC
forest_file = FOREST_FILE_NC
species_extinction_file = SPECIES_EXTINCTION_FILE_NC

# chunk_size = 10  # how many dates to process in a single execution

paths_by_day = load_era5_files_grouped_by_date(era5_directory)


# @app.command()
# def create_list_file():
#     all_pairs = get_paths_for_files_pairs_of_days(str(era5_directory))
#     json_list = []
#     for day_1_paths, day_2_paths in all_pairs:
#         json_list.append(
#             {
#                 "day_1": day_1_paths,
#                 "day_2": day_2_paths,
#             }
#         )

#     with open(list_file_path, "w") as file:
#         json.dump(json_list, file, indent=2)
#     print(f"List file created with {len(json_list)} pairs of days: {list_file_path}")


@app.command()
def get_max_index(chunk_size: int = 10):
    all_params = get_chunk_parameters(paths_by_day, chunk_size)
    print(f"Max index is {len(all_params)}")


@app.command()
def run_single(run_index: int, chunk_size: int = 10, dry_run: bool = False):
    print(f"Running single day ingestion for run_index={run_index}")
    all_params = get_chunk_parameters(paths_by_day, chunk_size)

    assert run_index < len(
        all_params
    ), f"run_index={run_index} is out of bounds. Max value is {len(all_params)} - 1"

    parameters = all_params[run_index]

    print(
        "Selected parameters:",
        {k: v for k, v in parameters.items() if k not in ["paths_by_day"]},
    )

    print("Loading datasets...")
    species_dataset = load_species_data(str(species_file))
    agriculture_dataset = load_world_bank_data(str(agriculture_file))
    land_dataset = xr.open_dataset(str(land_file))
    forest_dataset = xr.open_dataset(forest_file)
    species_extinction_dataset = xr.open_dataset(species_extinction_file)
    print("Datasets loaded")

    print("Loading ERA5 datasets...")
    era5_pairs_range = create_era5_range(
        end_index=parameters["end_index"],
        start_index=parameters["start_index"],
        include_day_after_end=parameters["include_day_after_end"],
        paths_by_day=paths_by_day,
    )
    atmospheric_dataset, single_dataset, surface_dataset = era5_pairs_range
    print("ERA5 datasets loaded")

    print("Creating batches...")
    count = create_batches(
        surface_dataset=surface_dataset,
        single_dataset=single_dataset,
        atmospheric_dataset=atmospheric_dataset,
        species_dataset=species_dataset,
        agriculture_dataset=agriculture_dataset,
        forest_dataset=forest_dataset,
        land_dataset=land_dataset,
        species_extinction_dataset=species_extinction_dataset,
        dry_run=dry_run,
    )
    print(f"successfully created {count} batches")
    print("FINISHED")


if __name__ == "__main__":
    app()

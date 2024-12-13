# USAGE:
# python src/dataset_creation/parallel_batch.py create-list-file
# python src/dataset_creation/parallel_batch.py run-single 0

import json

import typer

from src.config.paths import DATA_DIR, STORAGE_DIR
from src.dataset_creation.create_dataset import (
    create_batch_for_pair_of_days,
    get_paths_for_files_pairs_of_days,
)
from src.dataset_creation.load_data import (
    load_era5_datasets,
    load_era5_files_grouped_by_date,
    load_species_data,
    load_world_bank_data,
)

app = typer.Typer()

# the file to save the potential pairs of days for batch
list_file_path = "ERA5_days_pairs.json"

era5_directory = f"{DATA_DIR}/Copernicus/ERA5"

species_file = f"{STORAGE_DIR}/processed_data/species_dataset.parquet"
agriculture_file = f"{DATA_DIR}/Agriculture/Europe_combined_agriculture_data.csv"
land_file = f"{DATA_DIR}/Land/Europe_combined_land_data.csv"
forest_file = f"{DATA_DIR}/Forest/Europe_forest_data.csv"
species_extinction_file = f"{DATA_DIR}/RedList/Europe_red_list_index.csv"


@app.command()
def create_list_file():
    all_pairs = get_paths_for_files_pairs_of_days(era5_directory)
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
        atmospheric_dataset_day1=atmospheric_dataset_day1,
        single_dataset_day1=single_dataset_day1,
        surface_dataset_day1=surface_dataset_day1,
        atmospheric_dataset_day2=atmospheric_dataset_day2,
        single_dataset_day2=single_dataset_day2,
        surface_dataset_day2=surface_dataset_day2,
        species_dataset=species_dataset,
        agriculture_dataset=agriculture_dataset,
        forest_dataset=forest_dataset,
        land_dataset=land_dataset,
        species_extinction_dataset=species_extinction_dataset,
    )


if __name__ == "__main__":
    app()

"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import MAPPING_FILE, SPECIES_DATASET
from src.dataset_creation.load_data import load_species_data
from src.dataset_creation.utils import get_lat_lon_ranges

endangered_species_ids = [
    11824,
    2082,
    9783,
    16067,
    16348,
    5997,
    10261,
    327,
    13833,
    9319,
    18673,
    16870,
    10265,
    15761,
    9060,
    10200,
    2393,
    511,
    20832,
    17663,
    15861,
]  # 21 species


def get_label_mapping(label_mapping_file: str | Path):
    with open(label_mapping_file, "r") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def get_species_names_map():
    label_mapping = get_label_mapping(MAPPING_FILE)
    name_by_id = {v: k for k, v in label_mapping["Species"].items()}
    return name_by_id


def get_species_name(species_id: int):
    name_by_id = get_species_names_map()
    return name_by_id[species_id]


def get_most_frequent_species(
    species_dataset: pd.DataFrame,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
):
    species_in_area_alltime = species_dataset[
        (species_dataset["Latitude"] >= min(lat_range))
        & (species_dataset["Latitude"] <= max(lat_range))
        & (species_dataset["Longitude"] >= min(lon_range))
        & (species_dataset["Longitude"] <= max(lon_range))
    ]
    # find name of species from File_path

    species_in_area_alltime["Name"] = species_in_area_alltime["Species"].apply(
        lambda x: get_species_name(x)
    )
    grouped_by_species = species_in_area_alltime[["Species", "Name"]].groupby(
        ["Species"]
    )
    counts = (
        grouped_by_species["Species"]
        .count()
        .sort_values(ascending=False)
        .to_frame("counts")
    )
    names = grouped_by_species["Name"].first().to_frame("Name")
    result = counts.join(names)
    print("most frequent species:", result)
    return result


def get_final_species(
    species_dataset: pd.DataFrame,
    lat_range: np.ndarray,
    lon_range: np.ndarray,
    max_species: int = 100,
    use_endangered_species: bool = False,
):
    counts = get_most_frequent_species(
        species_dataset=species_dataset, lat_range=lat_range, lon_range=lon_range
    )
    if use_endangered_species:
        # combine pre-defined list and top it up with most frequent
        initial_species_ids = endangered_species_ids
        assert max_species >= len(
            initial_species_ids
        ), f"max_species ({max_species}) is lower than initial_species_ids ({initial_species_ids})"
        extra_species_limit = max_species - len(initial_species_ids)
        # take top species from counts
        extra_species_ids = counts.index[:extra_species_limit].tolist()
        final_species_ids = initial_species_ids + extra_species_ids
    else:
        # only take the most frequent
        final_species_ids = counts.index[:max_species].tolist()
    print("final_species:")
    for species_id in final_species_ids:
        print(f"{species_id}: {get_species_name(species_id=species_id)}")
    return final_species_ids


if __name__ == "__main__":
    species_file_path = SPECIES_DATASET
    species_dataset = load_species_data(species_file=str(species_file_path))
    lat_range, lon_range = get_lat_lon_ranges()
    counts = get_most_frequent_species(
        species_dataset=species_dataset, lat_range=lat_range, lon_range=lon_range
    )
    species_ids = get_final_species(
        species_dataset=species_dataset, lat_range=lat_range, lon_range=lon_range
    )

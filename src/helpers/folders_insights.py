"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import os
from itertools import combinations

from src.config.paths import LIFE_DIR, MODALITY_FOLDER_DIR

modality_extensions = {
    "edna": "_edna.csv",
    "jpg": ".jpg",
    "wav": ".wav",
    "distribution": "_distribution.csv",
    "description": "_description.csv",
}

all_modality_keys = list(modality_extensions.keys())
combinations_dict = {}

for i in range(1, len(all_modality_keys) + 1):
    for combo in combinations(all_modality_keys, i):

        included = "_and_".join(combo)
        excluded = "_no_" + "_no_".join(set(all_modality_keys) - set(combo))
        file_name = f"folders_with_{included}{excluded}.txt"

        combinations_dict[file_name] = set(combo)

combinations_dict["all_modalities.txt"] = set(all_modality_keys)


def check_modality(folder, required_modalities):
    """
    Check if a folder contains all and only the specified combination of modalities.

    Args:
        folder (str): Path to the folder to check.
        required_modalities (set): Set of required modalities for the folder.

    Returns:
        bool: True if the folder contains exactly the required modalities; False otherwise.
    """
    found_modalities = set()
    for file in os.listdir(folder):
        for modality, ext in modality_extensions.items():
            if file.endswith(ext):
                found_modalities.add(modality)

    if (
        required_modalities.issubset(found_modalities)
        and found_modalities <= required_modalities
    ):
        return True
    return False


def generate_files():
    """
    Generate text files listing folders based on each combination of modalities.

    Walks through the LIFE_DIR directory, checks each subfolder for modality matches,
    and writes the paths of matching folders to text files in MODALITY_FOLDER_DIR.

    Returns:
        None
    """
    matching_folders = {key: [] for key in combinations_dict.keys()}

    for root, dirs, files in os.walk(LIFE_DIR):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            for combo_name, required_modalities in combinations_dict.items():
                if check_modality(folder_path, required_modalities):
                    matching_folders[combo_name].append(folder_path)

    if not os.path.exists(MODALITY_FOLDER_DIR):
        os.makedirs(MODALITY_FOLDER_DIR)

    for combo_name, folders in matching_folders.items():
        output_path = os.path.join(MODALITY_FOLDER_DIR, combo_name)
        with open(output_path, "w") as f:
            for folder_path in folders:
                f.write(f"{folder_path}\n")
        print(f"Generated {output_path} with {len(folders)} entries.")

# src/data_ingestion/ingestion_scripts/inaturalist.py

import csv
import json
import os
import shutil
from pathlib import Path

from src.config.paths import DATA_DIR


class iNaturalistDownloaderFromFile:
    """
    Class for creating metadata files based on the images and moving the images to an other folder.
    """

    def __init__(self, source_dir: str, json_file: str, destination_dir: str):
        """
        Initialize the iNaturalistDownloaderFromFile.

        Args:
            source_dir (str): The path of the source directory containing the images.
            json_file (str): Path to the JSON file containing the image dataset and annotations.
            destination_dir (str): The base path where images will be moved, organized by species names.
        """
        self.source_dir = source_dir
        self.json_file = json_file
        self.destination_dir = destination_dir

    def create_metadata_files(self):
        """
        Generates CSV metadata files for images based on a provided JSON dataset.

        This function reads a JSON file containing image metadata (such as categories, annotations, and image information),
        parses the information, and creates a CSV file for each folder containing the images. Each CSV file will store
        relevant metadata for every image in that folder. The metadata includes details like common name, scientific name,
        classification, coordinates, and image dimensions.

        Returns:
            None: Creates 'images_metadata.csv' files in the respective image directories.
        """

        with open(self.json_file, "r") as f:
            data = json.load(f)

        csv_columns = [
            "Observation_id",
            "Common_name",
            "id",
            "Kingdom",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
            "Scientific_name",
            "Location",
            "Latitude",
            "Longitude",
            "Timestamp",
            "Photo_url",
            "Photo_dimensions",
        ]

        categories = {category["id"]: category for category in data["categories"]}

        for image in data["images"]:
            image_path = os.path.join(DATA_DIR, image["file_name"])

            if not os.path.isfile(image_path):
                print(f"Image file {image_path} not found, skipping...")
                continue

            image_dir = os.path.dirname(image_path)

            annotation = next(
                (ann for ann in data["annotations"] if ann["image_id"] == image["id"]),
                None,
            )
            if not annotation:
                print(f"No annotation found for image ID {image['id']}, skipping...")
                continue

            category = categories.get(annotation["category_id"], {})
            image_id = Path(image["file_name"]).stem

            row = {
                "Observation_id": "Unknown",
                "Common_name": category.get("common_name", "Unknown"),
                "id": image_id,
                "Kingdom": category.get("kingdom", "Unknown"),
                "Phylum": category.get("phylum", "Unknown"),
                "Class": category.get("class", "Unknown"),
                "Order": category.get("order", "Unknown"),
                "Family": category.get("family", "Unknown"),
                "Genus": category.get("genus", "Unknown"),
                "Scientific_name": f"{category.get('genus', 'Unknown')} {category.get('specific_epithet', 'Unknown')}",
                "Location": "Unknown",
                "Latitude": image.get("latitude", "Unknown"),
                "Longitude": image.get("longitude", "Unknown"),
                "Photo_url": "Uknown",
                "Photo_dimensions": f"{image['width']}x{image['height']}",
                "Timestamp": image.get("date", "Unknown"),
            }

            csv_file_path = os.path.join(image_dir, f"{image_id}_image.csv")
            with open(csv_file_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerow(row)

    def move_images(self):
        """
        Moves images from a source directory to a destination directory, organizing them by species.

        This function scans the subdirectories within a specified source directory. For each subdirectory,
        it assumes the folder names contain species information in a specific format (i.e., 'Genus_Species').
        It creates corresponding species-named folders in the destination directory, and moves the images into
        their respective species-specific folder.

        Returns:
            None: Moves images to the destination directory organized into species-specific folders.
        """

        for folder_name in os.listdir(self.source_dir):
            folder_path = os.path.join(self.source_dir, folder_name)

            if os.path.isdir(folder_path):
                species_name = (
                    folder_path.split("_")[-2] + " " + folder_path.split("_")[-1]
                )
                species_dir = os.path.join(self.destination_dir, species_name)
                print(species_dir)

                if not os.path.exists(species_dir):
                    os.makedirs(species_dir)

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    shutil.move(file_path, os.path.join(species_dir, file_name))

        print("Images have been successfully moved.")


# TODO: Create run function

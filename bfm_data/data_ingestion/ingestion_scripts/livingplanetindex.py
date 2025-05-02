"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import os

import pandas as pd

from bfm_data.config import paths


class LivingPlanetIndexDownloader:
    def __init__(self, data_file: str, data_dir: str) -> None:
        """
        Initialize the downloader with the data file and the output folder path.

        Args:
            data_file (str): Path to the main CSV file with species data.
            data_dir (str): Path to the folder containing species-specific folders.
        """
        self.data_file = data_file
        self.data_dir = data_dir

    def extract_species_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the relevant columns for species distributions over the years,
        including Latitude, Longitude, and yearly data.

        Args:
            data (pd.DataFrame): DataFrame containing the original dataset.

        Returns:
            pd.DataFrame: DataFrame with species, latitude, longitude, and yearly data.
        """
        years_columns = [str(year) for year in range(1950, 2021)]
        species_columns = ["Binomial", "Latitude", "Longitude"] + years_columns

        species_data = data[species_columns]
        species_data = species_data.dropna(how="all", subset=years_columns)

        return species_data

    def save_species_distribution(self, species_data: pd.DataFrame) -> None:
        """
        Save species distribution for each species into its respective folder, grouped by species name.

        Args:
            species_data (pd.DataFrame): DataFrame containing species, lat/lon, and their yearly data.
        """
        species_data["Latitude"] = (species_data["Latitude"] * 4).round() / 4
        species_data["Longitude"] = (species_data["Longitude"] * 4).round() / 4

        grouped = species_data.groupby("Binomial")

        for species_name, group in grouped:
            species_name_ = species_name.replace("_", " ")

            species_folder = os.path.join(self.data_dir, species_name_)
            output_file = os.path.join(
                species_folder, f"{species_name_}_distribution.csv"
            )

            if not os.path.exists(species_folder):
                os.makedirs(species_folder)

            group.to_csv(output_file, index=False)
            print(f"Saved distribution for {species_name_} to {output_file}")

    def run(self) -> None:
        """
        Main method to process the file and save species distributions.
        """
        try:
            data = pd.read_csv(self.data_file, encoding="ISO-8859-1")
        except UnicodeDecodeError:
            print(
                f"Error reading {self.data_file}. Trying with a different encoding..."
            )
        data = pd.read_csv(self.data_file, encoding="latin1")
        species_data = self.extract_species_data(data)
        self.save_species_distribution(species_data)


def livingplanextindex():
    """
    Run the LivingPlanetIndexDownloader for species distribution data.

    Args:
        metadata (bool): If True, create metadata files.
        move (bool): If True, move the species distribution CSVs to the appropriate directories.
    """
    living_planet_downloader = LivingPlanetIndexDownloader(
        paths.LPI_FILE, paths.LIFE_DIR
    )
    living_planet_downloader.run()

    print("LivingPlanetIndex operation completed.")

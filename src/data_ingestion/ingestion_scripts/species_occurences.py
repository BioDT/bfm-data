# src/data_ingestion/ingestion_scripts/species_occurences.py

import os

import pandas as pd

from src.config import paths


class GBIFSpeciesOccurrenceDownloader:
    def __init__(self, data_file: str, data_dir: str) -> None:
        """
        Initialize the downloader with the data file and the output folder path.

        Args:
            data_file (str): Path to the main CSV file with GBIF species occurrence data.
            data_dir (str): Path to the folder where the output files will be saved.
        """
        self.data_file = data_file
        self.data_dir = data_dir

    def extract_species_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the relevant columns for species occurrences, including Latitude, Longitude, and eventDate.

        Args:
            data (pd.DataFrame): DataFrame containing the original dataset.

        Returns:
            pd.DataFrame: DataFrame with species, latitude, longitude, and eventDate.
        """
        species_columns = [
            "species",
            "decimalLatitude",
            "decimalLongitude",
            "eventDate",
        ]

        species_data = data[species_columns]
        species_data = species_data.dropna(subset=["eventDate", "species"])

        species_data["eventDate"] = pd.to_datetime(
            species_data["eventDate"], errors="coerce"
        )
        species_data["year"] = species_data["eventDate"].dt.year

        return species_data

    def save_species_distribution(self, species_data: pd.DataFrame) -> None:
        """
        Save species occurrences for each species into its respective folder, grouped by species name and year.

        Args:
            species_data (pd.DataFrame): DataFrame containing species, lat/lon, and eventDate.
        """
        grouped = species_data.groupby("species")

        for species_name, group in grouped:
            species_name_ = species_name.replace(" ", "_")
            species_folder = os.path.join(self.data_dir, species_name_)

            if not os.path.exists(species_folder):
                os.makedirs(species_folder)

            output_file = os.path.join(
                species_folder, f"{species_name_}_occurrences.csv"
            )

            yearly_group = (
                group.groupby("year")
                .agg(
                    total_observations=("decimalLatitude", "size"),
                    lat_lon_list=(
                        "decimalLatitude",
                        lambda x: list(zip(x, group["decimalLongitude"])),
                    ),
                )
                .reset_index()
            )

            yearly_group.to_csv(output_file, index=False)
            print(f"Saved observations for {species_name_} to {output_file}")

    def run(self) -> None:
        try:
            data = pd.read_csv(
                self.data_file, sep="\t", encoding="ISO-8859-1", on_bad_lines="skip"
            )
        except UnicodeDecodeError:
            print(
                f"Error reading {self.data_file}. Trying with a different encoding..."
            )
            data = pd.read_csv(
                self.data_file, sep="\t", encoding="latin1", on_bad_lines="skip"
            )

        species_data = self.extract_species_data(data)
        self.save_species_distribution(species_data)


def gbif_species_occurrence():
    """
    Run the GBIFSpeciesOccurrenceDownloader for species occurrence data.

    Args:
        metadata (bool): If True, create metadata files.
        move (bool): If True, move the species occurrence CSVs to the appropriate directories.
    """
    gbif_downloader = GBIFSpeciesOccurrenceDownloader(
        paths.SPECIES_OCCURRENCES_FILE, paths.TEST_DATA_DIR
    )
    gbif_downloader.run()

    print("GBIFSpeciesOccurrence operation completed.")

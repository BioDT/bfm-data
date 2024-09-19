# src/data_ingestion/api_clients/mapoflife.py

import os

from src.config.paths import ALL_MOD_PATHS
from src.data_ingestion.api_clients.downloader import Downloader
from src.utils.merge_data import extract_species_names


class MOL(Downloader):
    """
    A class to handle downloading data from Map of Life.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the MOL.

        Args:
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "Life")
        self.base_url = "https://api.mol.org/1.x/species/info"

    def get_save_data(self, scientific_name: str):
        """
        Fetches species information from the Map of Life API and saves it to a CSV file.

        Args:
            scientific_name (str): The scientific name of the species to fetch information for.

        Retrieves species descriptions and redlist status, formats the data, and saves
        the results into a CSV file named after the species.

        Returns:
            None
        """
        params = {"scientificname": scientific_name}

        json_response = self.get_base_url_page(params)

        data = []

        formatted_description = "No description available"
        redlist = "Not available"

        for _ in json_response:
            if (
                _["info"] is not None
                and isinstance(_["info"], list)
                and len(_["info"]) > 0
            ):
                description = _["info"][0].get("content", "")
                formatted_description = description.replace("\n", " ").replace("\r", "")
            if "redlist" in _:
                redlist = _["redlist"]

        scientific_name_path = os.path.join(self.base_path, scientific_name)
        os.makedirs(scientific_name_path, exist_ok=True)

        print(formatted_description)
        print(redlist)

        data.append(
            {
                "Scientific_name": scientific_name,
                "description": formatted_description,
                "redlist": redlist,
            }
        )

        self.save_to_csv(
            data,
            os.path.join(scientific_name_path, f"{scientific_name}_description.csv"),
        )

    def run(self):
        """
        Executes the data downloading process for all species in the provided dataset.

        Extracts species names from the defined paths, fetches their information from the
        Map of Life API, and saves the results for each species into individual CSV files.

        Returns:
            None
        """
        species_names = extract_species_names(ALL_MOD_PATHS)

        for scientific_name in species_names:
            print(f"Processing species: {scientific_name}")
            self.get_save_data(scientific_name)

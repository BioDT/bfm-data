# src/data_ingestion/api_clients/mapoflife.py

import os

import pandas as pd

from src.data_ingestion.api_clients.downloader import Downloader


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

    def get_save_data(self, scientific_name: str, filename: str):

        params = {"scientificname": scientific_name}

        json_response = self.get_base_url_page(params)

        data = []
        for _ in json_response:
            description = _["info"][0]["content"]
            formatted_description = description.replace("\n", " ").replace("\r", "")

        data.append(
            {"Scientific_name": scientific_name, "description": formatted_description}
        )

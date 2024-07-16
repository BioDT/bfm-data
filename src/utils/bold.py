# src/utils/bold.py

import os
from multiprocessing import Pool
from pathlib import Path

import pycountry

from src.utils.downloader import Downloader


class BOLDDownloader(Downloader):
    def __init__(self, data_dir: str):
        """
        Initialize the BOLDDownloader.

        Args:
        data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "BOLD")
        self.base_url = "https://www.boldsystems.org/index.php/API_Public/combined"
        self.limit = 100

    def get_bold_data(self, country: str, page: int = 1) -> list:
        """
        Fetch observations from BOLD.

        Args:
            country (str): Country name for querying the data from BOLD.
            page (int): Page number for paginated API results. Default is 1.

        Returns:
            list: List of observations.
        """

        while True:
            params = {
                "geo": country,
                "format": "json",
                "offset": (page - 1) * self.limit,
                "limit": self.limit,
            }
            json_response = self.get_base_url_page(params)
            bold_records = json_response.get("bold_records", {}).get("records", {})
            data = []

            if not bold_records:
                break

            self.process_data(bold_records, data)

            if len(bold_records) < self.limit:
                break

            page += 1

        return data

    def get_and_save_data(self):
        """
        Save observations to a CSV file.
        """
        countries = [country.name for country in pycountry.countries]
        for country in countries:
            data = self.get_bold_data(country)
            if data:
                if not os.path.exists(self.base_path):
                    os.makedirs(self.base_path)
                self.save_to_csv(data, os.path.join(self.base_path, "BOLD.csv"))

    def process_data(self, bold_records, data: list):

        for _, record_data in bold_records.items():
            record_id = record_data.get("record_id", "Unknown")
            bin_uri = record_data.get("bin_uri", "Unknown")

            taxonomy_ranks = ["phylum", "class", "order", "family", "genus", "species"]
            taxonomy = {}

            for rank in taxonomy_ranks:
                taxonomy[rank] = (
                    record_data.get("taxonomy", {})
                    .get(rank, {})
                    .get("taxon", {})
                    .get("name", "Unknown")
                )

            phylum = taxonomy["phylum"]
            class_ = taxonomy["class"]
            order = taxonomy["order"]
            family = taxonomy["family"]
            genus = taxonomy["genus"]
            species = taxonomy["species"]

            country = record_data.get("collection_event", {}).get("country", "Unknown")
            coordinates_data = record_data.get("collection_event", {}).get(
                "coordinates", {}
            )

            try:
                lat = float(coordinates_data.get("lat", "0.0"))
                lon = float(coordinates_data.get("lon", "0.0"))
            except ValueError:
                lat = 0.0
                lon = 0.0

            coordinates = f"[{lat}, {lon}]"

            sequence_info = record_data.get("sequences", {}).get("sequence", [{}])[0]
            sequence_id = sequence_info.get("sequenceID", "Unknown")
            nucleotides = sequence_info.get("nucleotides", "Unknown")

            data.append(
                {
                    "Record_id": record_id,
                    "Bin_uri": bin_uri,
                    "Phylum": phylum,
                    "Class": class_,
                    "Order": order,
                    "Family": family,
                    "Genus": genus,
                    "Species": species,
                    "Country": country,
                    "Coordinates": coordinates,
                    "Sequence_id": sequence_id,
                    "Nucleotides": nucleotides,
                }
            )

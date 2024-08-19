# src/data_ingestion/api_clients/bold.py

import os
from multiprocessing import Pool
from pathlib import Path

import pycountry

from src.data_ingestion.api_clients.downloader import Downloader


class BOLDDownloader(Downloader):
    def __init__(self, data_dir: str):
        """
        Initialize the BOLDDownloader.

        Args:
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "Life")
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

            if not bold_records:
                break

            self.process_data(bold_records)

            if len(bold_records) < self.limit:
                break

            page += 1

    def get_and_save_data(self):
        """
        Fetch BOLD data for all countries.
        """
        countries = [country.name for country in pycountry.countries]
        for country in countries:
            self.get_bold_data(country)

    def process_data(self, bold_records: dict):
        """
        Process BOLD records and save detailed taxonomic and observational data to CSV files.

        Args:
            bold_records (dict): A dictionary of BOLD records, where each key is a record identifier
                             and each value is a dictionary containing the record's data.
        """

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

            phylum = taxonomy.get("phylum", "Unknown")
            class_ = taxonomy.get("class", "Unknown")
            order = taxonomy.get("order", "Unknown")
            family = taxonomy.get("family", "Unknown")
            genus = taxonomy.get("genus", "Unknown")
            scientific_name = taxonomy.get("species", "Unknown")

            country = record_data.get("collection_event", {}).get("country", "Unknown")
            scientific_name_path = os.path.join(
                self.base_path, country, scientific_name
            )
            os.makedirs(scientific_name_path, exist_ok=True)

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

            data = [
                {
                    "Record_id": record_id,
                    "Bin_uri": bin_uri,
                    "Phylum": phylum,
                    "Class": class_,
                    "Order": order,
                    "Family": family,
                    "Genus": genus,
                    "Scientific_name": scientific_name,
                    "Country": country,
                    "Coordinates": coordinates,
                    "Sequence_id": sequence_id,
                    "Nucleotides": nucleotides,
                }
            ]

            self.save_to_csv(
                data, os.path.join(scientific_name_path, f"{scientific_name}_edna.csv")
            )

# src/data_ingestion/api_clients/bold.py

import os
import time

import pycountry

from src.config.paths import ONLY_IMGS_PATHS
from src.data_ingestion.api_clients.downloader import Downloader
from src.utils.merge_data import extract_species_names


class BOLDDownloader(Downloader):
    def __init__(self, data_dir: str):
        """
        Initialize the BOLDDownloader.

        Args:
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "Life")
        self.base_url = "https://www.boldsystems.org/index.php/API_Public/combined"

    def get_bold_data(self, query: str, is_species: bool = False) -> None:
        """
        Fetch observations from BOLD.

        Args:
            query (str): The query parameter, which could be a country name or a species name.
            is_species (bool): If True, query is treated as a species name; otherwise, as a country.

        Returns:
            list: List of observations.
        """
        param_key = "taxon" if is_species else "geo"

        while True:
            params = {
                param_key: query,
                "format": "json",
            }
            json_response = self.get_base_url_page(params)

            if json_response is None:
                break

            bold_records = json_response.get("bold_records", {}).get("records", {})

            if not bold_records:
                break

            self.process_data(bold_records)

            time.sleep(3)
            break

    def download(self, scientific_name: str = None) -> None:
        """
        Download BOLD data based on a country or species.

        Args:
            scientific_name (str): The scientific name of the species. If None, fetches data for all countries.
        """
        if scientific_name:
            self.get_bold_data(scientific_name, is_species=True)
        else:
            self.download_by_country()

    def download_by_country(self):
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
            scientific_name_path = os.path.join(self.base_path, scientific_name)
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
                    "Latitude": lat,
                    "Longitude": lon,
                    "Sequence_id": sequence_id,
                    "Nucleotides": nucleotides,
                }
            ]

            self.save_to_csv(
                data, os.path.join(scientific_name_path, f"{scientific_name}_edna.csv")
            )

    def download_species_based_on_txt(self):
        """
        Download species data based on species names extracted from text file paths.

        This function extracts species names from a predefined file path (`ONLY_IMGS_PATHS`),
        then processes each species name starting from index 853. It downloads data for
        each species using the `download` method.

        """
        species_names = extract_species_names(ONLY_IMGS_PATHS)
        species_names_to_process = species_names[853:]
        for scientific_name in species_names_to_process:
            print(f"Processing species: {scientific_name}")
            self.download(scientific_name)

    def run(self, txt: bool = False, scientific_name: str = None, name: bool = False):
        """
        Entry point to run the download process based on user input.

        This function decides the downloading method based on the provided arguments. It can
        either:
        1. Trigger the download process based on species names from text (if `txt` is True).
        2. Trigger the download process for a specific species using its scientific name (if `name` is True).
        3. Run the default download method if neither `txt` nor `name` is provided.

        Args:
            txt (bool): If True, download species based on text file paths. Default is False.
            scientific_name (str): The scientific name of the species to download. Required if `name` is True.
            name (bool): If True, download species based on the provided `scientific_name`. Default is False.
        """
        if txt:
            self.download_species_based_on_txt()
        elif name:
            if scientific_name:
                self.download(scientific_name)
            else:
                raise ValueError(
                    "Scientific name must be provided when 'name' is True."
                )
        else:
            self.download()

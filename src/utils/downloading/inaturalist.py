# src/utils/downloading/inaturalist.py

import os
from pathlib import Path

from geopy.geocoders import Nominatim

from src.utils.downloading.downloader import Downloader


class iNaturalistDownloader(Downloader):
    """
    A class to handle downloading data from iNaturalist.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the iNaturalistDownloader.

        Args:
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "Life")
        self.base_url = "https://api.inaturalist.org/v1/observations"
        self.geolocator = Nominatim(user_agent="iNaturalistdata")

    def get_observations(self, page: int = 1) -> list:
        """
        Fetch observations from iNaturalist.

        Args:
            page (int): Page number for paginated API results. Default is 1.

        Returns:
            list: List of observations.
        """

        # per_page (int): Allowed values: 1 to 200
        # has[] (list): Catch-all for some boolean selectors. (photos) - only show observations with photos.
        #                                                     (geo) - only show georeferenced observations
        # quality_grade (str) = 'research' / 'casual'

        params = {
            "per_page": 200,
            "has[]": ["photos", "geo"],
            "quality_grade": "research",
            "page": page,
        }

        json_response = self.get_base_url_page(params)

        num_pages = (json_response["total_results"] // json_response["per_page"]) + (
            1 if json_response["total_results"] % json_response["per_page"] != 0 else 0
        )

        observations = json_response["results"]

        self.process_and_download_observations(observations)

        if page < num_pages:
            self.get_observations(page + 1)

    def process_and_download_observations(self, observations: list):
        """
        Process and download the observations.

        Args:
            observations (list): List of observations.
        """
        for observation in observations:
            taxon = observation.get("taxon", {})
            taxonomy_ranks = ["kingdom", "phylum", "class", "order", "family", "genus"]
            taxonomy = {}

            ancestors = (
                observation.get("identifications", [{}])[0]
                .get("taxon", {})
                .get("ancestors", [{}])
            )
            for ancestor in ancestors:
                rank = ancestor.get("rank", "")
                name_ = ancestor.get("name", "")
                if rank in taxonomy_ranks:
                    taxonomy[rank] = name_

            kindgom = taxonomy.get("kindgom", "Unknown")
            phylum = taxonomy.get("phylum", "Unknown")
            class_ = taxonomy.get("class", "Unknown")
            order = taxonomy.get("order", "Unknown")
            family = taxonomy.get("family", "Unknown")
            genus = taxonomy.get("genus", "Unknown")

            for photo_counter, photo in enumerate(
                observation.get("photos", []), start=1
            ):

                photo_url = photo["url"].replace("square", "medium")
                if not photo_url.startswith("https://"):
                    continue
                country_name = self.get_country_from_coordinates(
                    observation.get("geojson", {}).get("coordinates", [])
                )
                observation_id = observation.get("id", "Unknown")
                preferred_common_name = taxon.get("preferred_common_name", "Unknown")
                scientific_name = taxon.get("name", "Unknown")
                scientific_name_path = os.path.join(
                    self.base_path, country_name, scientific_name
                )
                if not os.path.exists(scientific_name_path):
                    Path(scientific_name_path).mkdir(parents=True, exist_ok=True)

                if not photo_url.startswith("https://"):
                    return

                image_name = os.path.join(
                    scientific_name_path,
                    f"{preferred_common_name}_{scientific_name}_{observation_id}_{photo_counter}.jpg",
                )

                if os.path.exists(image_name):
                    continue

                species_data = [
                    {
                        "Observation_id": observation_id,
                        "Common_name": preferred_common_name,
                        "id": taxon["id"],
                        "Kingdom": kindgom,
                        "Phylum": phylum,
                        "Class": class_,
                        "Order": order,
                        "Family": family,
                        "Genus": genus,
                        "Scientific_name": scientific_name,
                        "Country": country_name,
                        "Place": observation["place_guess"],
                        "Coordinates": observation["geojson"]["coordinates"],
                        "Photo_url": photo_url,
                        "Photo_dimensions": " 375x500",
                    }
                ]

                self.download_file(photo_url, image_name)
                self.save_to_csv(
                    species_data,
                    os.path.join(scientific_name_path, f"{preferred_common_name}.csv"),
                )

    def get_country_from_coordinates(self, coords: str):
        """
        Returns the country associated with those coordinates using the Nominatim geocoding service.

        Args:
            coords (str): A string representing the coordinates in the format '[longitude, latitude]'.

        Returns:
            str: The name of the country corresponding to the given coordinates.
        """
        longitude, latitude = coords
        location = self.geolocator.reverse((latitude, longitude), language="en")
        return location.raw["address"].get("country", "") if location else "Unknown"

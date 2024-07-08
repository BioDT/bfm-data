# src/utils/inaturalist.py

import os
from multiprocessing import Pool
from pathlib import Path

from geopy.geocoders import Nominatim

from src.utils.downloader import Downloader


class iNaturalistDownloader(Downloader):
    """
    A class to handle downloading data from iNaturalist.
    """

    def __init__(self, data_dir):
        """
        Initialize the iNaturalistDownloader with country and data directory.

        Args:
            country (str): Country name for querying data.
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "iNaturalist")
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

        observations = [observation for observation in json_response["results"]]

        if page < num_pages:
            observations.extend(self.get_observations(page + 1))

        return observations

    def download_save_observations(self, observations, filename):
        """
        Save observations to a CSV file.

        Args:
            observations (list): List of observations.
            filename (str): Name of the CSV file to save the observations.
        """
        data = []
        for observation in observations:
            taxon = observation.get("taxon", {})
            for photo_counter, photo in enumerate(
                observation.get("photos", []), start=1
            ):

                photo_url = photo["url"].replace("square", "medium")
                country_name = self.get_country_from_coordinates(
                    observation["geojson"]["coordinates"]
                )
                taxon_name = taxon.get("iconic_taxon_name", "Unknown")
                preferred_common_name = taxon.get("preferred_common_name", "Uknown")
                name = taxon.get("name", "Unknown")
                observation_id = observation["id"]

                taxon_path = os.path.join(self.base_path, country_name, taxon_name)
                if not os.path.exists(taxon_path):
                    Path(taxon_path).mkdir(parents=True, exist_ok=True)

                if not photo_url.startswith("https://"):
                    return

                image_name = os.path.join(
                    taxon_path,
                    f"{preferred_common_name}_{name}_{observation_id}_{photo_counter}.jpg",
                )

                if os.path.exists(image_name):
                    continue

                data.append(
                    {
                        "Observation_id": observation_id,
                        "Taxon": taxon_name,
                        "preferred_common_name": preferred_common_name,
                        "Name": name,
                        "Species_Taxon_id": taxon["min_species_taxon_id"],
                        "Place": observation["place_guess"],
                        "Coordinates": observation["geojson"]["coordinates"],
                        "Photo_url": photo_url,
                        "Photo_dimensions": " 375x500",
                    }
                )

                self.download_file(photo_url, image_name)
        if data:
            self.save_to_csv(data, filename)
        else:
            return

    def get_country_from_coordinates(self, coords):
        """
        Returns the country associated with those coordinates using the Nominatim geocoding service.

        Parameters:
        coords (str): A string representing the coordinates in the format '[longitude, latitude]'.

        Returns:
        str: The name of the country corresponding to the given coordinates.
        If the country cannot be determined, returns 'Unknown'.
        """
        longitude, latitude = coords
        location = self.geolocator.reverse((latitude, longitude), language="en")
        return location.raw["address"].get("country", "") if location else "Unknown"

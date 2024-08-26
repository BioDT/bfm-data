# src/data_ingestion/api_clients/inaturalist.py

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.data_ingestion.api_clients.downloader import Downloader


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
        self.max_requests_per_minute = 60
        self.max_download_size_per_hour = 5 * 1024 * 1024 * 1024
        self.downloaded_size = 0
        self.last_request_time = time.time()

    def throttle_requests(self):
        """
        Throttle the requests to avoid exceeding the rate limit.
        """
        elapsed_time = time.time() - self.last_request_time
        if elapsed_time < 1.0 / self.max_requests_per_minute:
            time.sleep(1.0 / self.max_requests_per_minute - elapsed_time)
        self.last_request_time = time.time()

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

        self.throttle_requests()

        params = {
            "per_page": 200,
            "has[]": ["photos", "geo"],
            "quality_grade": "research",
            "page": page,
        }

        json_response = self.get_base_url_page(params)

        total_results = json_response.get("total_results", 0)
        num_pages = (total_results // json_response.get("per_page", 200)) + (
            1 if total_results % json_response.get("per_page", 200) != 0 else 0
        )

        observations = json_response["results"]

        self.process_and_download_observations(observations)

        if page < num_pages:
            self.get_observations(page + 1)

    def process_and_download_observations(
        self, observations: List[Dict[str, Any]]
    ) -> None:
        """
        Process and download the observations using parallel threads.

        Args:
            observations (list): List of observations.
        """
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for observation in observations:
                futures.append(
                    executor.submit(self.process_single_observation, observation)
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing observation: {e}")

    def process_single_observation(self, observation: Dict[str, Any]) -> None:
        """
        Process and download data for a single observation.

        Args:
            observation (dict): The observation data.
        """
        taxonomy = self.extract_taxonomy(observation)
        timestamp = observation.get("time_observed_at", "Unknown")
        taxon = observation.get("taxon", {})
        observation_id = observation.get("id", "Unknown")
        preferred_common_name = taxon.get("preferred_common_name", "Unknown")
        scientific_name = taxon.get("name", "Unknown")

        for photo_counter, photo in enumerate(observation.get("photos", []), start=1):
            photo_url = self.format_photo_url(photo)
            if not photo_url:
                continue

            image_name, csv_path = self.construct_paths(
                scientific_name, preferred_common_name, observation_id, photo_counter
            )

            if os.path.exists(image_name):
                continue

            species_data = self.prepare_species_data(
                observation, taxon, taxonomy, photo_url, timestamp
            )

            self.throttle_requests()
            image_size = self.download_file(photo_url, image_name)

            # Only update downloaded size if the image_size is valid
            if image_size is not None:
                self.downloaded_size += image_size

            # Check if the download size exceeds the hourly limit
            if self.downloaded_size >= self.max_download_size_per_hour:
                print("Download limit reached for the hour.")
                return

            self.save_to_csv(species_data, csv_path)

    def extract_taxonomy(self, observation: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract taxonomy details from the observation.

        Args:
            observation (dict): The observation data.

        Returns:
            dict: Dictionary of taxonomy ranks.
        """
        taxonomy_ranks = ["kingdom", "phylum", "class", "order", "family", "genus"]
        taxonomy = {}

        ancestors = (
            observation.get("identifications", [{}])[0]
            .get("taxon", {})
            .get("ancestors", [])
        )
        for ancestor in ancestors:
            rank = ancestor.get("rank", "")
            name = ancestor.get("name", "")
            if rank in taxonomy_ranks:
                taxonomy[rank] = name

        return taxonomy

    def format_photo_url(self, photo: Dict[str, Any]) -> str:
        """
        Format the photo URL to a usable medium size.

        Args:
            photo (dict): Photo information.

        Returns:
            str: Formatted photo URL.
        """
        photo_url = photo.get("url", "").replace("square", "medium")
        return photo_url if photo_url.startswith("https://") else ""

    def construct_paths(
        self,
        scientific_name: str,
        common_name: str,
        observation_id: str,
        photo_counter: int,
    ) -> Tuple[str, str]:
        """
        Construct file paths for images and CSV files.

        Args:
            scientific_name (str): Scientific name of the species.
            common_name (str): Common name of the species.
            observation_id (str): Observation ID.
            photo_counter (int): Photo counter.

        Returns:
            tuple: Image path and CSV path.
        """
        scientific_name_path = os.path.join(self.base_path, scientific_name)
        Path(scientific_name_path).mkdir(parents=True, exist_ok=True)

        image_name = os.path.join(
            scientific_name_path,
            f"{common_name}_{observation_id}_{photo_counter}.jpg",
        )
        csv_path = os.path.join(scientific_name_path, f"{common_name}_image.csv")

        return image_name, csv_path

    def prepare_species_data(
        self,
        observation: Dict[str, Any],
        taxon: Dict[str, Any],
        taxonomy: Dict[str, str],
        photo_url: str,
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        """
        Prepare the species data for saving.

        Args:
            observation (dict): The observation data.
            taxon (dict): The taxon data.
            taxonomy (dict): The taxonomy information.
            photo_url (str): The URL of the photo.
            timestamp (str): Timestamp of the observation.

        Returns:
            list: A list containing a dictionary of species data.
        """
        return [
            {
                "Observation_id": observation.get("id", "Unknown"),
                "Common_name": taxon.get("preferred_common_name", "Unknown"),
                "id": taxon.get("id", "Unknown"),
                "Kingdom": taxonomy.get("kingdom", "Unknown"),
                "Phylum": taxonomy.get("phylum", "Unknown"),
                "Class": taxonomy.get("class", "Unknown"),
                "Order": taxonomy.get("order", "Unknown"),
                "Family": taxonomy.get("family", "Unknown"),
                "Genus": taxonomy.get("genus", "Unknown"),
                "Scientific_name": taxon.get("name", "Unknown"),
                "Place": observation.get("place_guess", "Unknown"),
                "Coordinates": observation.get("geojson", {}).get("coordinates", []),
                "Photo_url": photo_url,
                "Photo_dimensions": "375x500",
                "timestamp": timestamp,
            }
        ]

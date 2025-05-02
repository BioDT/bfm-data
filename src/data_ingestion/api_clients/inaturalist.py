"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
from cachetools import TTLCache

from src.config.paths import DATA_DIR
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
        self.downloaded_ids = set()
        self.load_downloaded_ids()
        self.per_page_limit = 200
        self.page_limit = 50
        self.semaphore = asyncio.Semaphore(10)
        self.cache = TTLCache(maxsize=1000, ttl=3600)

    def load_downloaded_ids(self):
        """
        Load the IDs of already downloaded observations from a file.
        """
        ids_file = os.path.join(self.data_dir, "downloaded_ids.json")
        if os.path.exists(ids_file):
            with open(ids_file, "r") as file:
                self.downloaded_ids = set(json.load(file))

    def save_downloaded_ids(self):
        """
        Save the IDs of downloaded observations to a file.
        """
        ids_file = os.path.join(self.data_dir, "downloaded_ids.json")
        with open(ids_file, "w") as file:
            json.dump(list(self.downloaded_ids), file)

    def throttle_requests(self):
        """
        Throttle the requests to avoid exceeding the rate limit.
        """
        elapsed_time = time.time() - self.last_request_time
        if elapsed_time < 1.0 / self.max_requests_per_minute:
            time.sleep(1.0 / self.max_requests_per_minute - elapsed_time)
        self.last_request_time = time.time()

    async def fetch_observations(
        self, session: aiohttp.ClientSession, id_above: int = None, start_page: int = 1
    ) -> list:
        """
        Fetch observations from iNaturalist.

        Args:
            id_above (int): ID above which to fetch observations.
            start_page (int): Page number for paginated API results. Default is 1.

        Returns:
            list: List of observations.
        """
        observations = []
        page = start_page
        try:
            while True:
                async with self.semaphore:
                    self.throttle_requests()

                    # per_page (int): Allowed values: 1 to 200
                    # has[] (list): Catch-all for some boolean selectors. (photos) - only show observations with photos.
                    #                                                     (geo) - only show georeferenced observations
                    # quality_grade (str) = 'research' / 'casual'

                    params = {
                        "per_page": self.per_page_limit,
                        "has[]": ["photos", "geo"],
                        "quality_grade": "research",
                    }

                    if id_above:
                        params["id_above"] = id_above
                    else:
                        params["page"] = page

                    cache_key = (page, id_above)
                    if cache_key in self.cache:
                        json_response = self.cache[cache_key]

                    else:
                        async with session.get(
                            self.base_url, params=params
                        ) as response:
                            json_response = await response.json()
                            self.cache[cache_key] = json_response

                    new_observations = json_response.get("results", [])

                    if not new_observations:
                        return

                    await self.process_and_download_observations(new_observations)

                    observations.extend(new_observations)

                    if page is not None and page >= self.page_limit:
                        id_above = new_observations[-1]["id"]
                        page = None
                        continue

                    if id_above:
                        id_above = new_observations[-1]["id"]
                    else:
                        page += 1

        except Exception as e:
            print(f"Error occurred while fetching observations: {e}")
            return []

    async def process_and_download_observations(
        self, observations: List[Dict[str, Any]]
    ) -> None:
        """
        Process and download the observations using parallel threads.

        Args:
            observations (list): List of observations.
        """
        batch_size = 20
        for i in range(0, len(observations), batch_size):
            batch = observations[i : i + batch_size]
            await self.process_batch(batch)

    async def process_batch(self, batch: List[Dict[str, Any]]) -> None:
        with ThreadPoolExecutor(max_workers=5) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor, self.process_single_observation, observation
                )
                for observation in batch
                if observation["id"] not in self.downloaded_ids
            ]
            await asyncio.gather(*tasks)

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

        self.downloaded_ids.add(observation["id"])
        self.save_downloaded_ids()

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

            if image_size is not None:
                self.downloaded_size += image_size

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
            f"{str(common_name).replace('/', '_')}_{observation_id}_{photo_counter}.jpg",
        )
        csv_path = os.path.join(
            scientific_name_path,
            f"{str(common_name).replace('/', '_')}_{observation_id}_{photo_counter}_image.csv",
        )

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
        coordinates = observation.get("geojson", {}).get("coordinates", [])
        if len(coordinates) == 2:
            lon, lat = coordinates
        else:
            lon, lat = None, None

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
                "Location": observation.get("place_guess", "Unknown"),
                "Longitude": lon,
                "Latitude": lat,
                "Timestamp": timestamp,
                "Photo_url": photo_url,
                "Photo_dimensions": "375x500",
            }
        ]

    async def run(self):
        async with aiohttp.ClientSession() as session:
            await self.fetch_observations(session, start_page=1)


def inaturalist():
    """
    Function to initialize iNaturalistDownloader and start the observation download process.
    """
    downloader = iNaturalistDownloader(DATA_DIR)
    asyncio.run(downloader.run())

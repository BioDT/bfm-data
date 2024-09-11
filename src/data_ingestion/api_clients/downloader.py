# src/data_ingestion/api_clients/downloader.py

import csv
import os

import requests


class Downloader:
    """
    A class to handle downloading files from the internet and saving them to a specified directory.
    """

    def __init__(self, data_dir, source, base_url=None):
        """
        Initialize the Downloader with a directory for saving downloaded files.

        Args:
            data_dir (str): Directory path for storing downloaded files.
            source (str): source for organizing data.
            country (str): Country name for organizing data.
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.base_url = base_url
        self.source = source
        self.base_path = os.path.join(data_dir, source)

    def get_base_url_page(self, params: dict):
        """
        Fetch a page from the a source's API.

        Args:
            params (dict): The parameters for the API request.

        Returns:
            dict: The JSON response from the source's API containing the data.
        """
        try:
            request = requests.Request("GET", url=self.base_url, params=params)
            prepared_request = request.prepare()

            with requests.Session() as session:
                response = session.send(prepared_request)
                response.raise_for_status()

                print(f"Response Status Code: {response.status_code}")
                print(f"Response Content (raw): {response.content}")

                try:
                    return response.json()
                except requests.exceptions.JSONDecodeError:
                    print("Failed to decode JSON from the response.")
                    return None
        except requests.exceptions.RequestException:
            return None

    def save_to_csv(self, data: list, filename: str):
        """
        Save data to a CSV file.

        Parameters:
            data (list): The list of data to be saved.
            filename (str): The path including the filename where the CSV will be saved.

        """

        keys = data[0].keys()

        if os.path.exists(filename):
            with open(filename, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing_data = list(reader)
        else:
            existing_data = []

        def make_hashable(item):
            if isinstance(item, list):
                return tuple(make_hashable(i) for i in item)
            elif isinstance(item, dict):
                return tuple((k, make_hashable(v)) for k, v in item.items())
            return item

        data_ = [make_hashable(item) for item in data]
        existing_data_ = [make_hashable(item) for item in existing_data]

        unique_data = set(existing_data_)

        for item in data_:
            if item not in unique_data:
                unique_data.add(item)
                existing_data.append(dict(item))

        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(existing_data)

    def download_file(self, url: str, file_path: str) -> int:
        """
        Download a file from the given URL and save it to the specified path.

        Args:
            url (str): The URL of the file to download.
            file_path (str): The local file path where the file will be saved.

        Returns:
            int: The size of the downloaded file in bytes. Returns 0 if the download fails.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(file_path, "wb") as file:
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        total_size += len(chunk)

            return total_size
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return 0

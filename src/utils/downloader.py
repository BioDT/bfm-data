# src/utils/downloader.py

import csv
import os
import shutil

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

    def get_base_url_page(self, params: str):
        """
        Fetch a page from the a source's API.

        Args:
            query (str): The query string to search for specific data.

        Returns:
            dict: The JSON response from the source's API containing the data.
        """
        request = requests.Request("GET", url=self.base_url, params=params)
        prepared_request = request.prepare()

        print(prepared_request.url)

        session = requests.Session()
        response = session.send(prepared_request)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

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

        data_ = [tuple(item.items()) for item in data]
        existing_data_ = [tuple(item.items()) for item in existing_data]

        unique_data = set(existing_data_)

        for item in data_:
            if item not in unique_data:
                unique_data.add(item)
                existing_data.append(dict(item))

        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(existing_data)

    def download_file(self, url: str, filename: str):
        """
        Download a file from the given URL and save it to the specified filename.

        Args:
            url (str): The URL to download the file from.
            filename (str): The name of the file to save the downloaded content.
        """
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, "wb") as file:
                shutil.copyfileobj(response.raw, file)
        else:
            response.raise_for_status()
        response.close()

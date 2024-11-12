# src/data_ingestion/api_clients/copernicus_land.py

import os
import time

import requests

from src.config import paths
from src.helpers.clms_api_config import CopernicusLandConfigurator


class CopernicusLandApiDownloader:
    """
    A class to download data from the Copernicus Land Monitoring System API.
    """

    def __init__(self, configurator: CopernicusLandConfigurator):
        self.configurator = configurator
        self.access_token = None

    def search_datasets(self, search_term: str, access_token: str) -> list:
        """
        Searches for datasets matching the search_term and handles pagination.
        """
        api_url = f"https://land.copernicus.eu/api/@search?portal_type=DataSet&SearchableText={search_term}&b_size=100"
        datasets = []

        while api_url:
            response = self.configurator.make_authenticated_request(
                api_url, access_token
            )
            if response.status_code == 200:
                data = response.json()
                datasets.extend(data["items"])

                api_url = data.get("batching", {}).get("next", None)
            else:
                print(f"Failed to request data. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                break

        return datasets

    def request_data(
        self,
        dataset_id: str,
        datasetdownloadinfoid: str,
        start_date: str,
        end_date: str,
        bbox: list = None,
    ):
        """
        Request data from the Copernicus API using spatial and temporal constraints.

        Args:
            dataset_id (str): The ID of the dataset you want to download.
            datasetdownloadinfoid (str): The information ID of the dataset you want to download.
            bbox (list): The bounding box to restrict the spatial extent (optional).
            start_date (str): The start date of the data request in format 'YYYY-MM-DD'.
            end_date (str): The end date of the data request in format 'YYYY-MM-DD'.
            frequency (str): The temporal frequency of the data (monthly in this case).
        """
        url = "https://land.copernicus.eu/api/@datarequest_post"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        start_epoch = int(time.mktime(time.strptime(start_date, "%Y-%m-%d")) * 1000)
        end_epoch = int(time.mktime(time.strptime(end_date, "%Y-%m-%d")) * 1000)

        json = {
            "Datasets": [
                {
                    "DatasetID": dataset_id,
                    "DatasetDownloadInformationID": datasetdownloadinfoid,
                    "OutputFormat": "Netcdf",
                    "OutputGCS": "EPSG:4326",
                    "TemporalFilter": {"StartDate": start_epoch, "EndDate": end_epoch},
                }
            ]
        }

        if bbox:
            json["Datasets"][0]["BoundingBox"] = bbox

        response = requests.post(url, headers=headers, json=json)
        if response.status_code == 201:
            data_request_id = response.json().get("TaskIds")[0].get("TaskID")
            print(f"Data request submitted successfully. Task ID: {data_request_id}")
            return data_request_id
        else:
            print(f"Failed to request data. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    def get_download_url(self, request_id: str):
        """
        Get the download URL for the data request.
        """
        url = f"https://land.copernicus.eu/api/@@datarequest_search?id={request_id}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            download_url = data.get("download_url")
            if download_url:
                print(f"Download URL: {download_url}")
                return download_url
            else:
                print("Download is not ready yet.")
                return None
        else:
            print(
                f"Failed to fetch the download URL. Status code: {response.status_code}"
            )
            return None

    def download_dataset(self, dataset_url: str, save_path: str):
        """
        Downloads a dataset using its download URL and saves it to the specified path.
        """
        response = requests.get(dataset_url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as file:
                file.write(response.content)
            print(f"Dataset downloaded successfully: {save_path}")
        else:
            print(
                f"Failed to download the dataset. Status Code: {response.status_code}"
            )
            print(f"Response: {response.text}")

    def run(
        self,
        key_file: str,
        search_term: str,
        dataset_title: str,
        dataset_id: str,
        dataset_download_info_id: str,
        save_dir: str,
        bbox: list,
        start_date: str,
        end_date: str,
    ):
        """
        Runs the process of loading the service key, obtaining an access token, searching for datasets,
        and downloading the specific dataset based on the provided title.
        """
        service_key = self.configurator.load_service_key(key_file)
        jwt_token = self.configurator.create_jwt_token(service_key)
        self.access_token = self.configurator.get_access_token(service_key, jwt_token)

        if self.access_token:
            datasets = self.search_datasets(
                search_term,
                self.access_token,
            )

            if datasets:
                print(f"Found {len(datasets)} datasets.")
                dataset_to_download = None
                for dataset in datasets:
                    print(f"Dataset Title: {dataset['title']}")
                    if (
                        dataset["title"].strip().lower()
                        == dataset_title.strip().lower()
                    ):
                        dataset_to_download = dataset
                        break

                if dataset_to_download:
                    if dataset_id and dataset_download_info_id:
                        request_id = self.request_data(
                            dataset_id,
                            dataset_download_info_id,
                            bbox=bbox,
                            start_date=start_date,
                            end_date=end_date,
                        )

                        if request_id:
                            print(request_id)
                            download_url = None
                            while not download_url:
                                download_url = self.get_download_url(request_id)

                            if download_url:
                                save_path = os.path.join(
                                    save_dir,
                                    f"{dataset_to_download['title'].replace(' ', '_')}.zip",
                                )
                                self.download_dataset(download_url, save_path)
                        else:
                            print(
                                f"Failed to submit a data request for dataset: {dataset_title}"
                            )
                    else:
                        print(
                            f"Dataset ID not found for {dataset_to_download['title']}"
                        )
                else:
                    print(f"No dataset found matching the title: {dataset_title}")
            else:
                print(f"No dataset found for the search term: {search_term}")
        else:
            print("Failed to obtain an access token")


def CopernicusLand():
    """
    Run the CopernicusLand for getting the access token and to authenticate requests.
    """
    configurator = CopernicusLandConfigurator()

    downloader = CopernicusLandApiDownloader(configurator)

    bbox = [2.354736328128108, 46.852958688910306, 4.639892578127501, 45.88264619696234]
    downloader.run(
        "/home/stylianos.stasinos/bfm-data/src/config/copernicus_land_config/key.json",
        search_term="Normalised Difference Vegetation Index",
        dataset_title="Normalised Difference Vegetation Index 1999-2020 (raster 1 km), global, 10-daily – version 3",
        dataset_id="7714f261ebe64372bef240232aa5219a",
        dataset_download_info_id="bc49eff5-8e94-49ae-ac5a-5ad9a8626970",
        save_dir=paths.NDVI_DIR,
        bbox=bbox,
        start_date="2000-01-01",
        end_date="2000-01-02",
    )

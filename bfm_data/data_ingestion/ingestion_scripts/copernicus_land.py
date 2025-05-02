"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import csv
import os
from concurrent.futures import ThreadPoolExecutor

import netCDF4 as nc
import numpy as np
import requests
import reverse_geocode

from bfm_data.config import paths
from bfm_data.utils.geo import (
    get_bounding_boxes_for_countries,
    get_countries_by_continent,
    get_country_name_from_iso,
)


class CopernicusLandDownloader:
    """
    A class to handle downloading and extracting NDVI data for specific vegetation locations.
    """

    def __init__(
        self,
        links_url: str,
        data_dir: str,
        csv_file: str,
        region: str = None,
        ndvi_threshold: float = 0.2,
        global_mode: bool = False,
    ):
        """
        Initializes the downloader with a manifest URL, save directory, and vegetation threshold.

        Args:
            links_url (str): The URL to the file containing download links.
            data_dir (str): The directory where the downloaded files will be saved.
            csv_file (str): The path to the output CSV file where results will be stored.
            region (str): The region name to fetch bounding boxes.
            ndvi_threshold (float): NDVI threshold to define vegetated locations.
            global_mode (bool): Flag to indicate if global processing is enabled.
        """
        self.links_url = links_url
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.ndvi_threshold = ndvi_threshold
        self.global_mode = global_mode
        self.country_rectangles = {}

        if region and not global_mode:
            self.load_region_bounding_boxes(region)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_region_bounding_boxes(self, region: str):
        """
        Load country bounding boxes for a specified region using utility functions.

        Args:
            region (str): The region name (e.g., 'Europe', 'Latin America').
        """
        _, iso_codes = get_countries_by_continent(region)
        self.country_rectangles = get_bounding_boxes_for_countries(iso_codes)

    def filter_11th_day_files(self, file_urls: list) -> list:
        """
        Filters the list of URLs to get only the file for the 11th of each month.

        Args:
            file_urls (list): List of URLs to filter.

        Returns:
            list: Filtered list containing only files for the 11th of each month.
        """
        eleventh_day_files = []
        for url in file_urls:

            year_month_day_part = url.split("/")[-2]

            if year_month_day_part.endswith("11"):
                eleventh_day_files.append(url)

        return eleventh_day_files

    def download_files(self, file_urls: list):
        """
        Downloads files from the provided URLs and saves them to the save directory.

        Args:
            file_urls (list): List of file URLs to download.
        """
        for file_url in file_urls:
            file_name = os.path.basename(file_url)
            file_path = os.path.join(self.data_dir, file_name)

            if os.path.exists(file_path):
                print(f"File already exists: {file_name}. Skipping download.")
                continue

            file_response = requests.get(file_url)
            if file_response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(file_response.content)
                print(f"Downloaded: {file_name}")
            else:
                print(f"Failed to download: {file_url}")

    def download_files_per_year(self):
        """
        Downloads only the first January files (based on the manifest URL).
        """
        response = requests.get(self.links_url)
        if response.status_code == 200:
            file_urls = response.text.splitlines()
            january_files = self.filter_11th_day_files(file_urls)

            self.download_files(january_files)
        else:
            print(f"Failed to fetch the manifest file: {self.links_url}")

    def extract_ndvi_locations(self, nc_file_path: str):
        """
        Extract NDVI values and filter for vegetation locations from a NetCDF file.

        Args:
            nc_file_path (str): Path to the NetCDF file.
        """
        try:
            dataset = nc.Dataset(nc_file_path, "r")
            lat = dataset.variables["lat"][:]
            lon = dataset.variables["lon"][:]
            ndvi = dataset.variables["NDVI"][0, :, :]
            time_var = dataset.variables["time"]
            month_year = nc.num2date(time_var[0], time_var.units).strftime("%m/%Y")

            lat_points = np.arange(90, -90 - 0.25, -0.25)
            lon_points = np.arange(-180, 180 + 0.25, 0.25)

            if self.global_mode:
                ndvi_data = {}

                for lat_val in lat_points:
                    for lon_val in lon_points:
                        i = np.abs(lat - lat_val).argmin()
                        j = np.abs(lon - lon_val).argmin()
                        ndvi_value = ndvi[i, j]

                        if ndvi_value != 255 and ndvi_value > self.ndvi_threshold:
                            transformed_lon = lon_val if lon_val >= 0 else lon_val + 360
                            coord = (lat_val, transformed_lon)
                            country = reverse_geocode.get(coord)["country"]
                            if country not in ndvi_data:
                                ndvi_data[country] = []
                            ndvi_data[country].append(
                                (lat_val, transformed_lon, ndvi_value)
                            )

            else:
                ndvi_data = {
                    get_country_name_from_iso(country): []
                    for country in self.country_rectangles.keys()
                }

                for country_iso, bbox in self.country_rectangles.items():
                    min_lon, min_lat, max_lon, max_lat = bbox

                    for lat_val in lat_points:
                        for lon_val in lon_points:
                            if min_lon > max_lon:
                                in_lon_range = lon_val >= min_lon or lon_val <= max_lon
                            else:
                                in_lon_range = min_lon <= lon_val <= max_lon

                            if min_lat <= lat_val <= max_lat and in_lon_range:
                                i = np.abs(lat - lat_val).argmin()
                                j = np.abs(lon - lon_val).argmin()
                                ndvi_value = ndvi[i, j]

                                if (
                                    ndvi_value != 255
                                    and ndvi_value > self.ndvi_threshold
                                ):
                                    transformed_lon = (
                                        lon_val if lon_val >= 0 else lon_val + 360
                                    )
                                    country_name = get_country_name_from_iso(
                                        country_iso
                                    )
                                    ndvi_data[country_name].append(
                                        (lat_val, transformed_lon, ndvi_value)
                                    )

            dataset.close()
            return month_year, ndvi_data

        except Exception as e:
            print(f"Error processing the file {nc_file_path}: {e}")
            return None, []

    def update_csv(self, month_year: str, ndvi_data: list):
        """
        Update the CSV file with NDVI data for each country based on vegetation points.

        Args:
            month_year (str): The month and the year of the data.
            ndvi_data (list): A list of tuples with (latitude, longitude, NDVI value).
        """
        file_exists = os.path.exists(self.csv_file)
        existing_data = {}
        fieldnames = ["Country", "Latitude", "Longitude"]

        if file_exists:
            with open(self.csv_file, "r", newline="") as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames

                if fieldnames:
                    year_columns = sorted(
                        [col for col in fieldnames if col.startswith("NDVI_")]
                    )
                    fieldnames = ["Country", "Latitude", "Longitude"] + year_columns

                for row in reader:
                    key = (row["Country"], row["Latitude"], row["Longitude"])
                    existing_data[key] = row

        ndvi_column = f"NDVI_{month_year}"
        if ndvi_column not in fieldnames:
            fieldnames.append(ndvi_column)

        for country, points in ndvi_data.items():
            for lat, lon, ndvi_value in points:
                key = (country, f"{lat:.10f}", f"{lon:.10f}")
                if key not in existing_data:
                    existing_data[key] = {
                        "Country": country,
                        "Latitude": f"{lat:.10f}",
                        "Longitude": f"{lon:.10f}",
                        ndvi_column: f"{ndvi_value:.4f}",
                    }
                else:
                    existing_data[key][ndvi_column] = f"{ndvi_value:.4f}"

        fieldnames = sorted(fieldnames, key=lambda x: (x.startswith("NDVI_"), x))

        with open(self.csv_file, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for row in existing_data.values():
                writer.writerow(row)

    def process_files(self):
        """
        Process all NetCDF files in the directory and extract NDVI values.
        """
        existing_months = set()
        if os.path.exists(self.csv_file):
            with open(self.csv_file, "r", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames:
                    existing_months = {
                        col.split("_")[1]
                        for col in reader.fieldnames
                        if col.startswith("NDVI_")
                    }

        nc_files = [f for f in os.listdir(self.data_dir) if f.endswith(".nc")]

        for nc_file in nc_files:
            file_path = os.path.join(self.data_dir, nc_file)
            try:
                dataset = nc.Dataset(file_path, "r")
                time_var = dataset.variables["time"]
                month_year = nc.num2date(time_var[0], time_var.units).strftime("%m/%Y")
                dataset.close()

                if month_year in existing_months:
                    print(
                        f"Skipping file {nc_file} as month_year {month_year} already exists in the CSV."
                    )
                    continue

                month_year, ndvi_data = self.extract_ndvi_locations(file_path)
                if ndvi_data:
                    self.update_csv(month_year, ndvi_data)
                    print(f"Updated CSV for file: {nc_file}, month_year: {month_year}")

            except Exception as e:
                print(f"Error processing the file {nc_file}: {e}")


def run_data_download(
    download: bool = False,
    process: bool = False,
    global_mode: bool = False,
    region: str = None,
):
    """
    Main function to trigger the download of files from the manifest URL and also to process the files.

    Args:
        download (bool): Whether to download data files.
        process (bool): Whether to process data files.
        global_mode (bool): If True, process in global mode.
        region (str): If provided, process based on the specified region.
    """
    links_url = "https://globalland.vito.be/download/manifest/ndvi_1km_v3_10daily_netcdf/manifest_clms_global_ndvi_1km_v3_10daily_netcdf_latest.txt"
    data_dir = paths.NDVI_DIR
    land_dir = paths.LAND_DIR

    if global_mode:
        csv_file = f"{land_dir}/global_ndvi.csv"
    elif region:
        region_cleaned = region.replace(" ", "_")
        csv_file = f"{land_dir}/{region_cleaned}_ndvi_test.csv"
    else:
        csv_file = f"{land_dir}/default_ndvi.csv"

    downloader = CopernicusLandDownloader(
        links_url=links_url,
        data_dir=data_dir,
        csv_file=csv_file,
        region=region,
        global_mode=global_mode,
    )

    if download:
        downloader.download_files_per_year()
    if process:
        downloader.process_files()

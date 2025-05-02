"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import csv
import os

import numpy as np

from src.utils.geo import (
    get_bounding_boxes_for_countries,
    get_countries_by_continent,
    get_country_bounding_box,
    get_country_name_from_iso,
)


class WorldBankDataProcessor:
    """
    A class to process and store World Bank data (Agriculture, Forest, Land) for each country.
    """

    def __init__(
        self,
        data_file: str,
        output_csv: str,
        region: str = None,
        global_mode: bool = False,
    ):
        """
        Initialize the processor with the data file and the output CSV path.

        Args:
            data_file (str): Path to the CSV file containing data.
            output_csv (str): Path to the output CSV file for processed data.
            region (str): The region for bounding box selection (e.g., 'Europe').
            global_mode (bool): Flag for global processing.
        """
        self.data_file = data_file
        self.output_csv = output_csv
        self.global_mode = global_mode
        self.country_rectangles = {}

        if region and not global_mode:
            self.load_region_bounding_boxes(region)

        if not os.path.exists(os.path.dirname(self.output_csv)):
            os.makedirs(os.path.dirname(self.output_csv))

    def load_region_bounding_boxes(self, region: str):
        """
        Load bounding boxes for countries in a specific region.

        Args:
            region (str): The region name for filtering (e.g., 'Europe').
        """
        _, iso_codes = get_countries_by_continent(region)
        self.country_rectangles = get_bounding_boxes_for_countries(iso_codes)

    def read_data(self) -> dict:
        """
        Reads data from a CSV file and organizes it by country and year.

        Returns:
            dict: A dictionary where keys are country names, and values are another
                dictionary with year as keys and data values as values.
        """
        data = {}
        with open(self.data_file, "r") as file:
            reader = csv.reader(file)

            for row in reader:
                if row and row[0] == "Country Name":
                    headers = row
                    break

            for row in reader:
                if row and row[0]:
                    country = row[headers.index("Country Name")]
                    for year, value in zip(headers[4:], row[4:]):
                        if country not in data:
                            data[country] = {}
                        if value:
                            data[country][year] = float(value)
        return data

    def generate_country_grid(self, bbox: tuple, degree_step: float = 0.25) -> list:
        """
        Generate grid points for a given bounding box with a fixed 0.25-degree interval,
        from 90 to -90 for latitude and from 0 to 360 for longitude.

        Args:
            bbox (tuple): Bounding box (min_lon, min_lat, max_lon, max_lat).
            degree_step (float): Step size for grid.

        Returns:
            list: A list of (lat, lon) points within the bounding box.
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        if min_lon < 0:
            min_lon += 360
        if max_lon < 0:
            max_lon += 360

        lat_points = np.arange(90, -90 - degree_step, -degree_step)
        lon_points = np.arange(0, 360 + degree_step, degree_step)

        lat_points = lat_points[(lat_points >= min_lat) & (lat_points <= max_lat)]

        if min_lon > max_lon:
            lon_points = np.concatenate(
                [
                    lon_points[(lon_points >= min_lon)],
                    lon_points[(lon_points <= max_lon)],
                ]
            )
        else:
            lon_points = lon_points[(lon_points >= min_lon) & (lon_points <= max_lon)]

        if len(lat_points) == 0 or len(lon_points) == 0:
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            return [(center_lat, center_lon)]

        return [(lat, lon) for lat in lat_points for lon in lon_points]

    def process_data(self, value_key: str):
        """
        Processes data to average values over grid cells. If `global_mode` is True,
        it processes each country individually from the file, getting each country's bounding box.

        Args:
            value_key (str): The key prefix (e.g., "Agri", "Forest", "Land") for output CSV columns.
        """
        data = self.read_data()
        processed_data = {}

        if self.global_mode:
            for country in data.keys():
                bbox = get_country_bounding_box(country)
                if bbox:
                    grid_points = self.generate_country_grid(bbox)
                    processed_data[country] = []

                    for point in grid_points:
                        lat, lon = point
                        for year, value in data[country].items():
                            processed_data[country].append(
                                {
                                    "Latitude": lat,
                                    "Longitude": lon,
                                    "Year": year,
                                    value_key: value,
                                }
                            )
        else:
            for country_iso, bbox in self.country_rectangles.items():
                grid_points = self.generate_country_grid(bbox)

                country = get_country_name_from_iso(country_iso)
                processed_data[country] = []

                for point in grid_points:
                    lat, lon = point
                    if country in data:
                        for year, value in data[country].items():
                            processed_data[country].append(
                                {
                                    "Latitude": lat,
                                    "Longitude": lon,
                                    "Year": year,
                                    value_key: value,
                                }
                            )

        self.update_csv(processed_data, value_key)

    def update_csv(self, data: dict, value_key: str):
        """
        Write the processed data to CSV in the required format.

        Args:
            data (dict): Processed data.
            value_key (str): The key prefix (e.g., "Agri", "Forest", "Land") for CSV columns.
        """
        _ = os.path.exists(self.output_csv)
        existing_data = {}
        fieldnames = ["Country", "Latitude", "Longitude"]

        all_years = sorted(
            set(data["Year"] for country_data in data.values() for data in country_data)
        )
        for year in all_years:
            fieldnames.append(f"{value_key}_{year}")

        with open(self.output_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for country, grid_data in data.items():
                for entry in grid_data:
                    lat = f"{entry['Latitude']:.10f}"
                    lon = f"{entry['Longitude']:.10f}"
                    year = entry["Year"]
                    value = entry[value_key]

                    key = (country, lat, lon)
                    if key not in existing_data:
                        existing_data[key] = {
                            "Country": country,
                            "Latitude": lat,
                            "Longitude": lon,
                        }

                    existing_data[key][f"{value_key}_{year}"] = (
                        f"{value:.4f}" if value else ""
                    )

            for row in existing_data.values():
                writer.writerow(row)

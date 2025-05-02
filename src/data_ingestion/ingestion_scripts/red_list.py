"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import csv
import os

import numpy as np
import reverse_geocode

from src.config import paths
from src.utils.geo import (
    get_bounding_boxes_for_countries,
    get_countries_by_continent,
    get_country_name_from_iso,
)


class RedListDataProcessor:
    """
    A class to handle processing and averaging Red List Index (RLI) data for countries.

    This class reads a CSV file containing RLI values, creates 0.25-degree grid cells
    for each country's bounding box, and averages RLI data over the grid cells.
    """

    def __init__(
        self,
        rli_file: str,
        output_csv: str,
        region: str = None,
        global_mode: bool = False,
    ):
        """
        Initialize the processor with the path to the RLI file and output directory.

        Args:
            rli_file (str): Path to the input CSV file with Red List Index data.
            output_csv (str): Path to the output CSV file where processed data will be stored.
            region (str): The region name to fetch bounding boxes.
            global_mode (bool): Flag to indicate if global processing is enabled.
        """
        self.rli_file = rli_file
        self.output_csv = output_csv
        self.global_mode = global_mode
        self.country_rectangles = {}

        if region and not global_mode:
            self.load_region_bounding_boxes(region)

        if not os.path.exists(os.path.dirname(self.output_csv)):
            os.makedirs(os.path.dirname(self.output_csv))

    def load_region_bounding_boxes(self, region: str):
        """
        Load country bounding boxes for a specified region using utility functions.

        Args:
            region (str): The region name (e.g., 'Europe', 'Latin America').
        """
        _, iso_codes = get_countries_by_continent(region)
        self.country_rectangles = get_bounding_boxes_for_countries(iso_codes)

    def read_redlist_data(self) -> dict:
        """
        Reads Red List Index data from a CSV file and organizes it by country and year.

        Returns:
            dict: A dictionary where keys are country codes, and values are another
                  dictionary with year as keys and RLI values as values.
        """
        redlist_data = {}
        with open(self.rli_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                country = row["Entity"]
                year = row["Year"]
                rli_value = float(row["15.5.1 - Red List Index - ER_RSK_LST"])

                if country not in redlist_data:
                    redlist_data[country] = {}

                redlist_data[country][year] = rli_value

        return redlist_data

    def generate_global_grid(self, degree_step: float = 0.25) -> list:
        """
        Generate a global grid of latitude/longitude points with longitude from 0 to 360.

        Args:
            degree_step (float): Step size for grid generation. Default is 0.25.

        Returns:
            list: A list of tuples representing latitude and longitude grid points.
        """
        lat_points = np.arange(90, -90 - degree_step, -degree_step)
        lon_points = np.arange(0, 360 + degree_step, degree_step)

        grid_points = [(lat, lon) for lat in lat_points for lon in lon_points]
        return grid_points

    def generate_country_grid(self, bbox: tuple, degree_step: float = 0.25) -> list:
        """
        Generate a list of latitude/longitude grid points within a country's bounding box.

        Args:
            bbox (tuple): Bounding box coordinates (min_lon, min_lat, max_lon, max_lat).
            degree_step (float): Step size for grid generation. Default is 0.25.

        Returns:
            list: A list of tuples representing latitude and longitude grid points.
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

    def process_rli_data(self):
        """
        Process Red List Index data, average values over grid cells, and save the result to CSV.
        """
        redlist_data = self.read_redlist_data()
        processed_data = {}

        if self.global_mode:
            grid_points = self.generate_global_grid()
            processed_data["Global"] = []

            for point in grid_points:
                lat, lon = point
                adjusted_lon = lon - 360 if lon > 180 else lon

                nearest_country = reverse_geocode.get((lat, adjusted_lon))["country"]

                if nearest_country in redlist_data:
                    for year, rli_value in redlist_data[nearest_country].items():
                        processed_data["Global"].append(
                            {
                                "Country": nearest_country,
                                "Latitude": lat,
                                "Longitude": lon,
                                "Year": year,
                                "RLI": rli_value,
                            }
                        )
        else:
            for country_iso, bbox in self.country_rectangles.items():
                grid_points = self.generate_country_grid(bbox)

                country = get_country_name_from_iso(country_iso)
                processed_data[country] = []

                for point in grid_points:
                    lat, lon = point

                    if country in redlist_data:
                        for year, rli_value in redlist_data[country].items():
                            processed_data[country].append(
                                {
                                    "Latitude": lat,
                                    "Longitude": lon,
                                    "Year": year,
                                    "RLI": rli_value,
                                }
                            )

        self.update_csv(processed_data)

    def update_csv(self, rli_data: dict):
        """
        Update the CSV file with Red List Index data for each grid point.

        Args:
            rli_data (dict): Dictionary with Red List Index data by country and grid point.
        """
        _ = os.path.exists(self.output_csv)
        existing_data = {}
        fieldnames = ["Country", "Latitude", "Longitude"]

        all_years = sorted(
            set(
                data["Year"]
                for country_data in rli_data.values()
                for data in country_data
            )
        )
        for year in all_years:
            fieldnames.append(f"RLI_{year}")

        with open(self.output_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for country, grid_data in rli_data.items():
                for data in grid_data:
                    lat = f"{data['Latitude']:.10f}"
                    lon = f"{data['Longitude']:.10f}"
                    year = data["Year"]
                    rli_value = data["RLI"]

                    key = (country, lat, lon)
                    if key not in existing_data:
                        existing_data[key] = {
                            "Country": country,
                            "Latitude": lat,
                            "Longitude": lon,
                        }

                    existing_data[key][f"RLI_{year}"] = (
                        f"{rli_value:.4f}" if rli_value else ""
                    )

            for row in existing_data.values():
                writer.writerow(row)


def run_redlist_data_processing(region: str = None, global_mode: bool = False):
    """
    Main function to trigger the processing of Red List Index data.

    Args:
        region (str): The region to process (e.g., 'Europe').
        global_mode (bool): Whether to run in global mode.
    """
    rli_file = paths.RED_LIST_FILE
    data_dir = paths.REDLIST_DIR

    if global_mode:
        csv_file = f"{data_dir}/global_red_list_index.csv"
    elif region:
        region_cleaned = region.replace(" ", "_")
        csv_file = f"{data_dir}/{region_cleaned}_red_list_index.csv"
    else:
        csv_file = f"{data_dir}/default_red_list_index.csv"

    processor = RedListDataProcessor(
        rli_file=rli_file, output_csv=csv_file, region=region, global_mode=global_mode
    )

    processor.process_rli_data()

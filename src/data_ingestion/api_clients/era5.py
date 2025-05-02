"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import concurrent.futures
import os
from datetime import datetime, timedelta
from itertools import islice
from typing import Iterable, Iterator, Tuple, TypeVar

import cdsapi
import pandas as pd

from src.config.paths import DATA_DIR, ERA5_DIR, TIMESTAMPS
from src.data_ingestion.api_clients.downloader import Downloader
from src.helpers.era5_api_config import ERA5ApiConfigurator

T = TypeVar("T")


class ERA5Downloader(Downloader):
    """
    Class for downloading and processing climate data from ERA5(Copernicus) hourly data.
    """

    def __init__(self, data_dir, date: str = None, time: str = None, area: str = None):
        """
        Initialize the ERA5Downloader.

        Args:
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "ERA5")
        self.date = date
        self.time = time
        self.area = area
        self.era5_credentials = ERA5ApiConfigurator()
        self.url = self.era5_credentials.url
        self.key = self.era5_credentials.key

    def build_request(self, levtype: str):
        """
        Build the request payload for the ERA5 API based on the level type.

        Args:
            levtype (str): The type of level for which data is being requested (e.g., 'single', 'ressure', 'surface').
            'single': Single Level
            'pressure': Pressure Level
            'surface': Surface Level

        Returns:
            dict: A dictionary containing the request parameters.
        """
        product_type = "reanalysis"
        format = "netcdf"

        base_request = {
            "date": self.date,
            "variable": "",
            "time": self.time,
            "product_type": product_type,
            "data_format": format,
        }

        if levtype == "pressure":
            base_request["pressure_level"] = ""

        if self.area is not None:
            base_request["area"] = self.area

        levtype_values = self.set_levtype_values(levtype)
        request = {**base_request, **levtype_values}
        return request

    def set_levtype_values(self, levtype: str):
        """
        Set specific values for different level types.

        Args:
            levtype (str): The type of level for which data is being requested.

        Returns:
            dict: A dictionary with specific parameters for the given level type.
        """

        if levtype == "single":
            return {
                "variable": [
                    "geopotential",
                    "land_sea_mask",
                    "soil_type",
                ]
            }
        elif levtype == "pressure":
            return {
                "pressure_level": [
                    "50",
                    "100",
                    "150",
                    "200",
                    "250",
                    "300",
                    "400",
                    "500",
                    "600",
                    "700",
                    "850",
                    "925",
                    "1000",
                ],
                "variable": [
                    "geopotential",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                ],
            }
        elif levtype == "surface":
            return {
                "variable": [
                    "mean_sea_level_pressure",
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
            }
        else:
            raise ValueError(f"Unsupported levtype: {levtype}")

    def get_data(self, levtype: str, start_date: str, end_date: str):
        """
        Retrieve data from the ERA5 dataset based on the specified level type.

        Args:
            levtype (str): The type of level for which data is being requested.
            start_date (str): The start date for which data is being requested.
            end_date (str): The end date for which data is being requested.
        """
        atmospheric_dataset_name = "reanalysis-era5-pressure-levels"
        single_dataset_name = "reanalysis-era5-single-levels"
        request = self.build_request(levtype)

        connection = cdsapi.Client(url=self.url, key=self.key, verify=True)

        if levtype == "single":
            connection.retrieve(
                single_dataset_name,
                request,
                f"{ERA5_DIR}/ERA5-Reanalysis-single- {start_date} - {end_date} .nc",
            )
        elif levtype == "pressure":
            connection.retrieve(
                atmospheric_dataset_name,
                request,
                f"{ERA5_DIR}/ERA5-Reanalysis-pressure- {start_date} - {end_date} .nc",
            )
        elif levtype == "surface":
            connection.retrieve(
                single_dataset_name,
                request,
                f"{ERA5_DIR}/ERA5-Reanalysis-surface- {start_date} - {end_date} .nc",
            )

    def generate_date_ranges(self, start_year: int, end_year: int):
        """
        Generate date ranges in 6-month intervals between the given start and end years.

        Args:
            start_year (int): Start year for data download.
            end_year (int): End year for data download.

        Returns:
            list of tuples: A list of (start_date, end_date) pairs for each 6-month interval.
        """
        date_ranges = []
        current_date = datetime(start_year, 1, 1)

        today = datetime.today()
        if end_year > today.year:
            end_date = today - timedelta(days=30)
            end_date = end_date.replace(day=1) - timedelta(days=1)
        elif end_year == today.year:
            end_date = today.replace(day=1) - timedelta(days=1)
        else:
            end_date = datetime(end_year, 12, 31)

        while current_date <= end_date:
            next_date = current_date + timedelta(days=182)

            if next_date > end_date:
                next_date = end_date

            date_ranges.append(
                (current_date.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d"))
            )

            current_date = next_date + timedelta(days=1)
        return date_ranges

    def chunked_iterable(
        self, iterable: Iterable[T], size: int
    ) -> Iterator[Tuple[T, ...]]:
        """
        Helper function to create chunks of a given size from an iterable.

        Args:
            iterable (Iterable): The iterable to split into chunks.
            size (int): The size of each chunk.

        Yields:
            Tuple: A tuple containing a chunk of the iterable. The last chunk may be smaller if there
                are fewer elements left than the requested chunk size.
        """
        iterator = iter(iterable)
        while True:
            chunk = tuple(islice(iterator, size))
            if not chunk:
                break
            yield chunk

    def run(
        self,
        mode: str = "range",
        start_year: int = None,
        end_year: int = None,
        batch_size: int = 5,
        time: str = "00/to/23/by/6",
    ):
        """
        Run the ERA5 data download either in a date range mode or timestamp mode.

        Args:
            mode (str): "range" for date range mode, "timestamps" for timestamp mode.
            start_year (int): The starting year for data downloads (used in "range" mode).
            end_year (int): The ending year for data downloads (used in "range" mode).
            batch_size (int): Number of requests to process at once (used in "timestamps" mode).
            time (str): The time range for the data (e.g., "00/to/23/by/6").
        """
        levels = ["pressure", "single", "surface"]

        if mode == "range":
            self.time = time

            if start_year is None or end_year is None:
                raise ValueError(
                    "start_year and end_year must be provided in range mode."
                )

            date_ranges = self.generate_date_ranges(start_year, end_year)
            for start_date, end_date in date_ranges:
                self.date = f"{start_date}/{end_date}"
                print(f"Downloading data for period: {self.date}")

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(self.get_data, level, start_date, end_date)
                        for level in levels
                    ]

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            print(f"Request completed with result: {result}")
                        except Exception as e:
                            print(f"Request generated an exception: {e}")

        elif mode == "timestamps":
            if not os.path.exists(TIMESTAMPS):
                raise FileNotFoundError(f"The CSV file {TIMESTAMPS} does not exist.")

            df = pd.read_csv(TIMESTAMPS)
            dates = df["Timestamp"].tolist()

            for date_chunk in self.chunked_iterable(dates, batch_size):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for timestamp in date_chunk:
                        self.date = timestamp
                        self.time = time
                        for level in levels:

                            file_path = f"{ERA5_DIR}/ERA5-Reanalysis-{level}-{self.date}-{self.date}.nc"

                            if os.path.exists(file_path):
                                print(
                                    f"File for {level} on {self.date} already exists. Skipping download."
                                )
                                continue

                            futures.append(
                                executor.submit(
                                    self.get_data, level, self.date, self.date
                                )
                            )

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            print(f"Request completed with result: {result}")
                        except Exception as e:
                            print(f"Request generated an exception: {e}")
                print(f"Batch of {batch_size} dates processed.")

        else:
            raise ValueError("Invalid mode. Choose either 'range' or 'timestamps'.")


def era5(
    mode: str,
    start_year: int = None,
    end_year: int = None,
    batch_size: int = 5,
    time: str = "00/to/23/by/6",
):
    """
    Function to create an ERA5Downloader object and run data downloads based on the specified mode.

    Args:
        mode (str): "range" for date range mode, "timestamps" for timestamp mode.
        start_year (int): The starting year for data downloads (used in "range" mode).
        end_year (int): The ending year for data downloads (used in "range" mode).
        batch_size (int): Number of requests to process at once (used in "timestamps" mode).
        time (str): The time range for data download (default: "00/to/23/by/6").
    """
    era5_downloader = ERA5Downloader(data_dir=DATA_DIR, area=[72, -30, 34, 50])
    era5_downloader.run(
        mode=mode,
        start_year=start_year,
        end_year=end_year,
        batch_size=batch_size,
        time=time,
    )

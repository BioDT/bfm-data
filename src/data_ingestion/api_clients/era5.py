# src/data_ingestion/api_clients/era5.py

import cdsapi

from src.config.paths import ERA5_DIR
from src.data_ingestion.api_clients.downloader import Downloader
from src.helpers.era5_api_config import ERA5ApiConfigurator


class ERA5Downloader(Downloader):
    """
    Class for downloading and processing climate data from ERA5(Copernicus) hourly data.
    """

    def __init__(self, data_dir, date: str, time: str, area: str = None):
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
                f"{ERA5_DIR}/ERA5-Reanalysis-single-{start_date}-{end_date}.nc",
            )
        elif levtype == "pressure":
            connection.retrieve(
                atmospheric_dataset_name,
                request,
                f"{ERA5_DIR}/ERA5-Reanalysis-pressure-{start_date}-{end_date}.nc",
            )
        elif levtype == "surface":
            connection.retrieve(
                single_dataset_name,
                request,
                f"{ERA5_DIR}/ERA5-Reanalysis-surface-{start_date}-{end_date}.nc",
            )

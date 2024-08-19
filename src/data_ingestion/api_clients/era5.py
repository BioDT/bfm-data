# src/data_ingestion/api_clients/era5.py

import cdsapi
import xarray as xr

from src.data_ingestion.api_clients.downloader import Downloader
from src.helpers.era5_api_config import ERA5ApiConfigurator


class ERA5Downloader(Downloader):
    """
    Class for downloading and processing climate data from ERA5(Copernicus) hourly data.
    """

    def __init__(self, data_dir, date: str, time: str, area: str):
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
            levtype (str): The type of level for which data is being requested (e.g., 'ml', 'pl', 'sfc').
            'ml': Single Level
            'pl': Pressure Level
            'sfc': Land/Surface Level

        Returns:
            dict: A dictionary containing the request parameters.
        """
        stream = "oper"
        type = "an"
        grid = "1.0/1.0"
        format = "netcdf"

        base_request = {
            "date": self.date,
            "levtype": "",
            "levelist": "",
            "param": "",
            "stream": stream,
            "time": self.time,
            "type": type,
            "area": self.area,
            "grid": grid,
            "format": format,
        }

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

        if levtype == "ml":
            """
            129: Geopotential, , 130: Temperature, 131: U component of wind,
            132: V componentof wind, 133: Specific humidity
            """
            return {
                "levtype": "ml",
                "levelist": "1/10/100/137",
                "param": "129/130/131/132/133",
            }
        elif levtype == "pl":
            """
            129.128: Geopotential, 130.128: Temperature, 131: U component of wind,
            132: V componentof wind, 133.128: Specific humidity
            """
            return {
                "levtype": "pl",
                "levelist": "50/100/150/200/250/300/400/500/600/700/850/925/1000",
                "param": "129.128/130.128/131/132/133.128",
            }
        elif levtype == "sfc":
            """
            151.128: Mean sea level pressure, 165.128: 10 metre U wind component,
             166.128: 10 metre V wind component, 167.128: 2 metre temperature
            """
            return {
                "levtype": "sfc",
                "param": "151.128/165.128/166.128/167.128",
            }
        else:
            raise ValueError(f"Unsupported levtype: {levtype}")

    def get_data(self, levtype: str):
        """
        Retrieve data from the ERA5 dataset based on the specified level type.

        Args:
            levtype (str): The type of level for which data is being requested.
        """
        dataset_name = "reanalysis-era5-complete"
        request = self.build_request(levtype)

        connection = cdsapi.Client(url=self.url, key=self.key, verify=True)

        if levtype == "ml":
            connection.retrieve(
                dataset_name, request, "data/ERA5/ERA5-Reanalysis-single-2010-2023.nc"
            )
        elif levtype == "pl":
            connection.retrieve(
                dataset_name, request, "data/ERA5/ERA5-Reanalysis-pressure-2010-2023.nc"
            )
        elif levtype == "sfc":
            connection.retrieve(
                dataset_name, request, "data/ERA5/ERA5-Reanalysis-surface-2010-2023.nc"
            )

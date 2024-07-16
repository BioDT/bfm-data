# src/helpers/era5_api_config.py

from pathlib import Path


class ERA5ApiConfigurator:
    """
    A class to get the credential for connecting to CDS API, to retrieve data from Copernicus.
    """

    ERA5_CONFIG_PATH = Path.home() / "bfm-data/src/config/era5_config/cds_api.txt"
    CDSAPI_CONFIG_PATH = Path.home() / ".cdsapirc"

    def __init__(self):
        """
        Initialize the ERA5ApiConfigurator.
        """
        self.url, self.key = self.get_login()

    def get_login(self):
        """
        Retrieves the login credentials for the ERA5 dataset.
        """

        if self.ERA5_CONFIG_PATH.exists():
            url, key = self.load_era5_config()
        else:
            if self.CDSAPI_CONFIG_PATH.exists():
                cds_url, cds_key = self.load_cdsapi_config()
                cds_uid, cds_api_key = cds_key.split(":")
                if self.set_config(cds_url, cds_uid, cds_api_key):
                    url, key = self.load_era5_config()
        return url, key

    def set_config(self, url: str, uid: str, key: str):
        """
        Sets the user-input configuration for the ERA5 dataset.
        This function writes the provided URL and they uid with the key to a configuration file located at
        `self.ERA5_CONFIG_PATH`.

        Parameters:
        url (str): The URL to be written to the configuration file.
        uid (str): The uid to be written to the configuration file.
        key (str): The key to be written to the configuration file.

        Returns:
        bool: True if the configuration was successfully written, False otherwise.
        """
        try:
            self.ERA5_CONFIG_PATH.parent.mkdir(exist_ok=True, parents=True)
            with open(self.ERA5_CONFIG_PATH, mode="w", encoding="utf-8") as f:
                f.write(f"url:{url}\n")
                f.write(f"uid:{uid}\n")
                f.write(f"key:{key}\n")
            return True
        except Exception:
            return False

    def load_era5_config(self):
        """
        Loads the ERA5 configuration from a specified file.
        The configuration file has the URL and the key.

        Returns:
            tuple: A tuple containing the URL and the key as strings.
        """
        try:
            with open(self.ERA5_CONFIG_PATH, encoding="utf8") as file:
                url = file.readline().strip().removeprefix("url:")
                uid = file.readline().strip().removeprefix("uid:")
                key = file.readline().strip().removeprefix("key:")
            return url, f"{uid}:{key}"
        except FileNotFoundError:
            raise Exception(f"Configuration file not found: {self.ERA5_CONFIG_PATH}")

    def load_cdsapi_config(self):
        """
        Loads the ERA5 configuration from /.cdsapirc
        The configuration file has the URL and the key.

        Returns:
            tuple: A tuple containing the URL and the key as strings.
        """
        with open(self.CDSAPI_CONFIG_PATH, encoding="utf-8") as file:
            url = file.readline().replace("url:", "").strip()
            key = file.readline().replace("key:", "").strip()
        return url, key

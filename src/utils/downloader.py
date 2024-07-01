import os

import requests


class Downloader:
    """
    A class to handle downloading files from the internet and saving them to a specified directory.
    """

    def __init__(self, data_dir):
        """
        Initialize the Downloader with a directory for saving downloaded files.

        Args:
            data_dir (str): Directory path for storing downloaded files.
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

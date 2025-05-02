"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import logging
import os
import time


def get_folder_size(folder_path: str):
    """
    Function to get folder size.

    Args:
        folder_path (str): The path to the folder.
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def monitor_folder_size(folder_path: str, interval: int = 30):
    """
    Function to print the size of the folder every 30 seconds.

    Args:
        folder_path (str):
    """
    while True:
        try:
            size = get_folder_size(folder_path)
            print(f"Current folder size: {size / (1024 * 1024):.2f} MB")
            time.sleep(interval)
        except Exception as e:
            logging.error(f"Error while monitoring folder size: {e}")

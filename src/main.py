# src/main.py

import asyncio
import concurrent.futures
import gc
import threading

from src.config import settings
from src.data_ingestion.api_clients.bold import BOLDDownloader
from src.data_ingestion.api_clients.era5 import ERA5Downloader
from src.data_ingestion.api_clients.inaturalist import iNaturalistDownloader
from src.data_ingestion.api_clients.mapoflife import MOL
from src.data_ingestion.api_clients.xenocanto import XenoCantoDownloader
from src.monitoring_logging.monitoring.folder_size import (
    get_folder_size,
    monitor_folder_size,
)


def era5():
    """
    Date should be in this form '2010-01-01/to/2023-12-31'
    Time should be in this form '00/to/23/by/6', means 00:00, 06:00, 12:00 ...
    Area sould be max-min latitude and longitude, '80/-50/-25/0' means global.
    Options :
        'ml': Single Level
        'pl': Pressure Level
        'sfc': Land/Surface Level
    """
    date, time, area = "2010-01-01/to/2023-12-31", "00/to/23/by/6", "80/-50/-25/0"
    era5_downloader = ERA5Downloader(settings.DATA_DIR, date, time, area)
    era5_downloader.get_data("ml")


def xeno_canto():
    xeno_canto_downloader = XenoCantoDownloader(settings.DATA_DIR)
    xeno_canto_downloader.download()


async def iNaturalist():
    iNaturalist = iNaturalistDownloader(settings.DATA_DIR)
    await iNaturalist.run()


def BOLD():
    bold_downloader = BOLDDownloader(settings.DATA_DIR)
    bold_downloader.download()


def mapoflife():
    mol_downloader = MOL(settings.DATA_DIR)
    mol_downloader.get_save_data("Malurus cyaneus", "MOL.csv")


def run_async_in_thread(async_func):
    """
    Helper to run an async function in a synchronous context.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_func)
    loop.close()


def main():

    print(
        f"Initial folder size: {get_folder_size(settings.LIFE_DIR) / (1024 * 1024):.2f} MB"
    )

    monitoring_thread = threading.Thread(
        target=monitor_folder_size, args=(settings.LIFE_DIR,)
    )
    monitoring_thread.daemon = True
    monitoring_thread.start()

    functions = [lambda: run_async_in_thread(iNaturalist()), xeno_canto, BOLD]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fn) for fn in functions]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Function raised an exception: {e}")
            finally:
                gc.collect()


if __name__ == "__main__":
    main()

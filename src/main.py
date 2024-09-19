# src/main.py

import asyncio
import concurrent.futures
import gc
import threading

from src.config.paths import DATA_DIR, ERA5_DIR, LIFE_DIR, PROCESSED_DATA_DIR
from src.data_ingestion.api_clients.bold import BOLDDownloader
from src.data_ingestion.api_clients.era5 import ERA5Downloader
from src.data_ingestion.api_clients.inaturalist import iNaturalistDownloader
from src.data_ingestion.api_clients.mapoflife import MOL
from src.data_ingestion.api_clients.xenocanto import XenoCantoDownloader
from src.dataset_creation.create_dataset import create_dataset, create_species_dataset
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
        'single': Single Level
        'pressure': Pressure Level
        'surface': Land/Surface Level
    """
    start_date = "2001-01-01"
    end_date = "2001-02-31"
    date, time = f"{start_date}/{end_date}", "00/to/23/by/6"
    era5_downloader = ERA5Downloader(DATA_DIR, date, time)
    levels = ["pressure"]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(era5_downloader.get_data, level, start_date, end_date)
            for level in levels
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"Request completed with result: {result}")
            except Exception as e:
                print(f"Request generated an exception: {e}")


def xeno_canto():
    xeno_canto_downloader = XenoCantoDownloader(DATA_DIR)
    xeno_canto_downloader.download()


async def iNaturalist():
    iNaturalist = iNaturalistDownloader(DATA_DIR)
    await iNaturalist.run()


def BOLD():
    bold_downloader = BOLDDownloader(DATA_DIR)
    bold_downloader.run()


def mapoflife():
    mol_downloader = MOL(DATA_DIR)
    mol_downloader.run()


def run_async_in_thread(async_func):
    """
    Helper to run an async function in a synchronous context.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_func)
    loop.close()


from src.dataset_creation.load_data import load_species_data


def main():

    # print(f"Initial folder size: {get_folder_size(LIFE_DIR) / (1024 * 1024):.2f} MB")

    # monitoring_thread = threading.Thread(target=monitor_folder_size, args=(LIFE_DIR,))
    # monitoring_thread.daemon = True
    # monitoring_thread.start()

    # functions = [lambda: run_async_in_thread(iNaturalist()), xeno_canto, BOLD]

    # with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    #     futures = [executor.submit(fn) for fn in functions]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"Function raised an exception: {e}")
    #         finally:
    #             gc.collect()
    # create_multimodal_dataset(LIFE_DIR, f"{DATA_DIR}multimodal_dataset_final.parquet")
    # dataset.to_csv("dataset.csv", encoding='utf-8')
    # era5()

    species_file = f"{PROCESSED_DATA_DIR}/species_dataset.parquet"

    # dataset = load_species_data(species_file, 10)
    # print(dataset)

    # create_species_dataset(LIFE_DIR, species_file)

    # Try reading the file

    # batch_size = 10

    # # Open the parquet file as a reader
    # parquet_reader = pq.ParquetFile(species_file)

    # counter = 0
    # for batch in parquet_reader.iter_batches(batch_size=batch_size):
    #     df = batch.to_pandas()
    #     print(df.head())
    #     print(counter)
    #     counter += 1

    # parquet_reader.close()

    surface_file = f"{ERA5_DIR}/ERA5-Reanalysis-surface-2001-01-01-2001-12-31.nc"
    single_file = f"{ERA5_DIR}/ERA5-Reanalysis-single-2001-01-01-2001-12-31.nc"
    pressure_file = f"{ERA5_DIR}/ERA5-Reanalysis-pressure-2001-01-01-2001-02-31.nc"

    dataset_file = f"{PROCESSED_DATA_DIR}/final_dataset.parquet"
    batches = create_dataset(
        species_file, surface_file, single_file, pressure_file, dataset_file
    )

    for batch in batches:
        print(batch)

    # mapoflife()


if __name__ == "__main__":
    main()

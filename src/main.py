# src/main.py

from concurrent.futures import ProcessPoolExecutor

from src.config import settings
from src.utils.bold import BOLDDownloader
from src.utils.era5 import ERA5Downloader
from src.utils.inaturalist import iNaturalistDownloader
from src.utils.mapoflife import MOL
from src.utils.xenocanto import XenoCantoDownloader


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


def iNaturalist():
    iNaturalist = iNaturalistDownloader(settings.DATA_DIR)
    iNaturalist.download_save_observations()


def BOLD():
    bold_downloader = BOLDDownloader(settings.DATA_DIR)
    bold_downloader.get_and_save_data()


def mapoflife():
    mol_downloader = MOL(settings.DATA_DIR)
    mol_downloader.get_save_data("Malurus cyaneus", "MOL.csv")


def main():
    pass


if __name__ == "__main__":
    main()

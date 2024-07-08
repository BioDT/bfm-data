# src/main.py

from src.config import settings
from src.utils.inaturalist import iNaturalistDownloader
from src.utils.xenocanto import XenoCantoDownloader


def main():
    xeno_canto_downloader = XenoCantoDownloader(settings.DATA_DIR)
    iNaturalist = iNaturalistDownloader(settings.DATA_DIR)
    xeno_canto_downloader.download()
    observations = iNaturalist.get_observations()
    iNaturalist.download_save_observations(observations, "data/iNaturalist.csv")


if __name__ == "__main__":
    main()

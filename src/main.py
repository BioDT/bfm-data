# src/main.py

from src.config import settings
from src.utils.xeno_canto import XenoCantoDownloader


def main():
    XenoCantoDownloader.download_all()


if __name__ == "__main__":
    main()

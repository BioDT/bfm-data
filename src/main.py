# src/main.py

from src.config import settings
from src.utils.xenocanto import XenoCantoDownloader


def main():
    XenoCantoDownloader.download_all()


if __name__ == "__main__":
    main()

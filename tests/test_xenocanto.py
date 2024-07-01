# tests/test_xenocanto.py

import os
import unittest

from src.config import settings
from src.utils.xenocanto import XenoCantoDownloader


class TestXenoCantoDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = XenoCantoDownloader(settings.COUNTRY, "test_data/")

    def tearDown(self):
        if os.path.exists("test_data/"):
            for root, dirs, files in os.walk("test_data/", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir("test_data/")

    def test_download_bird_songs(self):
        self.downloader.download_bird_songs()
        self.assertTrue(os.path.exists("test_data/Xeno-Canto/" + settings.COUNTRY))


if __name__ == "__main__":
    unittest.main()

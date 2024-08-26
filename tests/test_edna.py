# tests/test_image.py

import os
import unittest
from urllib import request

from src.config.settings import TEST_DATA_DIR
from src.data_ingestion.api_clients.bold import BOLDDownloader


class TestEDNA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = BOLDDownloader(TEST_DATA_DIR)

    def test_connection(self):
        url = "https://www.boldsystems.org/index.php/API_Public/combined?geo=Aruba&format=json&offset=0&limit=100"
        status = request.urlopen(url).getcode()
        self.assertEqual(status, 200)

    def test_download(self):
        self.downloader.download("Coracias caudatus")

        self.assertTrue(
            os.path.exists(
                f"{TEST_DATA_DIR}/Life/Coracias caudatus/Coracias caudatus_edna.csv"
            )
        )


if __name__ == "__main__":
    unittest.main()

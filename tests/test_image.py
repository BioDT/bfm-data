# tests/test_image.py

import json
import os
import unittest
from urllib import request

from src.config.paths import TEST_DATA_DIR
from src.data_ingestion.api_clients.inaturalist import iNaturalistDownloader

# from src.utils.preprocessing.image import process_image


class TestAudio(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = iNaturalistDownloader(TEST_DATA_DIR)

    def test_connection(self):
        url = "https://api.inaturalist.org/v1/observations?per_page=200&has%5B%5D=photos&has%5B%5D=geo&quality_grade=research&page=1"
        status = request.urlopen(url).getcode()
        self.assertEqual(status, 200)

    async def test_download(self):
        with open(f"{TEST_DATA_DIR}/observations.txt", "r") as file:
            observations_str = file.read()

        observations = json.loads(observations_str)
        await self.downloader.process_and_download_observations(observations)
        self.assertTrue(
            os.path.exists(
                f"{TEST_DATA_DIR}/Life/Coracias caudatus caudatus/Lilac-breasted Roller_237094807_1.jpg"
            )
        )
        self.assertTrue(
            os.path.exists(
                f"{TEST_DATA_DIR}/Life/Lysimachia vulgaris/Yellow Loosestrife_237094847_1.jpg"
            )
        )
        self.assertTrue(
            os.path.exists(
                f"{TEST_DATA_DIR}/Life/Lysimachia vulgaris/Yellow Loosestrife_237094847_4.jpg"
            )
        )


if __name__ == "__main__":
    unittest.main()

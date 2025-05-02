"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import json
import os
import unittest
from urllib import request

from src.config.paths import TEST_DATA_DIR
from src.data_ingestion.api_clients.inaturalist import iNaturalistDownloader
from src.data_preprocessing.preprocessing import preprocess_image


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

    def test_preprocessing(self):

        input_file_path = f"{TEST_DATA_DIR}/Life/Coracias caudatus caudatus/Lilac-breasted Roller_237094807_1.jpg"
        output_file_path = f"{TEST_DATA_DIR}/Life/Coracias caudatus caudatus/Lilac-breasted Roller_237094807_1_test.jpg"

        input_file_path_2 = f"{TEST_DATA_DIR}/Life/Lysimachia vulgaris/Yellow Loosestrife_237094847_1.jpg"
        output_file_path_2 = f"{TEST_DATA_DIR}/Life/Lysimachia vulgaris/Yellow Loosestrife_237094847_1_test.jpg"

        self.assertTrue(
            os.path.exists(input_file_path),
            f"Input file does not exist: {input_file_path}",
        )

        self.assertTrue(
            os.path.exists(input_file_path_2),
            f"Input file does not exist: {input_file_path_2}",
        )

        preprocess_image(input_file_path, output_file_path)
        preprocess_image(input_file_path_2, output_file_path_2)

        self.assertTrue(
            os.path.exists(output_file_path),
            f"Output file was not created: {output_file_path}",
        )


if __name__ == "__main__":
    unittest.main()

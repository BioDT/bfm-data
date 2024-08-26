# tests/test_image.py

import json
import os
import unittest
from urllib import request

from src.config.settings import TEST_DATA_DIR
from src.data_ingestion.api_clients.inaturalist import iNaturalistDownloader

# from src.utils.preprocessing.image import process_image


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = iNaturalistDownloader(TEST_DATA_DIR)

    def test_connection(self):
        url = "https://api.inaturalist.org/v1/observations?per_page=200&has%5B%5D=photos&has%5B%5D=geo&quality_grade=research&page=1"
        status = request.urlopen(url).getcode()
        self.assertEqual(status, 200)

    def test_download(self):
        with open(f"{TEST_DATA_DIR}/observations.txt", "r") as file:
            observations_str = file.read()

        observations = json.loads(observations_str)
        self.downloader.process_and_download_observations(observations)
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

    # def test_preprocessing(self):

    #     input_file_path = "test_data/Life/Hungary/Thyatira batis/Peach Blossom Moth_Thyatira batis_233281271_1.jpg"
    #     output_file_path = "test_data/Life/Hungary/Thyatira batis/Peach Blossom Moth_Thyatira batis_233281271_1_test.jpg"

    #     self.assertTrue(
    #         os.path.exists(input_file_path),
    #         f"Input file does not exist: {input_file_path}",
    #     )

    #     process_image(input_file_path, output_file_path)

    #     self.assertTrue(
    #         os.path.exists(output_file_path),
    #         f"Output file was not created: {output_file_path}",
    #     )


if __name__ == "__main__":
    unittest.main()

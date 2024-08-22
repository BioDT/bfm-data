# tests/test_image.py

import json
import os
import unittest
from urllib import request

from src.data_ingestion.api_clients.inaturalist import iNaturalistDownloader
from src.data_preprocessing.process import process_image


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = iNaturalistDownloader("test_data")

    def test_connection(self):
        url = "https://api.inaturalist.org/v1/observations?per_page=200&has%5B%5D=photos&has%5B%5D=geo&quality_grade=research&page=1"
        status = request.urlopen(url).getcode()
        self.assertEqual(status, 200)

    def test_download(self):
        with open("test_data/image/observations.txt", "r") as file:
            observations_str = file.read()

        observations = json.loads(observations_str)
        self.downloader.process_and_download_observations(observations)
        self.assertTrue(
            os.path.exists(
                "test_data/Life/South Africa/Coracias caudatus caudatus/Lilac-breasted Roller_Coracias caudatus caudatus_237094807_1.jpg"
            )
        )
        self.assertTrue(
            os.path.exists(
                "test_data/Life/United Kingdom/Lysimachia vulgaris/Yellow Loosestrife_Lysimachia vulgaris_237094847_1.jpg"
            )
        )
        self.assertTrue(
            os.path.exists(
                "test_data/Life/United Kingdom/Lysimachia vulgaris/Yellow Loosestrife_Lysimachia vulgaris_237094847_4.jpg"
            )
        )

    def test_preprocessing(self):

        input_file_path = "test_data/Life/South Africa/Coracias caudatus caudatus/Lilac-breasted Roller_Coracias caudatus caudatus_237094807_1.jpg"
        output_file_path = "test_data/Life/South Africa/Coracias caudatus caudatus/Lilac-breasted Roller_Coracias caudatus caudatus_237094807_1_test.jpg"

        input_file_path_2 = "test_data/Life/United Kingdom/Lysimachia vulgaris/Yellow Loosestrife_Lysimachia vulgaris_237094847_1.jpg"
        output_file_path_2 = "test_data/Life/United Kingdom/Lysimachia vulgaris/Yellow Loosestrife_Lysimachia vulgaris_237094847_1_test.jpg"

        self.assertTrue(
            os.path.exists(input_file_path),
            f"Input file does not exist: {input_file_path}",
        )

        self.assertTrue(
            os.path.exists(input_file_path_2),
            f"Input file does not exist: {input_file_path_2}",
        )

        process_image(input_file_path, output_file_path)
        process_image(input_file_path_2, output_file_path_2)

        self.assertTrue(
            os.path.exists(output_file_path),
            f"Output file was not created: {output_file_path}",
        )

        self.assertTrue(
            os.path.exists(output_file_path_2),
            f"Output file was not created: {output_file_path_2}",
        )


if __name__ == "__main__":
    unittest.main()

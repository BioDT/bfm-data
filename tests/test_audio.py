# tests/test_audio.py

import os
import unittest
from urllib import request

import torch
import torchaudio

from src.config.paths import TEST_DATA_DIR
from src.data_ingestion.api_clients.xenocanto import XenoCantoDownloader
from src.data_preprocessing.feature_extraction.audio import extract_mfcc
from src.data_preprocessing.process import process_audio


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = XenoCantoDownloader(TEST_DATA_DIR)

    def test_connection(self):
        url = "https://xeno-canto.org/api/2/recordings?query=Coracias+caudatus+caudatus+q_gt:C&page=1"
        status = request.urlopen(url).getcode()
        self.assertEqual(status, 200)

    def test_download(self):
        self.downloader.download("Coracias caudatus caudatus")
        self.assertTrue(
            os.path.exists(
                f"{TEST_DATA_DIR}/Life/Coracias caudatus caudatus/517LBROL0911210809NAMA2.wav"
            )
        )

        self.assertTrue(
            os.path.exists(
                f"{TEST_DATA_DIR}/Life/Coracias caudatus caudatus/XC365265-Coracias_caudatus_nom-FL calls Liwonde NP south 6Dec15 9.15am LS118988b.wav"
            )
        )

        self.assertTrue(
            os.path.exists(
                f"{TEST_DATA_DIR}/Life/Coracias caudatus caudatus/XC429948-Coracius_caudatus_nom-FL calls NdutuLodge 3Nov17 7.38am LS113901a_audio.csv"
            )
        )

    def test_preprocessing(self):

        input_file_path = f"{TEST_DATA_DIR}/Life/Coracias caudatus caudatus/XC429640-Coracias_caudatus_nom-FL mobbing kite nr SimbaLodge TarangireNP 29Oct17 10.10am LS113587a.wav"
        output_file_path = f"{TEST_DATA_DIR}/Life/Coracias caudatus caudatus/XC429640-Coracias_caudatus_nom-FL mobbing kite nr SimbaLodge TarangireNP 29Oct17 10.10am LS113587a_preprocessed.wav"

        self.assertTrue(
            os.path.exists(input_file_path),
            f"Input file does not exist: {input_file_path}",
        )

        process_audio(input_file_path, output_file_path)

        self.assertTrue(
            os.path.exists(output_file_path),
            f"Output file was not created: {output_file_path}",
        )


if __name__ == "__main__":
    unittest.main()

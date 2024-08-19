# tests/test_audio.py

import os
import unittest
from urllib import request

from src.data_ingestion.api_clients.xenocanto import XenoCantoDownloader
from src.data_preprocessing.process import process_audio


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = XenoCantoDownloader("test_data")

    # @classmethod
    # def tearDownClass(cls):
    #     try:
    #         shutil.rmtree("test_data/")
    #     except OSError:
    #         pass

    def test_connection(self):
        url = "https://xeno-canto.org/api/2/recordings?query=cnt:Aruba+q_gt:AC+type:song&page=1"
        status = request.urlopen(url).getcode()
        self.assertEqual(status, 200)

    def test_download(self):
        self.downloader.download_sounds_per_country(
            "Afghanistan", "test_data/audio/Xeno-Canto/Afghanistan"
        )
        self.assertTrue(
            os.path.exists(
                "test_data/audio/Xeno-Canto/Afghanistan/Phylloscopus griseolus/XC181207-B09h01m16s10jun2008.wav"
            )
        )

    def test_preprocessing(self):

        input_file_path = "test_data/audio/Xeno-Canto/Afghanistan/Phylloscopus griseolus/XC181207-B09h01m16s10jun2008.wav"
        output_file_path = (
            "test_data/audio/Xeno-Canto/Afghanistan/Phylloscopus griseolus/test.wav"
        )

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

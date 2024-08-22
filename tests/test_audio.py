# tests/test_audio.py

import os
import unittest
from urllib import request

import torch
import torchaudio

from src.data_ingestion.api_clients.xenocanto import XenoCantoDownloader
from src.data_preprocessing.feature_extraction.audio import extract_mfcc
from src.data_preprocessing.process import process_audio


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.downloader = XenoCantoDownloader("test_data")

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

    def test_feature_extraction(self):
        input_file_path = "test_data/audio/Xeno-Canto/Afghanistan/Phylloscopus griseolus/XC181207-B09h01m16s10jun2008.wav"

        audio, sample_rate = torchaudio.load(input_file_path)

        mfcc_features = extract_mfcc(audio, sample_rate)

        self.assertIsInstance(
            mfcc_features, torch.Tensor, "MFCC features should be a torch.Tensor"
        )

        self.assertTrue(
            mfcc_features.shape[0] > 0,
            "MFCC features should have more than 0 coefficients",
        )


if __name__ == "__main__":
    unittest.main()

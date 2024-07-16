# tests/test_xenocanto.py

import asyncio
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
from urllib import request

import torch

from src.utils.downloader import Downloader
from src.utils.xenocanto import XenoCantoDownloader


class TestXenoCantoDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = XenoCantoDownloader("test_data")

    def tearDown(self):
        try:
            shutil.rmtree("test_data/")
        except OSError:
            pass

    def test_connection(self):
        url = "https://xeno-canto.org/api/2/recordings?query=cnt:Aruba+q_gt:AC+type:song&page=1"
        status = request.urlopen(url).getcode()
        self.assertEqual(status, 200)

    def test_download(self):
        self.downloader.download_sounds_per_country(
            "Afghanistan", "test_data/Xeno-Canto/Afghanistan"
        )
        self.assertTrue(os.path.exists("test_data/xeno_canto.csv"))
        self.assertTrue(
            os.path.exists(
                "test_data/Xeno-Canto/Afghanistan/Phylloscopus griseolus/XC181207-B09h01m16s10jun2008.wav"
            )
        )


if __name__ == "__main__":
    unittest.main()

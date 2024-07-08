# # tests/test_downloader.py

import csv
import os
import shutil
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

from src.utils.downloader import Downloader


class TestDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = Downloader(
            "test_data", "TestSource", "http://example.com/api"
        )

    @patch("os.makedirs")
    def test_init(self, mock_makedirs):
        # Remove the directory if it exists to ensure os.makedirs will be called
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")

        test_downloader = Downloader("test_data", "TestSource")
        mock_makedirs.assert_called_once_with("test_data")
        self.assertEqual(test_downloader.data_dir, "test_data")
        self.assertEqual(test_downloader.source, "TestSource")
        self.assertEqual(
            test_downloader.base_path, os.path.join("test_data", "TestSource")
        )

    @patch("requests.Session.send")
    def test_get_base_url_page(self, mock_send):
        mock_response = Mock()
        expected_response = {"key": "value"}
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_send.return_value = mock_response

        params = {"param1": "value1"}
        response = self.downloader.get_base_url_page(params)
        self.assertEqual(response, expected_response)

        mock_send.assert_called_once()
        request_sent = mock_send.call_args[0][0]

        parsed_url = urlparse(request_sent.url)
        self.assertEqual(
            f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}",
            "http://example.com/api",
        )
        self.assertEqual(parse_qs(parsed_url.query), {"param1": ["value1"]})

    def test_save_to_csv(self):
        data = [
            {"key1": "value1", "key2": "value2"},
            {"key1": "value3", "key2": "value4"},
        ]
        filename = "test_data/test.csv"
        self.downloader.save_to_csv(data, filename)

        with open(filename, mode="r") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["key1"], "value1")
            self.assertEqual(rows[1]["key1"], "value3")

        os.remove(filename)

    @patch("requests.get")
    def test_download_file(self, mock_get):
        url = "http://example.com/file"
        filename = "test_data/test_file.jpg"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raw = Mock()

        mock_get.return_value = mock_response

        with patch("shutil.copyfileobj") as mock_copyfileobj:
            self.downloader.download_file(url, filename)
            mock_copyfileobj.assert_called_once_with(
                mock_response.raw, unittest.mock.ANY
            )

        mock_get.assert_called_once_with(url, stream=True)


if __name__ == "__main__":
    unittest.main()

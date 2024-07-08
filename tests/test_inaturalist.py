# tests/test_inaturalist.py

import unittest
from unittest.mock import Mock, patch

from src.utils.inaturalist import iNaturalistDownloader


class TestiNaturalistDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = iNaturalistDownloader("test_data")

    @patch("src.utils.inaturalist.iNaturalistDownloader.get_base_url_page")
    def test_get_observations(self, mock_get_base_url_page):
        mock_response = {
            "total_results": 40,
            "per_page": 20,
            "results": [
                {
                    "id": 1,
                    "taxon": {
                        "iconic_taxon_name": "Bird",
                        "preferred_common_name": "Sparrow",
                        "name": "Passeridae",
                        "min_species_taxon_id": 12345,
                    },
                    "photos": [
                        {"url": "https://example.com/photo1/square.jpg"},
                        {"url": "https://example.com/photo2/square.jpg"},
                    ],
                    "geojson": {"coordinates": [-123.2620, 44.5646]},
                    "place_guess": "Some Place",
                }
            ],
        }
        mock_get_base_url_page.return_value = mock_response

        observations = self.downloader.get_observations()
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["id"], 1)
        self.assertEqual(observations[0]["taxon"]["iconic_taxon_name"], "Bird")

    @patch("src.utils.inaturalist.iNaturalistDownloader.get_country_from_coordinates")
    @patch("src.utils.inaturalist.iNaturalistDownloader.download_file")
    def test_download_save_observations(
        self, mock_download_file, mock_get_country_from_coordinates
    ):
        mock_get_country_from_coordinates.return_value = "United States"
        observations = [
            {
                "id": 1,
                "taxon": {
                    "iconic_taxon_name": "Bird",
                    "preferred_common_name": "Sparrow",
                    "name": "Passeridae",
                    "min_species_taxon_id": 12345,
                },
                "photos": [{"url": "https://example.com/photo1/square.jpg"}],
                "geojson": {"coordinates": [-123.2620, 44.5646]},
                "place_guess": "Some Place",
            }
        ]

        with patch(
            "src.utils.inaturalist.iNaturalistDownloader.save_to_csv"
        ) as mock_save_to_csv:
            self.downloader.download_save_observations(observations, "test.csv")
            mock_save_to_csv.assert_called_once()
            self.assertTrue(mock_download_file.called)

    @patch("geopy.geocoders.Nominatim.reverse")
    def test_get_country_from_coordinates(self, mock_reverse):
        mock_reverse.return_value = Mock()
        mock_reverse.return_value.raw = {"address": {"country": "United States"}}
        country = self.downloader.get_country_from_coordinates([-123.2620, 44.5646])
        self.assertEqual(country, "United States")


if __name__ == "__main__":
    unittest.main()

# src/utils/downloading/xenocanto.py

import os
import tempfile
from pathlib import Path

import pycountry
import requests
import torchaudio
import torchaudio.transforms as T

from src.utils.downloading.downloader import Downloader


class XenoCantoDownloader(Downloader):
    """
    Class for downloading and processing bird song recordings from Xeno-Canto.
    """

    AUDIO_SAMPLE_RATE = 16000

    def __init__(self, data_dir: str):
        """
        Initialize the XenoCantoDownloader.

        Args:
            country (str): Country name for querying bird songs.
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "Life")
        self.base_url = "https://xeno-canto.org/api/2/recordings"

    def get_xeno_canto_recordings(self, country: str, page: int = 1):
        """
        Fetch bird song information from xeno-canto API.

        Args:
            page (int): Page number for paginated API results. Default is 1.

        Returns:
            list: A list of acceptable bird recordings depends on our parameters.
        """

        # cnt (str): The country where the recording was made
        # q_ct (str): The quality rating for the recording sould be higher than. options: 'A', 'B', 'C', 'D', 'E'
        # type (str) = The sound type of the recording. options: 'call', 'song', and etc.

        query = f"cnt:{country} q_gt:C type:song"
        params = {"query": query, "page": page}
        json_response = self.get_base_url_page(params)
        num_pages = json_response.get("numPages", 1)

        def is_acceptable(recording):
            """
            Determine if a recording is acceptable based on the number of additional species.
            """
            return len(recording["also"]) <= 1 and (
                not recording["also"] or recording["also"][0] == ""
            )

        recordings = [
            recording
            for recording in json_response["recordings"]
            if is_acceptable(recording)
        ]

        if page < num_pages:
            recordings.extend(self.get_xeno_canto_recordings(country, page + 1))

        return recordings

    def download(self):
        """
        Download bird songs for all the countries. Create also the country's directory.
        """
        countries = [country.name for country in pycountry.countries]
        for country in countries:
            if country:
                country_path = os.path.join(self.base_path, country)
            else:
                country_path = self.base_path
            Path(country_path).mkdir(parents=True, exist_ok=True)
            self.download_sounds_per_country(country, country_path)

    def download_sounds_per_country(self, country: str, country_path: str):
        """
        Download bird songs for the specified country.

        Args:
            country (str): The name of the specified country.
            country_path (str): Country's directory path for saving the files.
        """
        recordings = self.get_xeno_canto_recordings(country)
        if not recordings:
            return

        for recording in recordings:
            self.download_process_recording((recording, country_path))

    def download_process_recording(self, args: dict):
        """
        Download and process a single bird song recording.

        Args:
            recording (dict): A recording entry from the API response.
        """
        recording, country_path = args

        file_url = recording.get("file", "Unknown")
        file_name = recording.get("file-name", "Unknown")
        scientific_name = (
            f"{recording.get('gen', 'Unknown')} {recording.get('sp', 'Unknown')}"
        )
        scientific_name_path = os.path.join(country_path, scientific_name)
        base_name = os.path.splitext(file_name)[0]
        if not os.path.exists(scientific_name_path):
            Path(scientific_name_path).mkdir(parents=True, exist_ok=True)

        existing_files = [
            file
            for file in os.listdir(scientific_name_path)
            if file.startswith(base_name)
        ]
        if existing_files:
            return
        if not file_url.startswith("https://"):
            return

        try:
            request_result = requests.get(file_url, allow_redirects=True)
            request_result.raise_for_status()
        except requests.exceptions.RequestException:
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(request_result.content)
            temp_file_path = temp_file.name

        waveform, sample_rate = torchaudio.load(temp_file_path)
        os.remove(temp_file_path)

        if sample_rate != self.AUDIO_SAMPLE_RATE:
            resampler = T.Resample(
                orig_freq=sample_rate, new_freq=self.AUDIO_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        self.save_audio(waveform, base_name, scientific_name_path)
        data = []
        id = recording.get("id", "Uknown")
        group = recording.get("group", "Uknown")
        country = recording.get("cnt", "Uknown")
        lat = recording.get("lat", "0.0")
        lon = recording.get("lng", "0.0")
        coordinates = f"[{lon}, {lat}]"
        preferred_common_name = recording.get("en", "Uknown")
        data.append(
            {
                "id": id,
                "Group": group,
                "Scientific_name": scientific_name,
                "Common_name": preferred_common_name,
                "Country": country,
                "Coordinates": coordinates,
                "audio": file_url,
            }
        )
        self.save_to_csv(data, os.path.join(scientific_name_path, f"{base_name}.csv"))

    def save_audio(self, waveform: T, base_name: str, scientific_name_path: str):
        """
        Save the waveform as .wav files.

        Args:
            waveform (Tensor): The waveform data.
            base_name (str): The base name for the output files.
            species_path (str): Species's directory path for saving the files.
        """
        torchaudio.save(
            os.path.join(scientific_name_path, f"{base_name}.wav"),
            waveform,
            self.AUDIO_SAMPLE_RATE,
        )

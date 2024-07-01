# src/utils/xenocanto.py

import io
import json
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import pycountry
import requests
import torchaudio
import torchaudio.transforms as T

from src.config import settings
from src.utils.downloader import Downloader


class XenoCantoDownloader:
    """
    Class for downloading and processing bird song recordings from Xeno-Canto.
    """

    AUDIO_MAX_MS = 8000  # Maximum audio clip duration in milliseconds
    AUDIO_MIN_MS = 4000  # Minimum audio clip duration in milliseconds
    AUDIO_SAMPLE_RATE = 16000  # Sample rate for audio processing

    URL = "https://xeno-canto.org/api/2/recordings"

    def __init__(self, country, data_dir):
        """
        Initialize the XenoCantoDownloader with country and data directory.

        Args:
            country (str): Country name for querying bird songs.
            data_dir (str): Directory path for storing downloaded data.
        """
        self.country = country
        self.downloader = Downloader(data_dir)
        self.country_path = os.path.join(data_dir, "Xeno-Canto", country)
        self.create_directories()

    def create_directories(self):
        """
        Create necessary directories if they don't exist.
        """
        Path(self.country_path).mkdir(parents=True, exist_ok=True)

    def get_xeno_canto_page(query, page: int = 1):
        """
        Fetch a specific page of bird song recordings from the xeno-canto API.

        Args:
            query (str): The query string to search for bird recordings.
            page (int): Page number for paginated API results. Default is 1.

        Returns:
            dict: The JSON response from the xeno-canto API containing bird recordings.
        """
        params = {"query": query, "page": page}
        response = requests.get(url=XenoCantoDownloader.URL, params=params)
        return response.json()

    def get_bird_info(self, page: int = 1):
        """
        Fetch bird song information from xeno-canto API.

        Args:
            page (int): Page number for paginated API results. Default is 1.

        Returns:
            list: A list of acceptable bird recordings depends on our parameters.
        """

        query = f"cnt:{self.country} q_gt:C type:song"
        json_response = XenoCantoDownloader.get_xeno_canto_page(query, page)
        num_pages = json_response["numPages"]

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
            recordings.extend(self.get_bird_info(page + 1))

        return recordings

    def download_bird_songs(self):
        """
        Download bird songs for the specified country.
        """
        recordings = self.get_bird_info()
        if not recordings:
            return

        with Pool(processes=10) as pool:
            pool.map(self.download_process_recording, recordings)

    def download_process_recording(self, recording):
        """
        Download and process a single bird song recording.

        Args:
            recording (dict): A recording entry from the API response.
        """
        file_url = recording["file"]
        file_name = recording["file-name"]
        species_name = f"{recording['gen']} {recording['sp']}"
        species_path = os.path.join(self.country_path, species_name)
        base_name = os.path.splitext(file_name)[0]
        if not os.path.exists(species_path):
            Path(species_path).mkdir(parents=True, exist_ok=True)

        existing_files = [
            f for f in os.listdir(species_path) if f.startswith(base_name)
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

        self.save_waveform(waveform, base_name, species_path)

    def save_waveform(self, waveform, base_name, species_path):
        """
        Save the waveform as .wav files, splitting if necessary.

        Args:
            waveform (Tensor): The waveform data.
            base_name (str): The base name for the output files.
            species_path (str): Directory path for saving the files.
        """
        audio_length = waveform.size(1) / self.AUDIO_SAMPLE_RATE * 1000

        if audio_length <= self.AUDIO_MAX_MS:
            torchaudio.save(
                os.path.join(species_path, f"{base_name}.wav"),
                waveform,
                self.AUDIO_SAMPLE_RATE,
            )
        else:
            self.split_and_save_waveform(
                waveform, base_name, species_path, audio_length
            )

    def split_and_save_waveform(self, waveform, base_name, species_path, audio_length):
        """
        Split and save long waveforms into smaller chunks.

        Args:
            waveform (Tensor): The waveform data.
            base_name (str): The base name for the output files.
            species_path (str): Directory path for saving the files.
            audio_length (float): The total duration of the audio in milliseconds.
        """
        for pos in range(0, int(audio_length), self.AUDIO_MAX_MS):
            end_pos = min(
                waveform.size(1),
                int((pos + self.AUDIO_MAX_MS) / 1000 * self.AUDIO_SAMPLE_RATE),
            )
            section = waveform[:, int(pos / 1000 * self.AUDIO_SAMPLE_RATE) : end_pos]
            if section.size(1) < self.AUDIO_MIN_MS / 1000 * self.AUDIO_SAMPLE_RATE:
                break
            section_name = os.path.join(species_path, f"{base_name}.{pos}.wav")
            torchaudio.save(section_name, section, self.AUDIO_SAMPLE_RATE)

    def download_all():
        """
        Download bird songs fror all the countries.
        """
        countries = [country.name for country in pycountry.countries]
        for country in countries:
            xeno_canto_downloader = XenoCantoDownloader(country, settings.DATA_DIR)
            xeno_canto_downloader.download_bird_songs()

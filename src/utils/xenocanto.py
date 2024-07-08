# src/utils/xenocanto.py

import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import pycountry
import requests
import torchaudio
import torchaudio.transforms as T

from src.utils.downloader import Downloader


class XenoCantoDownloader(Downloader):
    """
    Class for downloading and processing bird song recordings from Xeno-Canto.
    """

    AUDIO_MAX_MS = 8000  # Maximum audio clip duration in milliseconds
    AUDIO_MIN_MS = 4000  # Minimum audio clip duration in milliseconds
    AUDIO_SAMPLE_RATE = 16000  # Sample rate for audio processing

    def __init__(self, data_dir, country=None):
        """
        Initialize the XenoCantoDownloader with country and data directory.

        Args:
            country (str): Country name for querying bird songs.
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "Xeno-Canto")
        self.base_url = "https://xeno-canto.org/api/2/recordings"
        self.test_country = country

    def get_xeno_canto_recordings(self, country, page: int = 1):
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

    def download(self, test=False):
        """
        Download bird songs for all the countries. Create also the country's directory.
        """
        if test:
            country = self.test_country
            country_path = os.path.join(self.base_path, country)
            Path(country_path).mkdir(parents=True, exist_ok=True)
            self.download_sounds_per_country(country, country_path)
        else:
            countries = ["Afghanistan", "Albania"]
            for country in countries:
                if country:
                    country_path = os.path.join(self.base_path, country)
                else:
                    country_path = self.base_path
                Path(country_path).mkdir(parents=True, exist_ok=True)
                self.download_sounds_per_country(country, country_path)

    def download_sounds_per_country(self, country, country_path):
        """
        Download bird songs for the specified country.

        Args:
            country (str): The name of the specified country.
            country_path (str): Country's directory path for saving the files.
        """
        recordings = self.get_xeno_canto_recordings(country)
        if not recordings:
            return

        with Pool(processes=10) as pool:
            pool.map(
                self.download_process_recording,
                [(recording, country_path) for recording in recordings],
            )

    def download_process_recording(self, args):
        """
        Download and process a single bird song recording.

        Args:
            recording (dict): A recording entry from the API response.
        """
        recording, country_path = args

        file_url = recording["file"]
        file_name = recording["file-name"]
        species_name = f"{recording['gen']} {recording['sp']}"
        species_path = os.path.join(country_path, species_name)
        base_name = os.path.splitext(file_name)[0]
        if not os.path.exists(species_path):
            Path(species_path).mkdir(parents=True, exist_ok=True)

        existing_files = [
            file for file in os.listdir(species_path) if file.startswith(base_name)
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
            species_path (str): Species's directory path for saving the files.
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
            species_path (str): Species's directory path for saving the files.
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

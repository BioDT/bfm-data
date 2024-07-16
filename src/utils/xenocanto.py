# src/utils/xenocanto.py

import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import librosa
import pycountry
import requests
import torch
import torchaudio
import torchaudio.transforms as T

from src.utils.downloader import Downloader


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
        super().__init__(data_dir, "Xeno-Canto")
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

        with Pool(processes=10) as pool:
            pool.map(
                self.download_process_recording,
                [(recording, country_path) for recording in recordings],
            )

    def download_process_recording(self, args: dict):
        """
        Download and process a single bird song recording.

        Args:
            recording (dict): A recording entry from the API response.
        """
        recording, country_path = args

        file_url = recording.get("file", "Unknown")
        file_name = recording.get("file-name", "Unknown")
        species_name = (
            f"{recording.get('gen', 'Unknown')} {recording.get('sp', 'Unknown')}"
        )
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

        self.save_audio(waveform, base_name, species_path)
        data = []
        id = recording.get("id", "Uknown")
        group = recording.get("group", "Uknown")
        country = recording.get("cnt", "Uknown")
        lat = recording.get("lat", "0.0")
        lon = recording.get("lng", "0.0")
        coordinates = f"[{lat}, {lon}]"
        preferred_common_name = recording.get("en", "Uknown")
        data.append(
            {
                "id": id,
                "Group": group,
                "Species": species_name,
                "Preferred_common_name": preferred_common_name,
                "Country": country,
                "Coordinates": coordinates,
            }
        )
        self.save_to_csv(data, os.path.join(self.data_dir, "xeno_canto.csv"))

    def save_audio(self, waveform: T, base_name: str, species_path: str):
        """
        Save the waveform as .wav files.

        Args:
            waveform (Tensor): The waveform data.
            base_name (str): The base name for the output files.
            species_path (str): Species's directory path for saving the files.
        """
        torchaudio.save(
            os.path.join(species_path, f"{base_name}.wav"),
            waveform,
            self.AUDIO_SAMPLE_RATE,
        )

    def reduce_noise(self, waveform: T):
        """
        Reduce noise in the given waveform using pre-emphasis filtering.

        Args:
            waveform (Tensor): The waveform data.

        Returns:
            Tensor: The noise-reduced waveform tensor.
        """
        y = waveform.numpy()[0]
        y_denoised = librosa.effects.preemphasis(y)
        return torch.tensor(y_denoised).unsqueeze(0)

    def normalize_audio(self, waveform: T):
        """
        Normalize the given waveform to have zero mean and unit variance.

        Args:
            waveform (Tensor): The waveform data.

        Returns:
            Tensor: The normalized waveform tensor.
        """
        y = waveform.numpy()[0]
        y_normalized = librosa.util.normalize(y)
        return torch.tensor(y_normalized).unsqueeze(0)

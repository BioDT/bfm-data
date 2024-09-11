# src/data_ingestion/api_clients/xenocanto.py

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torchaudio
import torchaudio.transforms as T

from src.data_ingestion.api_clients.downloader import Downloader
from src.helpers.handle_values import parse_date_time


class XenoCantoDownloader(Downloader):
    """
    Class for downloading and processing bird song recordings from Xeno-Canto.
    """

    def __init__(self, data_dir: str, audio_sample_rate: int = 16000):
        """
        Initialize the XenoCantoDownloader.

        Args:
            country (str): Country name for querying bird songs.
            data_dir (str): Directory path for storing downloaded data.
        """
        super().__init__(data_dir, "Life")
        self.base_url = "https://xeno-canto.org/api/2/recordings"
        self.AUDIO_SAMPLE_RATE = audio_sample_rate

    def get_xeno_canto_recordings(self, species: str = "", page: int = 1) -> dict:
        """
        Fetch bird song information from the Xeno-Canto API.

        Args:
            species (str): Scientific name of the species to filter recordings. Default is empty string for all species.
            page (int): Page number for paginated API results. Default is 1.

        Returns:
            dict: A dictionary containing recordings and pagination info.
        """

        # q_ct (str): The quality rating for the recording sould be higher than. options: 'A', 'B', 'C', 'D', 'E'
        # type (str) = The sound type of the recording. options: 'call', 'song', and etc.

        query = f"{species} q_gt:C" if species else "q_gt:C"
        params = {"query": query, "page": page}
        json_response = self.get_base_url_page(params)

        def is_acceptable(recording):
            """Determine if a recording is acceptable based on the number of additional species."""
            return len(recording["also"]) <= 1 and (
                not recording["also"] or recording["also"][0] == ""
            )

        recordings = [
            recording
            for recording in json_response.get("recordings", [])
            if is_acceptable(recording)
        ]

        return {"recordings": recordings, "numPages": json_response.get("numPages", 1)}

    def download(self, scientific_name: str = "") -> None:
        """
        Download bird songs for the specified species or all species if none specified.

        Args:
            scientific_name (str): Scientific name of the species. If empty, download all species.
        """
        if scientific_name:
            scientific_name_path = os.path.join(self.base_path, scientific_name)
            Path(scientific_name_path).mkdir(parents=True, exist_ok=True)
            self.download_sounds_for_species(scientific_name, scientific_name_path)
        else:
            self.download_sounds_for_all_species()

    def download_sounds_for_all_species(self) -> None:
        """
        Download bird songs for all available species.
        """

        page = 1
        while True:
            response = self.get_xeno_canto_recordings(page=page)
            recordings = response.get("recordings", [])
            if not recordings:
                break

            for recording in recordings:
                scientific_name = f"{recording.get('gen', 'Unknown')} {recording.get('sp', 'Unknown')}"
                scientific_name_path = os.path.join(self.base_path, scientific_name)
                Path(scientific_name_path).mkdir(parents=True, exist_ok=True)
                self.download_process_recording((recording, scientific_name_path))

            page += 1
            if page > response.get("numPages", 1):
                break

    def download_sounds_for_species(
        self, scientific_name: str, scientific_name_path: str
    ) -> None:
        """
        Download bird songs for the specified species.

        Args:
            scientific_name (str): Scientific name of the specified species.
            scientific_name_path (str): Species' directory path for saving the files.
        """
        page = 1
        while True:
            response = self.get_xeno_canto_recordings(scientific_name, page=page)
            recordings = response.get("recordings", [])
            if not recordings:
                break

            for recording in recordings:
                self.download_process_recording((recording, scientific_name_path))

            page += 1
            if page > response.get("numPages", 1):
                break

    def download_process_recording(self, args: tuple) -> None:
        """
        Download and process a single bird song recording.

        Args:
            recording (dict): A recording entry from the API response.
        """
        recording, scientific_name_path = args

        Path(scientific_name_path).mkdir(parents=True, exist_ok=True)

        file_url = recording.get("file", "Unknown")
        file_name = recording.get("file-name", "Unknown")
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
        id = recording.get("id", "Uknown")
        group = recording.get("group", "Uknown")
        country = recording.get("cnt", "Uknown")
        try:
            lat = float(recording.get("lat", "0.0"))
            lon = float(recording.get("lng", "0.0"))
        except ValueError:
            lat, lon = 0.0, 0.0

        preferred_common_name = recording.get("en", "Uknown")
        time = recording.get("time", "Uknown")
        date = recording.get("date", "Uknown")
        timestamp = parse_date_time(date, time)

        data = [
            {
                "id": id,
                "Group": group,
                "Scientific_name": f"{recording.get('gen', 'Unknown')} {recording.get('sp', 'Unknown')}",
                "Common_name": preferred_common_name,
                "Country": country,
                "Timestamp": timestamp,
                "Latitude": lat,
                "Longitude": lon,
                "audio": file_url,
            }
        ]
        self.save_to_csv(
            data,
            os.path.join(
                scientific_name_path, f"{str(base_name).replace('/', '_')}_audio.csv"
            ),
        )

    def save_audio(self, waveform: T, base_name: str, scientific_name_path: str):
        """
        Save the waveform as .wav files.

        Args:
            waveform (Tensor): The waveform data.
            base_name (str): The base name for the output files.
            scientific_name_path (str): Species's directory path for saving the files.
        """
        Path(scientific_name_path).mkdir(parents=True, exist_ok=True)

        file_path = os.path.join(
            scientific_name_path, f"{base_name.replace('/', '_')}.wav"
        )

        try:
            torchaudio.save(file_path, waveform, self.AUDIO_SAMPLE_RATE)
        except Exception as e:
            print(f"Failed to save audio file {file_path}: {str(e)}")

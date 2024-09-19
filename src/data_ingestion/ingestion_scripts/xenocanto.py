# src/data_ingestion/ingestion_scripts/xenocanto.py

import os
import shutil
import tempfile
from pathlib import Path

import requests
import torchaudio
import torchaudio.transforms as T

from src.config.paths import TEST_DATA_DIR
from src.data_ingestion.api_clients.xenocanto import XenoCantoDownloader
from src.utils.handle_values import parse_date_time


class XenoCantoDownloaderFromFile:
    """
    Class for downloading bird songs based on the occurrence dataset and saving metadata from a text file.
    """

    def __init__(self, data_dir: str, text_file: str, processed_log_file: str):
        """
        Initialize the XenoCantoDownloaderFromFile.

        Args:
            data_dir (str): Directory path for storing downloaded data.
            text_file (str): Path to the text file containing occurrence data.
            processed_log_file (str): Path to the file where processed items will be logged.
        """
        self.data_dir = data_dir
        self.text_file = text_file
        self.processed_log_file = processed_log_file
        self.processed = self.load_processed_log()

    def load_processed_log(self) -> set:
        """
        Loads the log of processed items from the log file.

        Returns:
            set: A set of catalog numbers of already processed occurrences.
        """
        if not os.path.exists(self.processed_log_file):
            return set()

        with open(self.processed_log_file, "r", encoding="utf-8") as file:
            processed = {line.strip() for line in file.readlines()}
        return processed

    def log_processed(self, catalog_number: str):
        """
        Logs the occurrence as processed by writing its catalog number to the log file.

        Args:
            catalog_number (str): The catalog number of the processed occurrence.
        """
        with open(self.processed_log_file, "a", encoding="utf-8") as file:
            file.write(f"{catalog_number}\n")

    def read_occurrences(self) -> list:
        """
        Reads occurrences from the input file and extracts relevant fields.

        Returns:
            list: A list of dictionaries containing metadata for each occurrence.
        """
        occurrences = []
        with open(self.text_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

            for line in lines[1:]:
                fields = line.strip().split("\t")

                if len(fields) < 160:
                    continue

                if len(fields) > 222:
                    iucnRedListCategory = fields[222]
                else:
                    iucnRedListCategory = ""

                occurrences.append(
                    {
                        "id": fields[0],
                        "audio": fields[7],
                        "Scientific_name": fields[201],
                        "Common_name": fields[174],
                        "Kingdom": fields[156],
                        "Phylum": fields[157],
                        "Class": fields[158],
                        "Order": fields[159],
                        "Family": fields[161],
                        "Genus": fields[165],
                        "Location": fields[88],
                        "Lat": fields[97],
                        "Lon": fields[98],
                        "eventDate": fields[62],
                        "eventTime": fields[63],
                        "iucnRedListCategory": iucnRedListCategory,
                        "catalogNumber": fields[22],
                    }
                )

        return occurrences

    def download_and_process(self):
        """
        Download sounds and save metadata to CSV.
        """
        occurrences = self.read_occurrences()

        for occurrence in occurrences:
            catalog_number = occurrence.get("catalogNumber")
            if catalog_number in self.processed:
                print(f"Skipping already processed occurrence {catalog_number}")
                continue

            scientific_name = occurrence.get("Scientific_name", "Unknown")
            scientific_name_path = os.path.join(self.data_dir, scientific_name)
            Path(scientific_name_path).mkdir(parents=True, exist_ok=True)

            audio_url = occurrence.get("audio", "")
            if audio_url:
                self.download_sound(audio_url, scientific_name_path, occurrence)
                self.log_processed(catalog_number)

    def download_sound(
        self, audio_url: str, scientific_name_path: str, occurrence: str
    ):
        """
        Download and process a single sound recording.

        Args:
            audio_url (str): URL of the sound file.
            scientific_name_path (str): Path where the audio file and metadata will be saved.
            occurrence (dict): Metadata for the occurrence.
        """
        observation_id = audio_url.split("/")[-1].replace("XC", "")
        xeno_canto_download_url = f"https://xeno-canto.org/{observation_id}/download"

        try:
            response = requests.get(xeno_canto_download_url, allow_redirects=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {xeno_canto_download_url}: {e}")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        try:
            waveform, sample_rate = torchaudio.load(temp_file_path)
            os.remove(temp_file_path)

            xeno_canto_downloader = XenoCantoDownloader(TEST_DATA_DIR)

            if sample_rate != xeno_canto_downloader.AUDIO_SAMPLE_RATE:
                resampler = T.Resample(
                    orig_freq=sample_rate,
                    new_freq=xeno_canto_downloader.AUDIO_SAMPLE_RATE,
                )
                waveform = resampler(waveform)

            print(scientific_name_path)

            xeno_canto_downloader.save_audio(
                waveform, occurrence["catalogNumber"], scientific_name_path
            )

            metadata_file = os.path.join(
                scientific_name_path, f"{occurrence['catalogNumber']}_audio.csv"
            )
            self.save_metadata(metadata_file, occurrence)

        except Exception as e:
            print(f"Failed to process sound: {e}")

    def save_metadata(self, metadata_file: str, occurrence: dict):
        """
        Save occurrence metadata to CSV.

        Args:
            metadata_file (str): Path to save metadata CSV.
            occurrence (dict): The metadata for the occurrence.
        """
        time = occurrence.get("eventTime", "Uknown")
        date = occurrence.get("eventDate", "Uknown")
        timestamp = parse_date_time(date, time)

        metadata = {
            "id": occurrence.get("id", ""),
            "Scientific_name": occurrence.get("Scientific_name", ""),
            "Common_name": occurrence.get("Common_name", ""),
            "Kingdom": occurrence.get("Kingdom", ""),
            "Phylum": occurrence.get("Phylum", ""),
            "Class": occurrence.get("Class", ""),
            "Order": occurrence.get("Order", ""),
            "Family": occurrence.get("Family", ""),
            "Genus": occurrence.get("Genus", ""),
            "Location": occurrence.get("Location", ""),
            "Timestamp": timestamp,
            "Latitude": occurrence.get("Lat", ""),
            "Longitude": occurrence.get("Lon", ""),
            "iucnRedListCategory": occurrence.get("iucnRedListCategory", ""),
            "audio": occurrence.get("audio", ""),
        }

        XenoCantoDownloader.save_to_csv(None, [metadata], metadata_file)

    def uknown_files(self):
        """
        This method identifies files that are present directly in the main data directory (self.data_dir)
        but not inside any subdirectories. It creates an 'Unknown' folder (if it does not already exist)
        and moves these files into it. This helps to organize stray or misplaced files into a dedicated
        directory.

        Returns:
            None: Moves sounds to the 'Uknown' destination directory.
        """

        unknown_folder = os.path.join(self.data_dir, "Unknown")
        os.makedirs(unknown_folder, exist_ok=True)

        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)

            if os.path.isfile(item_path):
                new_location = os.path.join(unknown_folder, item)
                shutil.move(item_path, new_location)
                print(f"Moved {item} to {new_location}")

    def move_sounds(self, source_dir: str, destination_dir: str):
        """
        Moves sounds from a source directory to a destination directory, organizing them by species.

        This function scans the subdirectories within a specified source directory. For each subdirectory,
        it assumes the folder names contain species information in a specific format (i.e., 'Genus_Species').
        It creates corresponding species-named folders in the destination directory, and moves the images into
        their respective species-specific folder.

        Args:
            source_dir (str): The path of the source directory containing the sounds.
            destination_dir (str): The base path where sounds will be moved, organized by species names.

        Returns:
            None: Moves sounds to the destination directory organized into species-specific folders.
        """

        for folder_name in os.listdir(source_dir):
            folder_path = os.path.join(source_dir, folder_name)

            if os.path.isdir(folder_path):
                species_dir = os.path.join(destination_dir, folder_name)

                if not os.path.exists(species_dir):
                    os.makedirs(species_dir)

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    shutil.move(file_path, os.path.join(species_dir, file_name))

        print("Sounds have been successfully moved.")

    def process_scientific_names_from_list(
        self, scientific_names: list, max_audios_per_species: int = 5
    ):
        """
        Process a list of scientific names, find their data in the text file, download bird songs, and save metadata.

        Args:
            scientific_names (list): A list of scientific names to process.
            max_audios_per_species (int): The maximum number of audios per species.

        """
        occurrences = self.read_occurrences()

        for scientific_name in scientific_names:
            print(f"Processing species: {scientific_name}")

            matching_occurrences = [
                occ
                for occ in occurrences
                if occ.get("Scientific_name", "").lower() == scientific_name.lower()
            ]

            if not matching_occurrences:
                print(f"No matching occurrences found for species: {scientific_name}")
                continue

            audio_count = 0

            for occurrence in matching_occurrences:
                catalog_number = occurrence.get("catalogNumber")
                if catalog_number in self.processed:
                    print(f"Skipping already processed occurrence {catalog_number}")
                    continue

                scientific_name_path = os.path.join(self.data_dir, scientific_name)
                Path(scientific_name_path).mkdir(parents=True, exist_ok=True)

                audio_url = occurrence.get("audio", "")
                if audio_url:
                    self.download_sound(audio_url, scientific_name_path, occurrence)
                    self.log_processed(catalog_number)
                    audio_count += 1
                    if audio_count >= max_audios_per_species:
                        print(
                            f"Reached max limit of {max_audios_per_species} audios for species: {scientific_name}"
                        )
                        break
                else:
                    print(f"No audio URL found for {scientific_name}")

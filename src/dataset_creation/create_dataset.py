# src/data_creation/create_dataset.py

import os

import pandas as pd
import torch
import xarray

from src.config.paths import BATCHES_DATA_DIR
from src.data_preprocessing.batch import DataBatch
from src.data_preprocessing.metadata import BatchMetadata
from src.data_preprocessing.preprocess import (
    preprocess_audio,
    preprocess_edna,
    preprocess_image,
    preprocess_text,
)
from src.dataset_creation.load_data import load_era5_datasets, load_species_data
from src.dataset_creation.preprocessing import normalize_species_dataset, pad_tensors
from src.dataset_creation.save_data import (
    save_as_parquet,
    save_batch_metadata_to_parquet,
)
from src.utils.merge_data import (
    extract_metadata_from_csv,
    folder_has_all_modalities,
    matching_image_audios,
)


def create_species_dataset(root_folder: str, filepath: str) -> pd.DataFrame:
    """
    This function creates a species dataset by walking through the folder structure, loading image, audio,
    and eDNA metadata, and matching the images with audios based on metadata such as latitude, longitude,
    and timestamp.

    Args:
        root_folder (str): The path to the root folder containing species subfolders, each with images, audios, and eDNA files.
        filepath (str): The path to the output file to save processed data.

    Returns:
        None.
    """

    max_rows_per_species: int = 20
    rows = []

    for species_folder in os.listdir(root_folder):

        species_path = os.path.join(root_folder, species_folder)
        if not os.path.isdir(species_path):
            continue

        if not folder_has_all_modalities(species_path):
            continue

        image_metadata_list = []
        audio_metadata_list = []
        edna_file_path = None
        description_file_path = None

        for file_name in os.listdir(species_path):
            file_path = os.path.join(species_path, file_name)

            if file_name.endswith("_image.csv"):
                image_metadata = extract_metadata_from_csv(file_path, "image")
                image_metadata_list.append(image_metadata)
            elif file_name.endswith("_audio.csv"):
                audio_metadata = extract_metadata_from_csv(file_path, "audio")
                audio_metadata_list.append(audio_metadata)
            elif file_name.endswith("_edna.csv"):
                edna_file_path = file_path
            elif file_name.endswith("_description.csv"):
                description_file_path = file_path

        image_metadata_df = pd.DataFrame(image_metadata_list)
        audio_metadata_df = pd.DataFrame(audio_metadata_list)

        matched_data = matching_image_audios(image_metadata_df, audio_metadata_df)

        if len(matched_data) > max_rows_per_species:
            matched_data = matched_data[:max_rows_per_species]

        edna_data = pd.read_csv(edna_file_path) if edna_file_path else pd.DataFrame()
        description_data = (
            pd.read_csv(description_file_path)
            if description_file_path
            else pd.DataFrame()
        )

        for match in matched_data:

            image_file = match["Image_path"]
            audio_file = match["Audio_path"]

            image_features = preprocess_image(
                image_file,
                crop=False,
                denoise=True,
                blur=False,
                augmentation=True,
                normalize=True,
            )

            image_tensor = image_features["image_tensor"]

            audio_features = preprocess_audio(
                audio_file,
                rem_silence=False,
                red_noise=True,
                resample=True,
                normalize=True,
                mfcc=True,
            )

            waveform = audio_features["waveform"]

            edna_features = (
                preprocess_edna(
                    edna_file_path,
                    clean=True,
                    replace_ambiguous=True,
                    threshold=0.5,
                    extract_kmer=True,
                    k=4,
                    one_hot_encode=False,
                    max_length=512,
                    vectorize_kmers=True,
                )
                if edna_file_path
                else None
            )

            kmer_vector = edna_features["kmer_vector"]

            if not edna_data.empty:
                biological_features = {
                    "Phylum": edna_data["Phylum"].iloc[0],
                    "Class": edna_data["Class"].iloc[0],
                    "Order": edna_data["Order"].iloc[0],
                    "Family": edna_data["Family"].iloc[0],
                    "Genus": edna_data["Genus"].iloc[0],
                }
            else:
                biological_features = {
                    key: None for key in ["Phylum", "Class", "Order", "Family", "Genus"]
                }

            if not description_data.empty:
                description_features = (
                    preprocess_text(
                        description_data["description"].iloc[0],
                        clean=True,
                        use_bert=True,
                        max_length=512,
                    )
                    if "description" in description_data.columns
                    else None
                )
                redlist = (
                    description_data["redlist"].iloc[0]
                    if "redlist" in description_data.columns
                    else None
                )
            else:
                description_features, redlist = None, None

            bert_embeddings = description_features["bert_embeddings"]

            row = {
                "Species": species_folder,
                "Image": image_tensor.numpy().tolist(),
                "Audio": waveform.numpy().tolist(),
                "Latitude": match["Latitude"],
                "Longitude": match["Longitude"],
                "Timestamp": match["Timestamp"],
                "eDNA": kmer_vector.tolist(),
                "Phylum": biological_features["Phylum"],
                "Class": biological_features["Class"],
                "Order": biological_features["Order"],
                "Family": biological_features["Family"],
                "Genus": biological_features["Genus"],
                "Description": bert_embeddings.numpy().tolist(),
                "Redlist": redlist,
            }

            rows.append(row)

    species_dataset = pd.DataFrame(rows)
    normalized_dataset = normalize_species_dataset(species_dataset)

    if "Timestamp" in normalized_dataset.columns:
        normalized_dataset["Timestamp"] = pd.to_datetime(
            normalized_dataset["Timestamp"], errors="coerce", utc=True
        )

    save_as_parquet(normalized_dataset, filepath)


def create_batches(
    single_variables_dataset: xarray.Dataset,
    surface_variables_dataset: xarray.Dataset,
    pressure_variables_dataset: xarray.Dataset,
    species_dataset: pd.DataFrame,
) -> list[DataBatch]:
    """
    Create DataBatches from the multimodal and ERA5 datasets by padding and batching image,
    audio, and eDNA data, along with surface, single-level, and pressure-level ERA5 data.

    Args:
        single_variables_dataset (xarray.Dataset): The single-level ERA5 dataset.
        surface_variables_dataset (xarray.Dataset): The surface-level ERA5 dataset.
        pressure_variables_dataset (xarray.Dataset): The pressure-level ERA5 dataset.
        species_dataset (pd.DataFrame): The multimodal dataset containing species data.
        filepath (str): The directory where the batch files will be saved.

    Returns:
        list[DataBatch]: A list of DataBatch objects.
    """

    os.makedirs(BATCHES_DATA_DIR, exist_ok=True)
    print("start")

    image_tensors = torch.stack(species_dataset["Image"].tolist())
    audio_tensors = pad_tensors(species_dataset["Audio"].tolist())
    edna_tensors = torch.stack(species_dataset["eDNA"].tolist())

    species_names = torch.tensor(species_dataset["Species"].values, dtype=torch.int64)
    phylum = torch.tensor(species_dataset["Phylum"].values, dtype=torch.int64)
    class_ = torch.tensor(species_dataset["Class"].values, dtype=torch.int64)
    order = torch.tensor(species_dataset["Order"].values, dtype=torch.int64)
    family = torch.tensor(species_dataset["Family"].values, dtype=torch.int64)
    genus = torch.tensor(species_dataset["Genus"].values, dtype=torch.int64)
    description = pad_tensors(species_dataset["Description"].tolist())
    redlist = torch.tensor(species_dataset["Redlist"].values, dtype=torch.int64)

    surface_len = len(surface_variables_dataset.valid_time)
    atmospheric_len = len(pressure_variables_dataset.valid_time)
    species_len = len(species_dataset)

    max_len = max(surface_len, atmospheric_len, species_len)

    batches = []

    for i in range(max_len):
        print(i)

        surface_index = i % surface_len
        atmospheric_index = i % atmospheric_len
        species_index = i % species_len

        batch = DataBatch(
            surface_variables={
                "t2m": torch.from_numpy(
                    surface_variables_dataset["t2m"].values[
                        [surface_index - 1, surface_index]
                    ][None]
                ),
                "u10": torch.from_numpy(
                    surface_variables_dataset["u10"].values[
                        [surface_index - 1, surface_index]
                    ][None]
                ),
                "v10": torch.from_numpy(
                    surface_variables_dataset["v10"].values[
                        [surface_index - 1, surface_index]
                    ][None]
                ),
                "msl": torch.from_numpy(
                    surface_variables_dataset["msl"].values[
                        [surface_index - 1, surface_index]
                    ][None]
                ),
            },
            single_variables={
                "z": torch.from_numpy(single_variables_dataset["z"].values[0]),
                "lsm": torch.from_numpy(single_variables_dataset["lsm"].values[0]),
                "slt": torch.from_numpy(single_variables_dataset["slt"].values[0]),
            },
            atmospheric_variables={
                "t": torch.from_numpy(
                    pressure_variables_dataset["t"].values[
                        [atmospheric_index - 1, atmospheric_index]
                    ][None]
                ),
                "u": torch.from_numpy(
                    pressure_variables_dataset["u"].values[
                        [atmospheric_index - 1, atmospheric_index]
                    ][None]
                ),
                "v": torch.from_numpy(
                    pressure_variables_dataset["v"].values[
                        [atmospheric_index - 1, atmospheric_index]
                    ][None]
                ),
                "q": torch.from_numpy(
                    pressure_variables_dataset["q"].values[
                        [atmospheric_index - 1, atmospheric_index]
                    ][None]
                ),
                "z": torch.from_numpy(
                    pressure_variables_dataset["z"].values[
                        [atmospheric_index - 1, atmospheric_index]
                    ][None]
                ),
            },
            image_variables={"Image": image_tensors[species_index]},
            audio_variables={"Audio": audio_tensors[species_index]},
            edna_variables={"eDNA": edna_tensors[species_index]},
            species_names=species_names[species_index],
            phylum=phylum[species_index],
            class_=class_[species_index],
            order=order[species_index],
            family=family[species_index],
            genus=genus[species_index],
            description=description[species_index],
            redlist=redlist[species_index],
            batch_metadata=BatchMetadata(
                era5_latitude=torch.from_numpy(
                    surface_variables_dataset.latitude.values
                ),
                era5_longitude=torch.from_numpy(
                    surface_variables_dataset.longitude.values
                ),
                era5_timestamp=(
                    surface_variables_dataset.valid_time.values.astype(
                        "datetime64[s]"
                    ).tolist()[surface_index],
                ),
                pressure_levels=tuple(
                    int(level)
                    for level in pressure_variables_dataset.pressure_level.values
                ),
                species_latitude=torch.tensor(
                    species_dataset["Latitude"].values[species_index]
                ),
                species_longitude=torch.tensor(
                    species_dataset["Longitude"].values[species_index]
                ),
                species_timestamp=(species_dataset["Timestamp"].values[species_index],),
            ),
        )
        batch_file = os.path.join(BATCHES_DATA_DIR, f"batch_{i}.pt")
        torch.save(batch, batch_file)

        batches.append(batch)

    return batches


def create_dataset(
    parquet_file: str,
    surface_file: str,
    single_file: str,
    pressure_file: str,
    batch_metadata_file: str,
) -> list[DataBatch]:
    """
    Create DataBatches from the multimodal and ERA5 datasets and save the resulting batches
    and batch metadata for future use.

    Args:
        parquet_file (str): Path to the Parquet file for multimodal data.
        surface_file (str): Path to the ERA5 surface dataset.
        single_file (str): Path to the ERA5 single-level dataset.
        pressure_file (str): Path to the ERA5 pressure-level dataset.
        batch_metadata_file (str): Path to the Parquet file where batch metadata will be stored.

    Returns:
        list[DataBatch]: A list of DataBatch objects containing multimodal and ERA5 data.
    """
    # species_dataset = load_species_data(parquet_file, 10)

    (
        surface_variables_dataset,
        single_variables_dataset,
        pressure_variables_dataset,
    ) = load_era5_datasets(surface_file, single_file, pressure_file)

    batches = []

    for species_chunk in load_species_data(parquet_file, 5):
        batch_chunk = create_batches(
            single_variables_dataset=single_variables_dataset,
            surface_variables_dataset=surface_variables_dataset,
            pressure_variables_dataset=pressure_variables_dataset,
            species_dataset=species_chunk,
        )
    batches.extend(batch_chunk)

    save_batch_metadata_to_parquet(batches, batch_metadata_file)

    return batches

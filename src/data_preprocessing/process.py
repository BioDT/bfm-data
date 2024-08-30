# src/data_preprocessing/main.py

import torch
import torchaudio
import torchvision.transforms as transforms
import xarray
from PIL import Image

from src.data_preprocessing.batch import DataBatch
from src.data_preprocessing.cleaning.audio import reduce_noise, remove_silence
from src.data_preprocessing.cleaning.image import denoise_image, resize_crop_image
from src.data_preprocessing.metadata import BatchMetadata
from src.data_preprocessing.transformation.audio import normalise_audio, resample_audio
from src.data_preprocessing.transformation.image import (
    augment_image,
    convert_color_space,
    denormalise_tensor,
    normalise_image,
)


def process_audio(
    file_path: str,
    output_path: str,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.5,
    noise_reduce_factor: float = 0.1,
    target_sample_rate: int = 16000,
) -> None:
    """
    Processes an audio file by removing silence, reducing noise, normalising, and resampling.

    Args:
        file_path (str): The path to the input audio file.
        output_path (str): The path to save the processed audio file.
        silence_threshold (float): Amplitude threshold below which audio is considered silence. Default is 0.01.
        min_silence_duration (float): Minimum duration (in seconds) of silence to be removed. Default is 0.5.
        noise_reduce_factor (float): Factor by which to reduce noise. Default is 0.1.
        target_sample_rate (int): The target sample rate to resample to. Default is 16000 Hz.

    Returns:
        None
    """
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = remove_silence(
        waveform, sample_rate, silence_threshold, min_silence_duration
    )
    # waveform = reduce_noise(waveform, noise_reduce_factor)
    waveform = normalise_audio(waveform)
    waveform = resample_audio(waveform, sample_rate, target_sample_rate)

    torchaudio.save(output_path, waveform, target_sample_rate)


def process_image(
    image_path: str,
    output_path: str,
    size: tuple = (224, 224),
    mean: list = [0.5, 0.5, 0.5],
    std: list = [0.5, 0.5, 0.5],
) -> None:
    """
    Processes an image by resizing, normalising, augmenting, optionally converting to grayscale, and denoising.

    Args:
        image_path (str): The path to the input image file.
        output_path (str): The path to save the processed image file.
        size (tuple): The desired dimensions (width, height) for resizing. Default is (224, 224).
        mean (list): The mean values for each channel for normalisation. Default is [0.5, 0.5, 0.5].
        std (list): The standard deviation values for each channel for normalisation. Default is [0.5, 0.5, 0.5].
    """
    image = Image.open(image_path)

    resized_image = resize_crop_image(image, size)
    # augmented_image = augment_image(resized_image)

    normalised_tensor = normalise_image(resized_image, mean, std)
    denormalised_tensor = denormalise_tensor(normalised_tensor)

    transform_to_pil = transforms.ToPILImage()
    image = transform_to_pil(denormalised_tensor)

    # colored_image = convert_color_space(image)

    image = denoise_image(image)
    image.save(output_path)


def process_era5(
    single_variables_dataset: xarray.Dataset,
    surface_variables_dataset: xarray.Dataset,
    pressure_variables_dataset: xarray.Dataset,
):
    i = 1

    longitudes = surface_variables_dataset.longitude.values
    longitudes = (longitudes + 360) % 360

    batch = DataBatch(
        surface_variables={
            "2t": torch.from_numpy(
                surface_variables_dataset["t2m"].values[[i - 1, i]][None]
            ),
            "10u": torch.from_numpy(
                surface_variables_dataset["u10"].values[[i - 1, i]][None]
            ),
            "10v": torch.from_numpy(
                surface_variables_dataset["v10"].values[[i - 1, i]][None]
            ),
            "msl": torch.from_numpy(
                surface_variables_dataset["msl"].values[[i - 1, i]][None]
            ),
        },
        static_variables={
            "z": torch.from_numpy(single_variables_dataset["z"].values[0]),
            "t": torch.from_numpy(single_variables_dataset["t"].values[0]),
            "q": torch.from_numpy(single_variables_dataset["q"].values[0]),
        },
        atmospheric_variables={
            "t": torch.from_numpy(
                pressure_variables_dataset["t"].values[[i - 1, i]][None]
            ),
            "u": torch.from_numpy(
                pressure_variables_dataset["u"].values[[i - 1, i]][None]
            ),
            "v": torch.from_numpy(
                pressure_variables_dataset["v"].values[[i - 1, i]][None]
            ),
            "q": torch.from_numpy(
                pressure_variables_dataset["q"].values[[i - 1, i]][None]
            ),
            "z": torch.from_numpy(
                pressure_variables_dataset["z"].values[[i - 1, i]][None]
            ),
        },
        batch_metadata=BatchMetadata(
            latitude=torch.from_numpy(surface_variables_dataset.latitude.values),
            longitude=torch.from_numpy(longitudes),
            timestamp=(
                surface_variables_dataset.time.values.astype("datetime64[s]").tolist()[
                    i
                ],
            ),
            pressure_levels=tuple(
                int(level) for level in pressure_variables_dataset.level.values
            ),
        ),
    )

    return batch

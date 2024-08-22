# src/data_preprocessing/main.py

import torchaudio
import torchvision.transforms as transforms
from PIL import Image

from src.data_preprocessing.cleaning.audio import (
    normalise_audio,
    reduce_noise,
    remove_silence,
)
from src.data_preprocessing.cleaning.image import denoise_image, resize_crop_image
from src.data_preprocessing.transformation.audio import resample_audio
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
    waveform = reduce_noise(waveform, noise_reduce_factor)
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

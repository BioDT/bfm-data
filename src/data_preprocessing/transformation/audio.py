# src/data_preprocessing/transformation/audio.py

import torch
import torchaudio


def resample_audio(
    waveform: torch.Tensor, original_sample_rate: int, target_sample_rate: int = 16000
) -> torch.Tensor:
    """
    Resamples the audio waveform to a target sample rate.

    Args:
    waveform (torch.Tensor): The input audio waveform.
    original_sample_rate (int): The original sample rate of the audio waveform.
    target_sample_rate (int): The target sample rate to resample to. Default is 16000 Hz.

    Returns:
    torch.Tensor: The resampled audio waveform.
    """
    resampler = torchaudio.transforms.Resample(
        orig_freq=original_sample_rate, new_freq=target_sample_rate
    )
    return resampler(waveform)


def convert_to_spectrogram(audio: torch.Tensor) -> torch.Tensor:
    """
    Converts audio to a spectrogram.

    Args:
        audio (torch.Tensor): The input audio tensor.
        sample_rate (int): The sample rate of the audio.

    Returns:
        torch.Tensor: The spectrogram tensor.
    """
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=1024, hop_length=512, power=None
    )
    spectrogram = spectrogram_transform(audio)
    return spectrogram


def augment_audio(
    audio: torch.Tensor,
    sample_rate: int,
    noise_factor: float = 0.005,
    shift_factor: float = 0.2,
    speed_factor: float = 1.2,
) -> torch.Tensor:
    """
    Applies audio augmentation techniques: adds noise, shifts time, and changes speed.

    Args:
        audio (torch.Tensor): The input audio tensor.
        sample_rate (int): The sample rate of the audio.
        noise_factor (float): Factor by which noise is added.
        shift_factor (float): Factor for time shifting (proportion of total length).
        speed_factor (float): Factor for changing the speed (1.0 = no change).

    Returns:
        torch.Tensor: The augmented audio tensor.
    """
    noise = torch.randn_like(audio) * noise_factor
    augmented_audio = audio + noise

    shift_amount = int(sample_rate * shift_factor)
    augmented_audio = torch.roll(augmented_audio, shift_amount)

    augmented_audio = torchaudio.transforms.Resample(
        sample_rate, int(sample_rate * speed_factor)
    )(augmented_audio)

    return augmented_audio

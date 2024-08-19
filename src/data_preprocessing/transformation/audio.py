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

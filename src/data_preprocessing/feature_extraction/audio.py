# src/data_preprocessing/feature_extraction/audio.py

import torch
import torchaudio


def pad_waveform(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pads the waveform to the target length with zeros if it's shorter.

    Args:
        waveform (torch.Tensor): The input audio waveform.
        target_length (int): The minimum length to pad the waveform to.

    Returns:
        torch.Tensor: The padded waveform.
    """
    if waveform.shape[1] < target_length:
        padding_size = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding_size))
    return waveform


def extract_mfcc(
    waveform: torch.Tensor, sample_rate: int, n_mfcc: int = 13
) -> torch.Tensor:
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio waveform.

    Args:
        waveform (torch.Tensor): The input audio waveform.
        sample_rate (int): The sample rate of the audio waveform.
        n_mfcc (int): The number of MFCC features to extract. Default is 13.

    Returns:
        torch.Tensor: The extracted MFCC features.
    """
    n_fft = 512
    min_waveform_length = n_fft
    waveform = pad_waveform(waveform, min_waveform_length)
    hop_length = 160

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": 13,
            "center": False,
        },
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

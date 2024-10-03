# src/data_preprocessing/feature_extraction/audio.py

import torch
import torchaudio

from src.data_preprocessing.transformation.audio import pad_or_truncate


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
    waveform = pad_or_truncate(waveform, min_waveform_length)
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

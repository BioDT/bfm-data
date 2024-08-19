# src/data_preprocessing/feature_extraction/audio.py

import torch
import torchaudio


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
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

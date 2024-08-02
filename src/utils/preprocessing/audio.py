# src/utils/preprocessing/audio.py

import torch
import torchaudio


def process_audio(
    file_path: str,
    output_path: str,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.5,
    noise_reduce_factor: float = 0.1,
    target_sample_rate: int = 16000,
) -> None:
    """
    Processes an audio file by removing silence, reducing noise, normalizing, and resampling.

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
    waveform = normalize_audio(waveform)
    waveform = resample_audio(waveform, sample_rate, target_sample_rate)

    torchaudio.save(output_path, waveform, target_sample_rate)


def remove_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.5,
) -> torch.Tensor:
    """
    Removes silenced segments from an audio waveform.

    Args:
        waveform (torch.Tensor): The input audio waveform.
        sample_rate (int): The sample rate of the audio waveform.
        silence_threshold (float): Amplitude threshold below which audio is considered silence. Default is 0.01.
        min_silence_duration (float): Minimum duration (in seconds) of silence to be removed. Default is 0.5.

    Returns:
        torch.Tensor: The audio waveform with silenced segments removed.
    """
    min_silence_samples = int(min_silence_duration * sample_rate)

    silence_mask = torch.abs(waveform) < silence_threshold
    silence_mask = silence_mask.float()

    silence_regions = (
        torch.nn.functional.conv1d(
            silence_mask.unsqueeze(0),
            torch.ones(1, 1, min_silence_samples),
            padding=min_silence_samples // 2,
        ).squeeze()
        > min_silence_samples // 2
    )

    silence_changes = silence_regions[1:] != silence_regions[:-1]
    silence_indices = torch.nonzero(silence_changes).squeeze().tolist()

    silence_times = [index / sample_rate for index in silence_indices]
    silence_segments = [
        (silence_times[i], silence_times[i + 1])
        for i in range(0, len(silence_times), 2)
    ]

    non_silent_waveform = []
    start_idx = 0
    for start_time, end_time in silence_segments:
        end_idx = int(start_time * sample_rate)
        non_silent_waveform.append(waveform[:, start_idx:end_idx])
        start_idx = int(end_time * sample_rate)

    non_silent_waveform.append(waveform[:, start_idx:])
    non_silent_waveform = torch.cat(non_silent_waveform, dim=1)

    return non_silent_waveform


def reduce_noise(
    waveform: torch.Tensor, noise_reduce_factor: float = 0.1
) -> torch.Tensor:
    """
    Reduces noise from an audio waveform using spectral gating.

    Args:
        waveform (torch.Tensor): The input audio waveform.
        noise_reduce_factor (float): Factor by which to reduce noise. Default is 0.1.

    Returns:
        torch.Tensor: The audio waveform with reduced noise.
    """
    n_fft = 1024
    hop_length = 512
    window = torch.hann_window(n_fft)

    stft = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
    )
    magnitude, phase = torch.abs(stft), torch.angle(stft)
    noise_mag = torch.mean(magnitude, dim=-1, keepdim=True)
    reduced_mag = torch.max(
        magnitude - noise_reduce_factor * noise_mag, torch.tensor(0.0)
    )
    reduced_stft = reduced_mag * torch.exp(1j * phase)
    reduced_waveform = torch.istft(
        reduced_stft, n_fft=n_fft, hop_length=hop_length, window=window
    )
    return reduced_waveform


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the audio waveform to have zero mean and unit variance.

    Args:
        waveform (torch.Tensor): The input audio waveform.

    Returns:
        torch.Tensor: The normalized audio waveform.
    """
    waveform -= waveform.mean()
    waveform /= waveform.abs().max()
    return waveform


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

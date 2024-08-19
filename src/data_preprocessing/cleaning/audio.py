# src/data_preprocessing/cleaning/audio.py

import torch
import torchaudio


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

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
        n_fft=1024, hop_length=512, power=2.0
    )
    spectrogram = spectrogram_transform(audio)
    return spectrogram


def convert_to_log_mel_spectrogram(
    audio: torch.Tensor, sample_rate: int
) -> torch.Tensor:
    """
    Converts audio to a log-mel spectrogram.

    Args:
        audio (torch.Tensor): The input audio tensor.
        sample_rate (int): The sample rate of the audio.

    Returns:
        torch.Tensor: The log-mel spectrogram tensor.
    """
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=128
    )

    mel_spectrogram = mel_spectrogram_transform(audio)

    log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

    return log_mel_spectrogram


def pad_or_truncate(waveform: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Pads or truncates the waveform to the specified max_length.

    Args:
        waveform (torch.Tensor): The input audio waveform.
        max_length (int): The maximum length to pad or truncate the waveform to.

    Returns:
        torch.Tensor: The padded or truncated waveform.
    """

    if waveform.size(1) < max_length:
        padding_size = max_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding_size))

    elif waveform.size(1) > max_length:
        waveform = waveform[:, :max_length]

    return waveform


def pad_or_truncate_audio_features(
    audio_features: torch.Tensor, target_length: int
) -> torch.Tensor:
    """
    Pads or truncates audio features (log-mel spectrogram or MFCC) to a fixed length (target_length).

    Args:
        audio_features (torch.Tensor): The audio feature tensor (C, F, T) where C is the number of channels,
                                       F is the number of mel-frequency bins (or MFCC coefficients),
                                       and T is the time dimension (number of frames).
        target_length (int): The target number of time frames for the output audio features.

    Returns:
        torch.Tensor: The padded or truncated audio feature tensor with shape (C, F, target_length).
    """
    current_length = audio_features.size(-1)

    if current_length < target_length:
        padding = target_length - current_length
        audio_features = torch.nn.functional.pad(
            audio_features, (0, padding), "constant", 0
        )
    elif current_length > target_length:
        audio_features = audio_features[..., :target_length]

    return audio_features


def convert_stereo_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Converts stereo waveform to mono by averaging the two channels.

    Args:
        waveform (torch.Tensor): The mono waveform tensor, shape (2, ...), where 2 is the number of channels.

    Returns:
        torch.Tensor: The stereo waveform tensor, shape (1, ...), where 1 is the number of channels.
    """
    if waveform.size(0) == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
        return waveform
    else:
        return waveform


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


def normalise_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalises the audio waveform to have zero mean and unit variance.

    Args:
        waveform (torch.Tensor): The input audio waveform.

    Returns:
        torch.Tensor: The normalised audio waveform.
    """
    waveform -= waveform.mean()
    waveform /= waveform.abs().max()
    return waveform


def resize_audio_tensor(
    audio_tensor: torch.Tensor, target_shape: tuple
) -> torch.Tensor:
    """
    Pads or truncates an audio tensor to the target shape, aligning it to the specified length.

    Args:
        audio_tensor (torch.Tensor): The input audio tensor with shape (1, length, 1).
        target_shape (tuple): The desired shape for the audio (e.g., (1, 13, 1)).

    Returns:
        torch.Tensor: The resized or padded audio tensor.
    """
    padded_audio = torch.zeros(target_shape)
    length = min(audio_tensor.size(-2), target_shape[-2])
    padded_audio[..., :length, :] = audio_tensor[..., :length, :]
    return padded_audio

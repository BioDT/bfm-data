# src/data_preprocessing/preprocess.py


import torchaudio
from PIL import Image

from src.data_preprocessing.cleaning.audio import reduce_noise, remove_silence
from src.data_preprocessing.cleaning.edna import clean_sequence, replace_ambiguous_bases
from src.data_preprocessing.cleaning.image import (
    blur_image,
    denoise_image,
    resize_crop_image,
)
from src.data_preprocessing.cleaning.text import clean_text
from src.data_preprocessing.feature_extraction.audio import extract_mfcc
from src.data_preprocessing.feature_extraction.edna import extract_kmer_frequencies
from src.data_preprocessing.feature_extraction.image import (
    calculate_color_histogram,
    extract_hog_features,
)
from src.data_preprocessing.feature_extraction.text import extract_tfidf_features
from src.data_preprocessing.transformation.audio import (
    augment_audio,
    convert_stereo_to_mono,
    convert_to_log_mel_spectrogram,
    convert_to_spectrogram,
    normalise_audio,
    resample_audio,
)
from src.data_preprocessing.transformation.edna import (
    normalize_kmer_vector,
    one_hot_encode_sequence,
    vectorize_kmer_frequencies,
)
from src.data_preprocessing.transformation.image import (
    augment_image,
    convert_color_space,
    equalize_histogram,
    normalise_image,
)
from src.data_preprocessing.transformation.text import (
    bert_tokenizer,
    pad_or_truncate_embeddings,
    reduce_embedding_dimensions,
)


def preprocess_image(
    image_path: str,
    resize_size: tuple = (64, 64),
    crop: bool = False,
    denoise: bool = False,
    blur: bool = False,
    kernel_size: int = 5,
    augmentation: bool = False,
    color_space: str = None,
    normalize: bool = False,
    mean: list = [0.5, 0.5, 0.5],
    std: list = [0.5, 0.5, 0.5],
    hog_features: bool = False,
    histogram_equalization: bool = False,
) -> dict:
    """
    A general preprocessing pipeline for images, supporting resizing, denoising, blurring, augmentation,
    color space conversion, edge detection, histogram equalization, and feature extraction.

    Args:
        image_path (str): The path to the input image.
        resize_size (tuple): The target size for resizing (width, height).
        crop (bool): Whether to crop the image instead of resizing.
        denoise (bool): Whether to apply noise reduction.
        blur (bool): Whether to apply Gaussian blurring.
        kernel_size (int): The size of the Gaussian kernel for blurring.
        augmentation (bool): Whether to apply augmentation.
        color_space (str): Convert the image to a different color space ('gray', 'hsv', 'lab').
        normalize (bool): Whether to normalize the image tensor.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
        hog_features (bool): Whether to extract HOG features.
        histogram_equalization (bool): Whether to apply histogram equalization.

    Returns:
        dict: A dictionary containing the processed image, any extracted features, and transformations applied.
    """

    image = Image.open(image_path).convert("RGB")
    processed_image = image

    if crop:
        processed_image = resize_crop_image(processed_image, resize_size, crop=True)
    else:
        processed_image = resize_crop_image(processed_image, resize_size, crop=False)

    if denoise:
        processed_image = denoise_image(processed_image)

    if blur:
        processed_image = blur_image(processed_image, kernel_size)

    if augmentation:
        processed_image = augment_image(processed_image)

    if color_space:
        processed_image = convert_color_space(processed_image, color_space)

    if normalize:
        normalised_image = normalise_image(processed_image, mean, std)

    hog_feature_vector = None
    hog_image = None
    if hog_features:
        hog_feature_vector, hog_image = extract_hog_features(processed_image)

    if histogram_equalization:
        processed_image = equalize_histogram(processed_image)

    color_histogram = calculate_color_histogram(processed_image)

    return {
        "processed_image": processed_image,
        "normalised_image": normalised_image if normalize else None,
        "color_histogram": color_histogram,
        "hog_feature_vector": hog_feature_vector,
        "hog_image": hog_image,
    }


def preprocess_audio(
    audio_path: str,
    sample_rate: int = 16000,
    rem_silence: bool = False,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.5,
    red_noise: bool = False,
    noise_reduce_factor: float = 0.1,
    resample: bool = False,
    target_sample_rate: int = 16000,
    augment: bool = False,
    noise_factor: float = 0.005,
    shift_factor: float = 0.2,
    speed_factor: float = 1.2,
    normalize: bool = False,
    mfcc: bool = False,
    n_mfcc: int = 13,
    convert_spectrogram: bool = False,
    convert_log_mel_spectrogram: bool = False,
) -> dict:
    """
    A general preprocessing pipeline for audio that handles silence removal, noise reduction, resampling,
    augmentation, normalization, and feature extraction (MFCC or Spectrogram).

    Args:
        audio_path (str): The path to the input audio file.
        sample_rate (int): The sample rate of the audio. Default is 16000 Hz.
        rem_silence (bool): If True, remove silent parts of the audio.
        silence_threshold (float): Amplitude threshold below which audio is considered silence.
        min_silence_duration (float): Minimum duration of silence to remove.
        red_noise (bool): If True, reduce noise using spectral gating.
        noise_reduce_factor (float): Factor to reduce noise by.
        resample (bool): If True, resample the audio to the target sample rate.
        target_sample_rate (int): The target sample rate to resample to.
        augment (bool): If True, apply audio augmentation (noise, time shift, speed change).
        noise_factor (float): Factor to add noise for augmentation.
        shift_factor (float): Factor for time shifting during augmentation.
        speed_factor (float): Factor to change the speed of audio during augmentation.
        normalize (bool): If True, normalize the audio waveform.
        mfcc (bool): If True, extract MFCC features.
        n_mfcc (int): The number of MFCC features to extract.
        convert_spectrogram (bool): If True, convert the audio to a spectrogram.
        convert_log_mel_spectrogram (bool): If True, convert the audio to a log mel spectrogram.

    Returns:
        dict: A dictionary containing processed audio waveform and extracted features.
    """

    waveform, original_sample_rate = torchaudio.load(audio_path)

    if resample and original_sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, original_sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    waveform = convert_stereo_to_mono(waveform)

    if rem_silence:
        waveform = remove_silence(
            waveform, sample_rate, silence_threshold, min_silence_duration
        )

    if red_noise:
        waveform = reduce_noise(waveform, noise_reduce_factor)

    if augment:
        waveform = augment_audio(
            waveform, sample_rate, noise_factor, shift_factor, speed_factor
        )

    if normalize:
        waveform = normalise_audio(waveform)

    mfcc_features = None
    if mfcc:
        mfcc_features = extract_mfcc(waveform, sample_rate, n_mfcc)

    spectrogram = None
    if convert_spectrogram:
        spectrogram = convert_to_spectrogram(waveform)

    log_mel_spectrogram = None
    if convert_log_mel_spectrogram:
        log_mel_spectrogram = convert_to_log_mel_spectrogram(waveform, sample_rate)

    return {
        "waveform": waveform,
        "mfcc_features": mfcc_features,
        "spectrogram": spectrogram,
        "log_mel_spectrogram": log_mel_spectrogram,
    }


def preprocess_edna(
    edna_sequence: str,
    clean: bool = True,
    replace_ambiguous: bool = True,
    threshold: float = 0.05,
    extract_kmer: bool = True,
    k: int = 4,
    one_hot_encode: bool = False,
    max_length: int = 512,
    vectorize_kmers: bool = False,
    normalise: bool = False,
    vocab_size: int = None,
) -> dict:
    """
    A general preprocessing pipeline for eDNA sequences that handles cleaning, replacing ambiguous bases,
    k-mer frequency extraction, GC content calculation, one-hot encoding, and k-mer vectorization.

    Args:
        edna_sequence (str): The raw eDNA sequence to preprocess.
        clean (bool): If True, clean the sequence by removing non-ATCG characters.
        replace_ambiguous (bool): If True, replace ambiguous bases (e.g., N) with common nucleotides.
        threshold (float): Proportion of ambiguous bases allowed before discarding.
        extract_kmer (bool): If True, extract k-mer frequencies from the sequence.
        k (int): The length of k-mers to extract.
        one_hot_encode (bool): If True, one-hot encode the sequence.
        max_length (int): Maximum length for one-hot encoding.
        vectorize_kmers (bool): If True, vectorize k-mer frequencies.
        normalise (bool): If True, normalise the sequence.
        vocab_size (int): The size of the k-mer vocabulary for vectorization. If None, inferred from k.

    Returns:
        dict: A dictionary containing the processed eDNA sequence and extracted features.
    """

    if clean:
        edna_sequence = clean_sequence(edna_sequence)

    if replace_ambiguous:
        edna_sequence = replace_ambiguous_bases(edna_sequence, threshold)

    kmer_frequencies = None
    if extract_kmer:
        kmer_frequencies = extract_kmer_frequencies(edna_sequence, k)

    one_hot_encoded_sequence = None
    if one_hot_encode:
        one_hot_encoded_sequence = one_hot_encode_sequence(edna_sequence, max_length)

    kmer_vector = None
    if vectorize_kmers and kmer_frequencies is not None:
        kmer_vector = vectorize_kmer_frequencies(kmer_frequencies, k, vocab_size)

    normalised_vector = None
    if normalise and kmer_vector is not None:
        normalised_vector = normalize_kmer_vector(kmer_vector)

    return {
        "cleaned_sequence": edna_sequence,
        "kmer_frequencies": kmer_frequencies,
        "one_hot_encoded_sequence": one_hot_encoded_sequence,
        "kmer_vector": kmer_vector,
        "normalised_vector": normalised_vector,
    }


def preprocess_text(
    corpus: list,
    clean: bool = True,
    extract_tfidf: bool = False,
    use_bert: bool = True,
    max_length: int = 512,
) -> dict:
    """
    A general text preprocessing pipeline that handles cleaning, feature extraction (Bag of Words, TF-IDF, GloVe, BERT),
    one-hot encoding, label encoding, and more.

    Args:
        corpus (list): The input text corpus.
        clean (bool): If True, clean the text by lowercasing, removing punctuation, etc.
        extract_tfidf (bool): If True, extract TF-IDF features.
        use_bert (bool): If True, extract BERT embeddings.
        max_length (int): Maximum sequence length for BERT tokenization.

    Returns:
        dict: A dictionary containing the preprocessed corpus and extracted features.
    """

    if clean:
        corpus = [clean_text(text) for text in corpus]

    tfidf_features = None
    if extract_tfidf:
        tfidf_features = extract_tfidf_features(corpus)

    bert_embeddings = None
    if use_bert:
        bert_embeddings = bert_tokenizer(corpus, max_length)

    padded_embeddings = pad_or_truncate_embeddings(bert_embeddings, 64)

    bert_embeddings = reduce_embedding_dimensions(padded_embeddings, output_dim=64)

    return {
        "tfidf_features": tfidf_features,
        "bert_embeddings": bert_embeddings,
    }

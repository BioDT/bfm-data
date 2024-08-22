# src/data_preprocessing/transformation/edna.py

from itertools import product

import numpy as np
import torch


def one_hot_encode_sequence(sequence: str, max_length: int = 512) -> torch.Tensor:
    """
    One-hot encode a DNA sequence.

    Args:
        sequence (str): Cleaned DNA sequence.
        max_length (int): Maximum sequence length for padding/truncation.

    Returns:
        torch.Tensor: One-hot encoded sequence as a tensor.
    """
    base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    one_hot = torch.zeros((4, max_length), dtype=torch.float32)

    for i, base in enumerate(sequence[:max_length]):
        if base in base_to_index:
            one_hot[base_to_index[base], i] = 1.0

    return one_hot


def vectorize_kmer_frequencies(
    kmer_frequencies: dict, k: int, vocab_size: int = None
) -> np.array:
    """
    Vectorize k-mer frequencies into a fixed-length vector.

    Args:
        kmer_frequencies (dict): Dictionary of k-mer frequencies.
        k (int): Length of k-mers.
        vocab_size (int): Size of k-mer vocabulary. If None, inferred from k.

    Returns:
        np.array: Vectorized k-mer frequencies.
    """
    if vocab_size is None:
        vocab_size = 4**k

    vector = np.zeros(vocab_size)
    kmer_to_index = {
        "".join([a, b, c, d]): i
        for i, (a, b, c, d) in enumerate(product("ACGT", repeat=k))
    }

    for kmer, freq in kmer_frequencies.items():
        if kmer in kmer_to_index:
            vector[kmer_to_index[kmer]] = freq

    return vector

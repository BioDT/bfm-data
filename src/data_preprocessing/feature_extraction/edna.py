# src/data_preprocessing/feature_extraction/edna.py

from collections import Counter


def extract_kmer_frequencies(sequence: str, k: int = 4) -> dict:
    """
    Extract k-mer frequencies from a DNA sequence.

    Args:
        sequence (str): Cleaned DNA sequence.
        k (int): Length of k-mers to extract.

    Returns:
        dict: Dictionary of k-mer frequencies.
    """
    kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)

    total_kmers = sum(kmer_counts.values())
    kmer_frequencies = {
        kmer: count / total_kmers for kmer, count in kmer_counts.items()
    }

    return kmer_frequencies

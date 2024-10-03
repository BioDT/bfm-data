# src/data_preprocessing/cleaning/edna.py

import re


def clean_sequence(sequence: str) -> str:
    """
    Clean eDNA sequence by removing non-ATCG characters.

    Args:
        sequence (str): Raw DNA sequence.

    Returns:
        str: Cleaned DNA sequence.
    """
    cleaned_sequence = re.sub(r"[^ATCG]", "", sequence.upper())
    return cleaned_sequence


def replace_ambiguous_bases(sequence: str, threshold: float = 0.05) -> str:
    """
    Replace ambiguous bases (e.g., N) with the most common nucleotide in the sequence.

    Args:
        sequence (str): DNA sequence.
        threshold (float): Proportion of ambiguous bases allowed.

    Returns:
        str: Sequence with ambiguous bases replaced, or None if the threshold is exceeded.
    """
    if not sequence:
        return None

    ambiguous_bases = {"N": "A", "Y": "C", "R": "G", "W": "A", "S": "C"}
    total_length = len(sequence)
    ambiguous_count = sum(sequence.count(base) for base in ambiguous_bases.keys())

    if ambiguous_count / total_length <= threshold:
        for base, replacement in ambiguous_bases.items():
            sequence = sequence.replace(base, replacement)
    else:
        sequence = None

    return sequence

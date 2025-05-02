"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def extract_bag_of_words(corpus: list, max_features: int = 1000) -> torch.Tensor:
    """
    Extract Bag of words features from the text corpus.

    Args:
        corpus(list of str): The input text corpus.
        max_features(int): Maximum number of features to extract.

    Returns:
        torch.Tensor: Tensor containing BoW features.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return torch.tensor(X.toarray(), dtype=torch.float32)


def extract_tfidf_features(corpus: list, max_features: int = 1000) -> torch.Tensor:
    """
    Extracts TF-IDF features from the text corpus.

    Args:
        corpus (list of str): The input text corpus.
        max_features (int): Maximum number of features to extract.

    Returns:
        torch.Tensor: Tensor containing TF-IDF features.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return torch.tensor(X.toarray(), dtype=torch.float32)


def extract_ngram_features(
    corpus: list, ngram_range: tuple = (1, 2), max_features: int = 1000
) -> torch.Tensor:
    """
    Extracts N-gram features from the text corpus.

    Args:
        corpus (list of str): The input text corpus.
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams.
        max_features (int): Maximum number of features to extract.

    Returns:
        torch.Tensor: Tensor containing N-gram features.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return torch.tensor(X.toarray(), dtype=torch.float32)

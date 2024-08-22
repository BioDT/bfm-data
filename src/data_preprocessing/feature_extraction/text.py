# src/data_preprocessing/feature_extraction/text.py

import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe


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


def extract_glove_embeddings(corpus, embedding_dim=100):
    """
    Extracts GloVe word embeddings for each word in the corpus and averages them for sentence-level representation.

    Args:
        corpus (list of str): The input text corpus.
        embedding_dim (int): The dimensionality of the GloVe embeddings (50, 100, 200, 300).

    Returns:
        torch.Tensor: Tensor containing averaged GloVe embeddings for each sentence.
    """
    tokenizer = get_tokenizer("basic_english")
    glove = GloVe(name="6B", dim=embedding_dim)

    embeddings = []

    for sentence in corpus:
        tokens = tokenizer(sentence)
        word_vectors = [glove[token] for token in tokens if token in glove.stoi]
        if word_vectors:
            sentence_embedding = torch.mean(torch.stack(word_vectors), dim=0)
        else:
            sentence_embedding = torch.zeros(embedding_dim)
        embeddings.append(sentence_embedding)

    return torch.stack(embeddings)

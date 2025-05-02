"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def clean_text(
    text: str,
    use_remove_stopwords: bool = True,
    use_lemmatization: bool = False,
    use_spelling_correction: bool = False,
) -> str:
    """
    Cleans the input text by removing punctuation, lowercasing, and optionally removing stopwords,
    applying lemmatization and spelling correctly.

    Args:
        text (str): The input text string.
        remove_stopwords (bool): Whether to remove stopwords. Defaults to True.
        use_lemmatization (bool): Whether to apply lemmatization instead of stemming. Defaults to False.
        spelling_correction (bool): Whether to spell correctly the text. Defaults to False.

    Returns:
        str: The cleaned text.
    """
    text = lower_case(text)
    text = remove_punctuation(text)

    if use_remove_stopwords:
        text = remove_stopwords(text)

    if use_lemmatization:
        text = lemmatization(text)
    else:
        text = stemming(text)

    if use_spelling_correction:
        text = spelling_correction(text)

    return text


def lower_case(text: str) -> str:
    """
    Converts all characters in the input text to lowercase.

    Args:
        text (str): The input text string.

    Returns:
        str: The text converted to lowercase.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Removes all punctuation from the input text.

    Args:
        text (str): The input text string.

    Returns:
        str: The text with all punctuation removed.
    """
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(text: str) -> str:
    """
    Removes common stopwords from the input text.

    Args:
        text (str): The input text string.

    Returns:
        str: The text with stopwords removed.
    """
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stop_words])


def stemming(text: str) -> str:
    """
    Reduces words in the text to their word stems.

    Args:
        text (str): The input text string.

    Returns:
        str: The text with words stemmed.
    """
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in str(text).split()])


def lemmatization(text: str) -> str:
    """
    Reduces words in the text to their lemma or base form.

    Args:
        text (str): The input text string.

    Returns:
        str: The text with words lemmatized.
    """
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in str(text).split()])


def spelling_correction(text: str) -> str:
    """
    Corrects spelling errors in the input text.

    Args:
        text (str): The input text string.

    Returns:
        str: The text with corrected spelling.
    """
    return str(TextBlob(text).correct())

"""Reusable text preprocessing and leakage-safe TF-IDF feature extraction.

The vectorizer, the text normaliser and the classifier are always combined in a single
scikit-learn :class:`~sklearn.pipeline.Pipeline`. That guarantees the TF-IDF vocabulary and
IDF weights are learned from the training fold only — including during cross-validation —
so no information from the test set can leak into training.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .constants import MAX_FEATURES, MIN_DF, NGRAM_RANGE, RANDOM_SEED, TEST_SIZE

_WHITESPACE_PATTERN = re.compile(r"\s+")
# Keep words, numbers and the currency/percent symbols that carry spam signal
# (prices, "free", phone numbers, "win", "£1000", "50%", ...). Other punctuation is
# dropped because it mostly adds noise to a bag-of-words model.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[$€£%]")


def normalize_text(text: object) -> str:
    """Return a lightly normalised version of ``text``.

    Only Unicode form, whitespace and letter case are touched. Numbers, currency
    symbols, punctuation and SMS abbreviations are deliberately preserved: they carry a
    large part of the spam signal, and aggressively cleaning SMS text destroys it.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = "".join(char for char in text if char.isprintable())
    return _WHITESPACE_PATTERN.sub(" ", text.lower()).strip()


def preprocess_series(messages: Sequence[object] | pd.Series) -> pd.Series:
    """Normalise a sequence of messages into a pandas Series of cleaned strings."""
    return pd.Series([normalize_text(message) for message in messages], dtype="object")


def tokenize(text: str) -> list[str]:
    """Split already-normalised text into word/number/currency tokens."""
    return _TOKEN_PATTERN.findall(text)


def build_vectorizer() -> TfidfVectorizer:
    """Create the shared TF-IDF vectorizer used by every model.

    ``preprocessor=normalize_text`` means raw SMS text can be fed straight into a fitted
    pipeline — training and inference always apply exactly the same transformation.
    """
    return TfidfVectorizer(
        preprocessor=normalize_text,
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=True,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Bundle the TF-IDF vectorizer and an estimator into one pipeline."""
    return Pipeline([("tfidf", build_vectorizer()), ("clf", estimator)])


def split_data(
    messages: Iterable[str] | pd.Series,
    labels: Iterable[int] | pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return a stratified ``X_train, X_test, y_train, y_test`` split.

    Stratification keeps the spam/ham ratio identical in both splits, which matters
    because spam is a minority class here.
    """
    return train_test_split(
        messages,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

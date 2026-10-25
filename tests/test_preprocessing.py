"""Tests for text normalisation, vectorisation and the train/test split."""

from __future__ import annotations

import pandas as pd
import pytest

from spamshield.constants import MAX_FEATURES, TEST_SIZE
from spamshield.preprocessing import (
    build_pipeline,
    build_vectorizer,
    normalize_text,
    preprocess_series,
    split_data,
    tokenize,
)


class TestNormalizeText:
    def test_lowercases_and_collapses_whitespace(self) -> None:
        assert normalize_text("  FREE   Money\n\tNOW  ") == "free money now"

    def test_preserves_numbers_currency_and_punctuation(self) -> None:
        normalized = normalize_text("You won £1000! Call 09061701461, or 50% off?")

        assert "£1000" in normalized
        assert "09061701461" in normalized
        assert "50%" in normalized

    def test_handles_missing_and_non_string_values(self) -> None:
        assert normalize_text(None) == ""
        assert normalize_text("   ") == ""
        assert normalize_text(12345) == "12345"

    def test_applies_unicode_normalisation(self) -> None:
        assert normalize_text("ＦＲＥＥ") == "free"


class TestTokenize:
    def test_keeps_words_numbers_and_currency(self) -> None:
        tokens = tokenize(normalize_text("Win £500 now, call 80082!"))

        assert tokens == ["win", "£", "500", "now", "call", "80082"]

    def test_drops_punctuation_only_tokens(self) -> None:
        assert tokenize(normalize_text("hello!!! --- ???")) == ["hello"]


class TestVectorizer:
    def test_returns_a_sparse_matrix_within_feature_limit(self, sample_messages: list[str]) -> None:
        vectorizer = build_vectorizer()
        matrix = vectorizer.fit_transform(sample_messages)

        assert hasattr(matrix, "toarray"), "expected a scipy sparse matrix"
        assert matrix.shape[0] == len(sample_messages)
        assert matrix.shape[1] <= MAX_FEATURES

    def test_preprocessor_is_applied_to_raw_input(self) -> None:
        vectorizer = build_vectorizer()
        vectorizer.fit(["FREE MONEY NOW", "free money again"])

        features = vectorizer.get_feature_names_out()
        assert "free" in features
        assert "free money" in features


class TestPipeline:
    def test_fits_and_predicts_valid_labels(self, sample_messages: list[str]) -> None:
        from sklearn.linear_model import LogisticRegression

        labels = [1, 0, 1, 0, 1, 0]
        pipeline = build_pipeline(LogisticRegression(max_iter=200))
        pipeline.fit(sample_messages, labels)

        predictions = pipeline.predict(["FREE prize, text WIN to 80082 now"])

        assert predictions.shape == (1,)
        assert predictions[0] in {0, 1}


class TestSplitData:
    def test_split_is_stratified_and_seeded(self) -> None:
        # 100 balanced samples so stratification is easy to verify.
        messages = pd.Series([f"message {i}" for i in range(100)])
        labels = pd.Series([1] * 40 + [0] * 60)

        x_train, x_test, y_train, y_test = split_data(messages, labels)

        assert len(x_test) == pytest.approx(int(100 * TEST_SIZE), abs=1)
        assert len(x_train) + len(x_test) == 100
        assert y_train.mean() == pytest.approx(labels.mean(), abs=0.05)
        assert y_test.mean() == pytest.approx(labels.mean(), abs=0.05)

        # Same seed must reproduce exactly the same split.
        again = split_data(messages, labels)
        assert x_train.tolist() == again[0].tolist()

    def test_preprocess_series_matches_normalize_text(self) -> None:
        series = preprocess_series(["  Hello   WORLD ", None])

        assert series.tolist() == ["hello world", ""]

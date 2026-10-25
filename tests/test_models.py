"""Tests for the model registry."""

from __future__ import annotations

import pytest

from spamshield.models import (
    MODEL_KEYS,
    MODEL_SPECS,
    build_model_pipeline,
    create_estimator,
    get_spec,
)

REQUIRED_ALGORITHMS = (
    "logistic_regression",
    "decision_tree",
    "naive_bayes",
    "knn",
    "neural_network",
    "svm",
)


def test_registry_contains_exactly_the_six_required_algorithms() -> None:
    assert MODEL_KEYS == REQUIRED_ALGORITHMS


def test_every_registry_entry_is_fully_described() -> None:
    for key, spec in MODEL_SPECS.items():
        assert spec.key == key
        assert spec.name.strip()
        assert spec.lab.startswith("Lab")
        assert spec.description.strip()
        assert spec.notes.strip()
        assert spec.param_grid, f"{key} has no hyperparameter grid"
        assert all(name.startswith("clf__") for name in spec.param_grid)


def test_svm_is_not_advertised_as_probabilistic() -> None:
    spec = get_spec("svm")

    assert spec.supports_probability is False
    assert not hasattr(create_estimator("svm"), "predict_proba")
    assert hasattr(create_estimator("svm"), "decision_function")


@pytest.mark.parametrize("key", [key for key in REQUIRED_ALGORITHMS if key != "svm"])
def test_probabilistic_models_really_support_predict_proba(key: str) -> None:
    assert get_spec(key).supports_probability is True
    assert hasattr(create_estimator(key), "predict_proba")


def test_build_model_pipeline_returns_an_untrained_tfidf_pipeline() -> None:
    pipeline = build_model_pipeline("naive_bayes")

    assert list(pipeline.named_steps) == ["tfidf", "clf"]
    assert not hasattr(pipeline.named_steps["tfidf"], "vocabulary_")


def test_unknown_model_raises_a_helpful_error() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        get_spec("does_not_exist")

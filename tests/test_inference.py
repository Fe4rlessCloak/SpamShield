"""Tests for inference and for loading saved model artifacts.

These tests fit deliberately tiny pipelines on the in-repo fixtures, so they never
download the dataset, never require the trained artifacts and never retrain a real model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from spamshield import predict
from spamshield.constants import MODELS_DIR
from spamshield.evaluate import ModelArtifactError, load_artifact
from spamshield.models import build_model_pipeline
from spamshield.predict import predict_message

FREE_PRIZE_MESSAGE = "Free prize! Text WIN to 80082 now to claim your cash"
LUNCH_MESSAGE = "Are we still meeting for lunch tomorrow at one?"


def save_tiny_model(key: str, messages: list[str], labels: list[int], directory: Path) -> Path:
    """Fit a small model on fixtures and save it exactly like the trainer does."""
    pipeline = build_model_pipeline(key)
    pipeline.fit(messages, labels)
    artifact = {
        "model_key": key,
        "pipeline": pipeline,
        "metadata": {"model_key": key, "supports_probability": key != "svm"},
    }
    path = directory / f"{key}.joblib"
    joblib.dump(artifact, path)
    return path


@pytest.fixture
def tiny_models_dir(tmp_path: Path, sample_messages: list[str]) -> Path:
    labels = [1, 0, 1, 0, 1, 0]
    for key in ("naive_bayes", "svm", "logistic_regression"):
        save_tiny_model(key, sample_messages, labels, tmp_path)
    return tmp_path


class TestPredictMessage:
    def test_returns_a_valid_label(self, tiny_models_dir: Path) -> None:
        result = predict_message(FREE_PRIZE_MESSAGE, "naive_bayes", models_dir=tiny_models_dir)

        assert result.label in {"spam", "ham"}
        assert result.label_id in {0, 1}
        assert result.is_spam == (result.label_id == 1)

    def test_normalises_the_message_it_reports(self, tiny_models_dir: Path) -> None:
        result = predict_message("  FREE   Prize  ", "naive_bayes", models_dir=tiny_models_dir)

        assert result.normalized_text == "free prize"

    def test_exposes_a_probability_when_the_model_supports_it(self, tiny_models_dir: Path) -> None:
        result = predict_message(FREE_PRIZE_MESSAGE, "naive_bayes", models_dir=tiny_models_dir)

        assert result.has_probability
        assert result.probability is not None
        assert 0.0 <= result.probability <= 1.0
        assert result.decision_score is None

    def test_svm_reports_a_decision_score_instead_of_a_probability(
        self, tiny_models_dir: Path
    ) -> None:
        result = predict_message(FREE_PRIZE_MESSAGE, "svm", models_dir=tiny_models_dir)

        assert result.probability is None
        assert result.has_probability is False
        assert result.decision_score is not None

    def test_reports_which_model_was_used(self, tiny_models_dir: Path) -> None:
        result = predict_message(LUNCH_MESSAGE, "logistic_regression", models_dir=tiny_models_dir)

        assert result.model_key == "logistic_regression"
        assert result.model_name == "Logistic Regression"

    @pytest.mark.parametrize("bad_input", ["", "   ", "\n\t ", None, 42])
    def test_rejects_empty_or_non_text_input(
        self, bad_input: object, tiny_models_dir: Path
    ) -> None:
        with pytest.raises(ValueError, match="Please enter a message"):
            predict_message(bad_input, "naive_bayes", models_dir=tiny_models_dir)  # type: ignore[arg-type]

    def test_missing_artifact_raises_with_training_instructions(self, tmp_path: Path) -> None:
        with pytest.raises(ModelArtifactError, match="spamshield.train"):
            predict_message(FREE_PRIZE_MESSAGE, "naive_bayes", models_dir=tmp_path)


class TestModelDiscovery:
    def test_lists_only_available_models(self, tiny_models_dir: Path) -> None:
        available = predict.available_models(models_dir=tiny_models_dir)

        assert set(available) == {"naive_bayes", "svm", "logistic_regression"}
        assert predict.is_model_available("knn", models_dir=tiny_models_dir) is False

    def test_no_models_in_an_empty_directory(self, tmp_path: Path) -> None:
        assert predict.available_models(models_dir=tmp_path) == []


class TestLoadArtifact:
    def test_loads_a_fitted_pipeline_that_can_predict(self, tiny_models_dir: Path) -> None:
        pipeline, metadata = predict.load_pipeline("naive_bayes", models_dir=tiny_models_dir)

        assert metadata["supports_probability"] is True
        assert pipeline.predict([FREE_PRIZE_MESSAGE]).shape == (1,)

    def test_rejects_a_joblib_file_that_is_not_an_artifact(self, tmp_path: Path) -> None:
        joblib.dump({"not": "a spamshield artifact"}, tmp_path / "naive_bayes.joblib")

        with pytest.raises(ModelArtifactError, match="not a SpamShield model artifact"):
            load_artifact("naive_bayes", models_dir=tmp_path)

    def test_reports_a_missing_artifact(self, tmp_path: Path) -> None:
        with pytest.raises(ModelArtifactError, match="No trained model found"):
            load_artifact("svm", models_dir=tmp_path)


@pytest.mark.skipif(
    not (MODELS_DIR / "naive_bayes.joblib").exists(),
    reason="trained artifacts are not present (run `uv run python -m spamshield.train`)",
)
def test_real_trained_artifacts_are_loadable_and_predict_valid_labels() -> None:
    """End-to-end check against the real artifacts, when they exist."""
    available = predict.available_models()

    assert available, "expected at least one trained model"
    for key in available:
        result = predict.predict_message(FREE_PRIZE_MESSAGE, key)
        assert result.label in {"spam", "ham"}
        assert result.has_probability or result.decision_score is not None

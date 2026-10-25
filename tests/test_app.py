"""Tests that the Streamlit pages run without raising.

The pages are executed with Streamlit's own ``AppTest`` harness. Each test controls the
state it depends on (a temporary models directory, temporary metrics paths), so the suite
behaves the same whether or not the real artifacts have been trained.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from streamlit.testing.v1 import AppTest

from spamshield import predict, ui
from spamshield.models import build_model_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 120

CACHED_FUNCTIONS = (
    ui.get_pipeline,
    ui.get_metrics_frame,
    ui.get_metrics_payload,
    ui.get_dataset,
)


@pytest.fixture(autouse=True)
def fresh_caches() -> Iterator[None]:
    """Clear Streamlit caches around every test so pages never see stale state."""
    for cached in CACHED_FUNCTIONS:
        cached.clear()
    yield
    for cached in CACHED_FUNCTIONS:
        cached.clear()


def run_page(relative_path: str) -> AppTest:
    app = AppTest.from_file(str(PROJECT_ROOT / relative_path), default_timeout=TIMEOUT)
    app.run()
    return app


@pytest.fixture
def detector_with_tiny_model(
    tmp_path: Path,
    sample_messages: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point the app at a directory holding one tiny fitted model."""
    pipeline = build_model_pipeline("naive_bayes")
    pipeline.fit(sample_messages, [1, 0, 1, 0, 1, 0])
    joblib.dump(
        {
            "model_key": "naive_bayes",
            "pipeline": pipeline,
            "metadata": {"model_key": "naive_bayes", "supports_probability": True},
        },
        tmp_path / "naive_bayes.joblib",
    )
    monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)
    return tmp_path


class TestEntryPoint:
    def test_app_renders_the_default_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)

        app = run_page("app.py")

        assert not app.exception


class TestSmsDetectorPage:
    def test_asks_for_training_when_no_models_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(predict, "MODELS_DIR", tmp_path)

        app = run_page("pages/1_SMS_Detector.py")

        assert not app.exception
        assert any("artifacts/models" in error.value for error in app.error)
        assert any("spamshield.train" in block.value for block in app.code)

    def test_classifies_a_message_with_a_saved_model(self, detector_with_tiny_model: Path) -> None:
        app = run_page("pages/1_SMS_Detector.py")
        app.text_area[0].set_value("Free prize! Text WIN to 80082 now to claim")
        app.button[0].click()
        app.run()

        assert not app.exception
        metrics = {metric.label: metric.value for metric in app.metric}
        assert metrics["Prediction"] in {"SPAM", "HAM"}
        assert metrics["Model used"] == "Multinomial Naive Bayes"

    def test_rejects_an_empty_message(self, detector_with_tiny_model: Path) -> None:
        app = run_page("pages/1_SMS_Detector.py")
        app.text_area[0].set_value("   ")
        app.button[0].click()
        app.run()

        assert not app.exception
        assert any("Please enter a message" in warning.value for warning in app.warning)


class TestModelComparisonPage:
    def test_renders_the_saved_metrics(self) -> None:
        app = run_page("pages/2_Model_Comparison.py")

        assert not app.exception
        assert not app.error

    def test_asks_for_evaluation_when_metrics_are_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ui, "METRICS_CSV_PATH", tmp_path / "metrics.csv")
        monkeypatch.setattr(ui, "METRICS_JSON_PATH", tmp_path / "metrics.json")

        app = run_page("pages/2_Model_Comparison.py")

        assert not app.exception
        assert any("artifacts/metrics" in error.value for error in app.error)


class TestAboutPage:
    def test_renders_without_artifacts(self) -> None:
        app = run_page("pages/4_About.py")

        assert not app.exception

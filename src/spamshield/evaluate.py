"""Evaluate the six trained models on the held-out test set and report the results.

The test set created here is the *same* deterministic split used during training (same
cleaned dataset, same seed), and it was never used for hyperparameter selection. The
model used for final selection is chosen from the training-only cross-validated spam F1;
the test-set figures are reported for all six models for transparency.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .constants import (
    FIGURES_DIR,
    HAM_LABEL,
    LABEL_NAMES,
    METRICS_CSV_PATH,
    METRICS_JSON_PATH,
    MODELS_DIR,
    SPAM_LABEL,
)
from .models import MODEL_KEYS, MODEL_NAMES, get_spec
from .train import TRAINING_REPORT_PATH, DataBundle, prepare_data
from .utils import bar_chart, confusion_matrix_figure, ensure_directory


class ModelArtifactError(RuntimeError):
    """Raised when a trained model artifact is missing or unreadable."""


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_artifact(key: str, models_dir: str | Path = MODELS_DIR) -> dict[str, Any]:
    """Load a saved model artifact, raising a helpful error when it is missing."""
    path = Path(models_dir) / f"{key}.joblib"
    if not path.exists():
        raise ModelArtifactError(
            f"No trained model found at {path}. Train the models first with "
            "`uv run python -m spamshield.train`."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ModelArtifactError(f"{path} is not a SpamShield model artifact.")
    return artifact


def load_training_report(path: str | Path = TRAINING_REPORT_PATH) -> dict[str, Any]:
    """Return the training report (CV scores and training cost), if present."""
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Test-set metrics and measured cost for one model."""

    key: str
    name: str
    accuracy: float
    precision_spam: float
    recall_spam: float
    f1_spam: float
    cv_f1_spam: float
    train_seconds: float
    inference_seconds: float
    inference_ms_per_message: float
    peak_memory_mb: float
    n_test: int
    supports_probability: bool
    confusion: list[list[int]]
    report: dict[str, Any]

    @property
    def true_negatives(self) -> int:
        return int(self.confusion[0][0])

    @property
    def false_positives(self) -> int:
        return int(self.confusion[0][1])

    @property
    def false_negatives(self) -> int:
        return int(self.confusion[1][0])

    @property
    def true_positives(self) -> int:
        return int(self.confusion[1][1])

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_model(
    key: str,
    X_test: pd.Series,
    y_test: pd.Series,
    *,
    training_report: dict[str, Any] | None = None,
    models_dir: str | Path = MODELS_DIR,
) -> EvaluationResult:
    """Load one saved model and measure its performance on the test set."""
    artifact = load_artifact(key, models_dir)
    pipeline = artifact["pipeline"]
    metadata = artifact.get("metadata", {})
    y_true = np.asarray(y_test)

    started = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    inference_seconds = time.perf_counter() - started

    matrix = confusion_matrix(y_true, y_pred, labels=[HAM_LABEL, SPAM_LABEL])
    report = classification_report(
        y_true,
        y_pred,
        labels=[HAM_LABEL, SPAM_LABEL],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    training = (training_report or {}).get("models", {}).get(key, {})
    n_test = int(len(y_true))

    return EvaluationResult(
        key=key,
        name=MODEL_NAMES[key],
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_spam=float(
            precision_score(y_true, y_pred, pos_label=SPAM_LABEL, zero_division=0)
        ),
        recall_spam=float(recall_score(y_true, y_pred, pos_label=SPAM_LABEL, zero_division=0)),
        f1_spam=float(f1_score(y_true, y_pred, pos_label=SPAM_LABEL, zero_division=0)),
        cv_f1_spam=float(training.get("cv_f1_spam", math.nan)),
        train_seconds=float(training.get("train_seconds", math.nan)),
        inference_seconds=inference_seconds,
        inference_ms_per_message=(inference_seconds / n_test * 1000) if n_test else 0.0,
        peak_memory_mb=float(training.get("peak_memory_mb", math.nan)),
        n_test=n_test,
        supports_probability=bool(
            metadata.get("supports_probability", get_spec(key).supports_probability)
        ),
        confusion=matrix.tolist(),
        report=report,
    )


def evaluate_all(
    bundle: DataBundle,
    *,
    keys: tuple[str, ...] = MODEL_KEYS,
    training_report: dict[str, Any] | None = None,
    models_dir: str | Path = MODELS_DIR,
) -> dict[str, EvaluationResult]:
    """Evaluate every requested model on the shared held-out test set."""
    report = training_report if training_report is not None else load_training_report()
    results: dict[str, EvaluationResult] = {}
    for key in keys:
        result = evaluate_model(
            key,
            bundle.X_test,
            bundle.y_test,
            training_report=report,
            models_dir=models_dir,
        )
        results[key] = result
        print(
            f"{result.name:<28} acc={result.accuracy:.4f}  "
            f"P(spam)={result.precision_spam:.4f}  "
            f"R(spam)={result.recall_spam:.4f}  "
            f"F1(spam)={result.f1_spam:.4f}",
            flush=True,
        )
    return results


def select_best(results: dict[str, EvaluationResult]) -> str:
    """Return the model key chosen by training-only cross-validated spam F1."""
    scored = {
        key: result.cv_f1_spam
        for key, result in results.items()
        if not math.isnan(result.cv_f1_spam)
    }
    if scored:
        return max(scored, key=lambda key: scored[key])
    return max(results, key=lambda key: results[key].f1_spam)


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------
def results_to_frame(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """Build the comparison table (one row per model)."""
    rows = []
    for result in results.values():
        rows.append(
            {
                "model_key": result.key,
                "model": result.name,
                "accuracy": result.accuracy,
                "precision_spam": result.precision_spam,
                "recall_spam": result.recall_spam,
                "f1_spam": result.f1_spam,
                "cv_f1_spam": result.cv_f1_spam,
                "train_seconds": result.train_seconds,
                "inference_seconds": result.inference_seconds,
                "inference_ms_per_message": result.inference_ms_per_message,
                "peak_memory_mb": result.peak_memory_mb,
                "true_negatives": result.true_negatives,
                "false_positives": result.false_positives,
                "false_negatives": result.false_negatives,
                "true_positives": result.true_positives,
            }
        )
    return pd.DataFrame(rows).round(6)


def save_metrics(
    results: dict[str, EvaluationResult],
    *,
    csv_path: str | Path = METRICS_CSV_PATH,
    json_path: str | Path = METRICS_JSON_PATH,
) -> tuple[Path, Path]:
    """Persist the evaluation results as CSV and JSON."""
    csv_target = Path(csv_path)
    json_target = Path(json_path)
    ensure_directory(csv_target.parent)
    ensure_directory(json_target.parent)

    results_to_frame(results).to_csv(csv_target, index=False)

    payload = {
        "selection_metric": ("cross-validated spam-class F1 on the training set only"),
        "selected_model": select_best(results),
        "selection_note": (
            "The selected model comes from training-only cross-validation. Test-set "
            "metrics are reported for all six models for transparency and were not used "
            "to pick the winner."
        ),
        "memory_note": (
            "peak_memory_mb is measured with tracemalloc, which traces Python-level "
            "allocations only. It does not include memory held by native numerical "
            "libraries and is not the total process memory."
        ),
        "test_rows": next(iter(results.values())).n_test if results else 0,
        "models": {key: result.as_dict() for key, result in results.items()},
    }
    json_target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return csv_target, json_target


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def create_figures(
    results: dict[str, EvaluationResult],
    *,
    selected_key: str,
    directory: str | Path = FIGURES_DIR,
) -> list[Path]:
    """Generate every comparison chart and confusion matrix."""
    ensure_directory(directory)
    keys = list(results)
    names = [results[key].name for key in keys]
    highlight = results[selected_key].name
    paths: list[Path] = []

    common = {"highlight": highlight, "directory": directory}
    paths.append(
        bar_chart(
            names,
            [results[key].accuracy for key in keys],
            title="Accuracy by model (held-out test set)",
            ylabel="Accuracy",
            filename="accuracy_comparison.png",
            **common,
        )
    )
    paths.append(
        bar_chart(
            names,
            [results[key].f1_spam for key in keys],
            title="Spam-class F1 score by model (held-out test set)",
            ylabel="F1 score (spam)",
            filename="f1_comparison.png",
            **common,
        )
    )
    paths.append(
        bar_chart(
            names,
            [results[key].precision_spam for key in keys],
            title="Spam-class precision by model (held-out test set)",
            ylabel="Precision (spam)",
            filename="precision_comparison.png",
            **common,
        )
    )
    paths.append(
        bar_chart(
            names,
            [results[key].recall_spam for key in keys],
            title="Spam-class recall by model (held-out test set)",
            ylabel="Recall (spam)",
            filename="recall_comparison.png",
            **common,
        )
    )
    paths.append(
        bar_chart(
            names,
            [results[key].train_seconds for key in keys],
            title="Training time by model (includes cross-validated tuning)",
            ylabel="Seconds",
            filename="training_time_comparison.png",
            value_format="%.2fs",
            **common,
        )
    )
    paths.append(
        bar_chart(
            names,
            [results[key].inference_ms_per_message for key in keys],
            title="Inference time per message (held-out test set)",
            ylabel="Milliseconds per message",
            filename="inference_time_comparison.png",
            value_format="%.3fms",
            **common,
        )
    )

    for key in keys:
        paths.append(
            confusion_matrix_figure(
                np.asarray(results[key].confusion),
                LABEL_NAMES,
                title=f"Confusion matrix — {results[key].name}",
                filename=f"confusion_matrix_{key}.png",
                directory=directory,
            )
        )
    return paths


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main() -> None:
    """Evaluate all six models and write metrics + figures."""
    bundle = prepare_data()
    print(f"Evaluating on {len(bundle.X_test)} held-out test messages\n")

    results = evaluate_all(bundle)
    csv_path, json_path = save_metrics(results)
    selected_key = select_best(results)
    figures = create_figures(results, selected_key=selected_key)

    print()
    print(results_to_frame(results)[["model", "accuracy", "f1_spam"]].to_string(index=False))
    print()
    print(f"Selected model (training-only CV spam F1): {results[selected_key].name}")
    print(f"Metrics written to {csv_path} and {json_path}")
    print(f"{len(figures)} figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

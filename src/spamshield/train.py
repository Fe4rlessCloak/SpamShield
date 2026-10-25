"""Train the six spam classifiers and persist them as reusable joblib pipelines.

Training is deliberately leakage-safe:

* the dataset is cleaned (duplicates removed) before splitting,
* the split is stratified and fixed by a seed,
* hyperparameters are chosen with stratified cross-validation on the training portion
  only, and the test set is never touched here.

Each fitted model is saved as a single joblib artifact containing the complete
``TF-IDF -> classifier`` pipeline together with the metadata needed to identify it.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .constants import (
    CV_FOLDS,
    DATASET_PATH,
    FIGURES_DIR,
    INVERSE_LABEL_MAP,
    LABEL_MAP,
    MAX_FEATURES,
    METRICS_DIR,
    MIN_DF,
    MODELS_DIR,
    NGRAM_RANGE,
    RANDOM_SEED,
    TEST_SIZE,
)
from .data import DatasetReport, clean_dataset, load_dataset, sha256_of_file
from .models import MODEL_KEYS, ModelSpec, get_spec
from .preprocessing import build_pipeline, split_data

TRAINING_REPORT_PATH = METRICS_DIR / "training_report.json"

FEATURE_CONFIG: dict[str, Any] = {
    "vectorizer": "TfidfVectorizer",
    "max_features": MAX_FEATURES,
    "ngram_range": list(NGRAM_RANGE),
    "min_df": MIN_DF,
    "sublinear_tf": True,
}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
@dataclass
class DataBundle:
    """The cleaned dataset and its fixed train/test split."""

    frame: pd.DataFrame
    report: DatasetReport
    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    dataset_checksum: str

    @property
    def n_features_config(self) -> dict[str, Any]:
        return dict(FEATURE_CONFIG)


def _dataset_checksum() -> str:
    try:
        return sha256_of_file(DATASET_PATH) if DATASET_PATH.exists() else ""
    except OSError:
        return ""


def prepare_data(*, download: bool = True) -> DataBundle:
    """Load, validate, de-duplicate and split the dataset.

    The split is deterministic: the same cleaned data and seed always produce the same
    training and test sets, so training and evaluation agree on the split.
    """
    frame = load_dataset(download=download)
    cleaned, report = clean_dataset(frame)
    X_train, X_test, y_train, y_test = split_data(cleaned["message"], cleaned["label_id"])
    return DataBundle(
        frame=cleaned,
        report=report,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dataset_checksum=_dataset_checksum(),
    )


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
@dataclass
class TrainingResult:
    """Outcome of fitting one model, including the cost of doing so."""

    key: str
    name: str
    cv_f1_spam: float
    best_params: dict[str, Any]
    train_seconds: float
    peak_memory_mb: float
    n_features: int
    supports_probability: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "cv_f1_spam": round(self.cv_f1_spam, 4),
            "best_params": self.best_params,
            "train_seconds": round(self.train_seconds, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 3),
            "n_features": self.n_features,
            "supports_probability": self.supports_probability,
        }


def train_model(
    spec: ModelSpec,
    X_train: pd.Series,
    y_train: pd.Series,
    *,
    cv_folds: int = CV_FOLDS,
) -> tuple[Pipeline, TrainingResult]:
    """Fit one model with training-only cross-validated hyperparameter search.

    ``tracemalloc`` traces allocations made by the Python interpreter. joblib is run
    sequentially (a single worker) so that the measured peak actually includes the model
    fitting work rather than being split across untraced subprocesses.
    """
    pipeline = build_pipeline(spec.factory())
    cross_validation = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    tracemalloc.start()
    started = time.perf_counter()
    try:
        if spec.param_grid:
            search = GridSearchCV(
                pipeline,
                spec.param_grid,
                scoring="f1",
                cv=cross_validation,
                refit=True,
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = dict(search.best_params_)
            cv_f1 = float(search.best_score_)
        else:  # pragma: no cover - every bundled model has a grid
            fitted = pipeline.fit(X_train, y_train)
            best_params = {}
            cv_f1 = float("nan")
    finally:
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    vectorizer = fitted.named_steps["tfidf"]
    result = TrainingResult(
        key=spec.key,
        name=spec.name,
        cv_f1_spam=cv_f1,
        best_params=best_params,
        train_seconds=elapsed,
        peak_memory_mb=peak / (1024**2),
        n_features=int(len(vectorizer.get_feature_names_out())),
        supports_probability=spec.supports_probability,
    )
    return fitted, result


def build_model_metadata(
    spec: ModelSpec,
    result: TrainingResult,
    bundle: DataBundle,
) -> dict[str, Any]:
    """Collect everything needed to identify a trained model later."""
    return {
        "model_key": spec.key,
        "model_name": spec.name,
        "lab": spec.lab,
        "description": spec.description,
        "notes": spec.notes,
        "supports_probability": spec.supports_probability,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "name": "SMS Spam Collection (UCI id 228)",
            "rows_used": int(len(bundle.frame)),
            "sha256": bundle.dataset_checksum,
        },
        "feature_config": bundle.n_features_config,
        "label_map": dict(LABEL_MAP),
        "inverse_label_map": {str(k): v for k, v in INVERSE_LABEL_MAP.items()},
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "cv_f1_spam": round(result.cv_f1_spam, 4),
        "best_params": result.best_params,
        "train_seconds": round(result.train_seconds, 4),
        "peak_memory_mb": round(result.peak_memory_mb, 3),
        "n_features": result.n_features,
    }


def save_model(key: str, pipeline: Pipeline, metadata: dict[str, Any]) -> Path:
    """Persist a fitted pipeline and its metadata as one joblib artifact."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{key}.joblib"
    joblib.dump({"model_key": key, "pipeline": pipeline, "metadata": metadata}, path)
    return path


def train_all(
    bundle: DataBundle,
    *,
    keys: tuple[str, ...] = MODEL_KEYS,
) -> dict[str, TrainingResult]:
    """Train and save every requested model, returning their training results."""
    results: dict[str, TrainingResult] = {}
    for key in keys:
        spec = get_spec(key)
        print(f"Training {spec.name} ({spec.lab}) ...", flush=True)
        pipeline, result = train_model(spec, bundle.X_train, bundle.y_train)
        metadata = build_model_metadata(spec, result, bundle)
        path = save_model(key, pipeline, metadata)
        results[key] = result
        print(
            f"  cv F1(spam)={result.cv_f1_spam:.4f}  "
            f"features={result.n_features}  "
            f"time={result.train_seconds:.2f}s  "
            f"peak={result.peak_memory_mb:.2f} MB  -> {path.name}",
            flush=True,
        )
    return results


def write_training_report(results: dict[str, TrainingResult], bundle: DataBundle) -> Path:
    """Write the machine-readable training report (CV scores, params, cost)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": {
            "name": "SMS Spam Collection (UCI id 228)",
            "rows_raw": bundle.report.raw_rows,
            "duplicate_rows_removed": bundle.report.duplicate_rows_removed,
            "rows_used": int(len(bundle.frame)),
            "train_rows": int(len(bundle.X_train)),
            "test_rows": int(len(bundle.X_test)),
            "sha256": bundle.dataset_checksum,
        },
        "feature_config": bundle.n_features_config,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "selection_metric": "f1 (spam class), cross-validated on the training set only",
        "models": {key: result.as_dict() for key, result in results.items()},
    }
    TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return TRAINING_REPORT_PATH


def main() -> None:
    """Train all six models and run the full evaluation pipeline."""
    from .evaluate import create_figures, evaluate_all, save_metrics, select_best

    bundle = prepare_data()
    print(
        f"Dataset: {len(bundle.frame)} unique messages "
        f"({bundle.report.raw_rows} raw rows, "
        f"{bundle.report.duplicate_rows_removed} duplicates removed)"
    )
    print(f"Split: {len(bundle.X_train)} train / {len(bundle.X_test)} test (stratified)")
    print()

    results = train_all(bundle)
    report_path = write_training_report(results, bundle)

    best_key = max(results, key=lambda key: results[key].cv_f1_spam)
    print()
    print(f"Best model by cross-validated spam F1: {results[best_key].name}")
    print(f"Training report written to {report_path}")

    print()
    print(f"Evaluating on {len(bundle.X_test)} held-out test messages")
    evaluations = evaluate_all(bundle)
    csv_path, json_path = save_metrics(evaluations)
    selected_key = select_best(evaluations)
    figures = create_figures(evaluations, selected_key=selected_key)

    print()
    print(f"Selected model (training-only CV spam F1): {evaluations[selected_key].name}")
    print(f"Metrics written to {csv_path} and {json_path}")
    print(f"{len(figures)} figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

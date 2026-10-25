"""Load saved models and classify new SMS messages.

This module is deliberately free of any Streamlit dependency so it can also be used from
scripts and tests. It is the single place that turns raw SMS text into a labelled
prediction, and it is careful not to invent probabilities for models that do not have
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import INVERSE_LABEL_MAP, MODELS_DIR, SPAM_LABEL
from .evaluate import load_artifact
from .models import MODEL_KEYS, get_spec
from .preprocessing import normalize_text

TRAIN_COMMAND = "uv run python -m spamshield.train"


@dataclass
class Prediction:
    """A single classification result for one message."""

    text: str
    normalized_text: str
    model_key: str
    model_name: str
    label: str
    label_id: int
    probability: float | None
    decision_score: float | None
    explanation: str
    notes: str

    @property
    def is_spam(self) -> bool:
        return self.label_id == SPAM_LABEL

    @property
    def has_probability(self) -> bool:
        return self.probability is not None


def model_path(key: str, models_dir: str | Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"{key}.joblib"


def is_model_available(key: str, models_dir: str | Path = MODELS_DIR) -> bool:
    """Return ``True`` when a trained artifact exists for ``key``."""
    return model_path(key, models_dir).exists()


def available_models(models_dir: str | Path = MODELS_DIR) -> list[str]:
    """Return the keys of all models that have been trained and saved."""
    return [key for key in MODEL_KEYS if is_model_available(key, models_dir)]


def load_pipeline(key: str, models_dir: str | Path = MODELS_DIR) -> tuple[Any, dict[str, Any]]:
    """Load a fitted pipeline and its metadata for ``key``.

    Raises :class:`~spamshield.evaluate.ModelArtifactError` when the artifact is missing
    or unreadable.
    """
    artifact = load_artifact(key, models_dir)
    return artifact["pipeline"], artifact.get("metadata", {})


def _probability_of_spam(pipeline: Any, message: pd.Series) -> float | None:
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "predict_proba"):
        return None
    probabilities = pipeline.predict_proba(message)[0]
    classes = list(classifier.classes_)
    if SPAM_LABEL not in classes:
        return None
    return float(probabilities[classes.index(SPAM_LABEL)])


def _decision_score(pipeline: Any, message: pd.Series) -> float | None:
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "decision_function"):
        return None
    scores = np.asarray(pipeline.decision_function(message)).ravel()
    return float(scores[0]) if scores.size else None


def predict_message(
    text: str,
    key: str,
    *,
    models_dir: str | Path = MODELS_DIR,
) -> Prediction:
    """Classify one SMS message with the model identified by ``key``.

    The raw message is passed straight to the saved pipeline, which applies exactly the
    same normalisation and TF-IDF transformation that was used during training.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a message to analyse.")

    pipeline, _metadata = load_pipeline(key, models_dir)
    spec = get_spec(key)
    message = pd.Series([text])

    label_id = int(pipeline.predict(message)[0])
    probability = _probability_of_spam(pipeline, message)
    decision_score = None if probability is not None else _decision_score(pipeline, message)

    return Prediction(
        text=text,
        normalized_text=normalize_text(text),
        model_key=key,
        model_name=spec.name,
        label=INVERSE_LABEL_MAP.get(label_id, str(label_id)),
        label_id=label_id,
        probability=probability,
        decision_score=decision_score,
        explanation=spec.description,
        notes=spec.notes,
    )

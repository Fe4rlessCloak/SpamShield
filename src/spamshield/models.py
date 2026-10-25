"""Definition and construction of the six machine learning models.

Every model is described by a :class:`ModelSpec`, which bundles the estimator factory, a
small hyperparameter grid for leakage-safe search and the human-readable description that
the Streamlit frontend shows. Keeping them in one registry means the training script, the
evaluation pipeline and the UI can never disagree about which models exist.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .constants import RANDOM_SEED
from .preprocessing import build_pipeline


@dataclass(frozen=True)
class ModelSpec:
    """Everything the project needs to know about one algorithm."""

    key: str
    name: str
    lab: str
    description: str
    notes: str
    supports_probability: bool
    factory: Callable[[], BaseEstimator]
    param_grid: dict[str, list[Any]] = field(default_factory=dict)


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        name="Logistic Regression",
        lab="Lab 03 — Regression",
        description=(
            "A linear model that estimates the probability of spam from the TF-IDF "
            "features. Fast, interpretable and a strong baseline on sparse text."
        ),
        notes="Provides calibrated class probabilities via predict_proba.",
        supports_probability=True,
        factory=lambda: LogisticRegression(
            C=1.0, solver="liblinear", max_iter=1000, random_state=RANDOM_SEED
        ),
        param_grid={"clf__C": [0.5, 1.0, 5.0]},
    ),
    "decision_tree": ModelSpec(
        key="decision_tree",
        name="Decision Tree",
        lab="Lab 04 — Decision Tree",
        description=(
            "Learns a hierarchy of if/else rules over individual words and n-grams. "
            "Easy to explain, but prone to overfitting without depth limits."
        ),
        notes="Leaf class frequencies are used as probability estimates.",
        supports_probability=True,
        factory=lambda: DecisionTreeClassifier(random_state=RANDOM_SEED),
        param_grid={"clf__max_depth": [None, 20, 50]},
    ),
    "naive_bayes": ModelSpec(
        key="naive_bayes",
        name="Multinomial Naive Bayes",
        lab="Lab 05 — Naive Bayes",
        description=(
            "A probabilistic model that assumes word occurrences are independent given "
            "the class. The classic, very fast baseline for text classification."
        ),
        notes="Provides posterior class probabilities via predict_proba.",
        supports_probability=True,
        factory=lambda: MultinomialNB(alpha=0.1),
        param_grid={"clf__alpha": [0.1, 0.5, 1.0]},
    ),
    "knn": ModelSpec(
        key="knn",
        name="K-Nearest Neighbors",
        lab="Lab 06 — K-Nearest Neighbors",
        description=(
            "Classifies a message by the majority label among the most similar training "
            "messages, measured with cosine distance over TF-IDF vectors."
        ),
        notes="Distance-weighted neighbourhood votes provide probability estimates.",
        supports_probability=True,
        factory=lambda: KNeighborsClassifier(
            n_neighbors=5, metric="cosine", algorithm="brute", n_jobs=-1
        ),
        param_grid={"clf__n_neighbors": [3, 5, 11]},
    ),
    "neural_network": ModelSpec(
        key="neural_network",
        name="Neural Network (MLP)",
        lab="Lab 07 — Artificial Neural Network",
        description=(
            "A small feed-forward network with a single hidden layer that learns "
            "non-linear combinations of the TF-IDF features."
        ),
        notes="The softmax output of the network provides probability estimates.",
        supports_probability=True,
        factory=lambda: MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_SEED,
        ),
        param_grid={"clf__hidden_layer_sizes": [(50,), (100,)]},
    ),
    "svm": ModelSpec(
        key="svm",
        name="Support Vector Machine",
        lab="Lab 08 — Support Vector Machine",
        description=(
            "Finds the maximum-margin hyperplane separating spam from ham. Very "
            "effective on high-dimensional sparse data such as TF-IDF vectors."
        ),
        notes=(
            "LinearSVC has no predict_proba; the signed distance to the hyperplane is "
            "reported as a decision score instead of a probability."
        ),
        supports_probability=False,
        factory=lambda: LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_SEED),
        param_grid={"clf__C": [0.1, 1.0, 5.0]},
    ),
}

MODEL_KEYS: tuple[str, ...] = tuple(MODEL_SPECS)

MODEL_NAMES: dict[str, str] = {key: spec.name for key, spec in MODEL_SPECS.items()}


def get_spec(key: str) -> ModelSpec:
    """Return the spec for ``key`` or raise a helpful error."""
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        known = ", ".join(MODEL_KEYS)
        raise KeyError(f"Unknown model {key!r}. Available models: {known}") from exc


def create_estimator(key: str) -> BaseEstimator:
    """Create a fresh, unfitted estimator for ``key``."""
    return get_spec(key).factory()


def build_model_pipeline(key: str) -> Pipeline:
    """Create a fresh ``TF-IDF -> estimator`` pipeline for ``key``."""
    return build_pipeline(create_estimator(key))

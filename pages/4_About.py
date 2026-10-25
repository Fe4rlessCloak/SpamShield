"""Page 4 — About the Project."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from spamshield import ui
from spamshield.constants import (
    DATASET_CITATION,
    DATASET_LICENSE,
    DATASET_NAME,
    DATASET_SOURCE,
    RANDOM_SEED,
    TEST_SIZE,
)
from spamshield.models import MODEL_SPECS

st.title("About the Project")
st.write(
    "SpamShield is a university Machine Learning midterm project: it classifies SMS "
    "messages as **spam** or **ham** with six different algorithms, compares how well "
    "each one performs, and exposes the result through this dashboard."
)

st.header("How SMS classification works")
st.markdown(
    """
1. **Input** — the raw SMS text is taken exactly as written, including numbers,
   currency symbols and abbreviations.
2. **Normalisation** — text is lower-cased, Unicode-normalised and whitespace-collapsed.
   Nothing that carries spam signal is thrown away.
3. **Feature extraction** — the message is converted into TF-IDF weights over word
   unigrams and bigrams (up to 4,000 features), fitted on the training data only.
4. **Classification** — one of the six trained models scores the message.
5. **Output** — a label (`spam` / `ham`) plus a probability where the model genuinely
   supports one, or a decision score otherwise.
"""
)

st.header("The six algorithms")
rows = [
    {
        "Model": spec.name,
        "Lab": spec.lab,
        "Family": spec.description.split(".")[0] + ".",
        "Probabilities": "yes" if spec.supports_probability else "no (decision score)",
    }
    for spec in MODEL_SPECS.values()
]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.header("Lab 1–8 coverage")
lab_rows = [
    {
        "Lab": "Lab 01",
        "Topic": "Python basics",
        "Where it is used": (
            "Functions, loops, conditionals and input validation throughout "
            "`src/spamshield/`; command-line entry points in `data.py`, `train.py` and "
            "`evaluate.py`"
        ),
    },
    {
        "Lab": "Lab 02",
        "Topic": "Data structures and memory",
        "Where it is used": (
            "List/dict model registry in `models.py`, dataclasses (`ModelSpec`, "
            "`DatasetReport`, `DataBundle`, `Prediction`); memory measured with "
            "`tracemalloc` in `train.train_model`"
        ),
    },
    {
        "Lab": "Lab 03",
        "Topic": "Regression",
        "Where it is used": "`LogisticRegression` classifies spam as a linear probability model",
    },
    {
        "Lab": "Lab 04",
        "Topic": "Decision tree",
        "Where it is used": "`DecisionTreeClassifier` learns if/else rules over words",
    },
    {
        "Lab": "Lab 05",
        "Topic": "Naive Bayes",
        "Where it is used": "`MultinomialNB` with Laplace smoothing over TF-IDF counts",
    },
    {
        "Lab": "Lab 06",
        "Topic": "K-nearest neighbors",
        "Where it is used": "`KNeighborsClassifier` with cosine distance on sparse vectors",
    },
    {
        "Lab": "Lab 07",
        "Topic": "Artificial neural network",
        "Where it is used": "`MLPClassifier` with a single small hidden layer",
    },
    {
        "Lab": "Lab 08",
        "Topic": "Support vector machine",
        "Where it is used": "`LinearSVC` finds the maximum-margin hyperplane",
    },
]
st.dataframe(pd.DataFrame(lab_rows), hide_index=True, width="stretch")
st.caption(
    "Lab 03 is about regression, but ordinary Linear Regression is not appropriate for "
    "spam/ham labels, so Logistic Regression — a classification model from the same "
    "family — is used instead."
)

st.header("The processing pipeline")
st.code(
    """data/raw/SMSSpamCollection (download + SHA-256 check)
      -> clean_dataset()      drop empty messages and exact duplicates
      -> split_data()         stratified 80/20 split, fixed seed 42
      -> Pipeline(
             TfidfVectorizer  4,000 features, 1-2 grams, fitted on train only
             classifier       one of the six algorithms
         )
      -> joblib artifact in artifacts/models/
      -> evaluate.py          metrics.csv / metrics.json + reports/figures/""",
    language=None,
)
st.markdown(
    f"""
- Fixed random seed **{RANDOM_SEED}** and a fixed **{TEST_SIZE:.0%}** test split make runs
  reproducible.
- Hyperparameters are tuned with stratified cross-validation on the **training data
  only**, which is why the vectorizer and the classifier live in the same pipeline.
- The model is selected by cross-validated spam F1 computed on the training set; the
  test set is only used for the final report.
"""
)

st.header("Limitations")
st.markdown(
    f"""
- **The dataset is small and old.** The {DATASET_NAME} was collected around 2011 and
  mainly reflects the language and messaging style of its original sources. Spam
  wording has changed a lot since then.
- **No claim of generalisation.** This classifier was trained to demonstrate algorithms,
  not to filter real traffic: it should not be assumed to work reliably on modern SMS,
  on other languages, or against current phishing and smishing techniques.
- **Bag-of-words only.** TF-IDF ignores word order and meaning, and the models are
  classic, shallow learners rather than modern language models. It is also
  keyword-sensitive: unusual but legitimate wording can be misclassified.
- **Not perfectly balanced.** Spam is a minority class (~13%), so recall, not accuracy,
  is the honest metric; the best models still miss roughly 8–12% of spam.
- **Memory figures are approximate.** `tracemalloc` reports Python-level allocations
  only, not memory held by native numerical libraries and not total process memory.
- **`LinearSVC` has no probabilities**, so its score is a decision value, not a
  confidence.
- **Scope.** No real-time SMS interception, no user accounts, no database and no
  deployment — none of that is part of the brief.
"""
)

st.header("Dataset and citation")
st.markdown(
    f"""
{DATASET_NAME}, {DATASET_SOURCE}, licensed **{DATASET_LICENSE}**.

> {DATASET_CITATION}

The dataset is downloaded on first use, cached in `data/raw/` and verified with a
SHA-256 checksum recorded in `data/raw/dataset_metadata.json`.
"""
)

st.header("Commands")
st.code(
    """uv sync --locked                          # install locked dependencies
uv run python -m spamshield.train         # download, train, evaluate, plot
uv run streamlit run app.py               # launch this dashboard
uv run pytest                             # run the test suite""",
    language="bash",
)

if not ui.models_directory().exists():  # pragma: no cover - defensive
    st.warning("Model artifacts were not found. Run `uv run python -m spamshield.train`.")

# SpamShield — SMS Spam Detection & ML Algorithm Comparison

A small, reproducible university Machine Learning midterm project that classifies SMS
messages as **spam** or **ham (legitimate)** using **six** different algorithms, compares
their performance, and serves the result through an interactive Streamlit dashboard.

> This README is filled in progressively as the project is implemented. The final version
> documents setup, training, evaluation, the Lab 1–8 coverage table and known limitations.

## Objective

Train, evaluate and compare six classic machine learning algorithms on the same SMS
dataset and provide a working interface for classifying new messages.

## Machine learning algorithms

| # | Algorithm              | scikit-learn estimator      | Lab    |
| - | ---------------------- | --------------------------- | ------ |
| 1 | Logistic Regression    | `LogisticRegression`        | Lab 03 |
| 2 | Decision Tree          | `DecisionTreeClassifier`    | Lab 04 |
| 3 | Multinomial Naive Bayes| `MultinomialNB`             | Lab 05 |
| 4 | K-Nearest Neighbors    | `KNeighborsClassifier`      | Lab 06 |
| 5 | Neural Network (MLP)   | `MLPClassifier`             | Lab 07 |
| 6 | Support Vector Machine | `LinearSVC`                 | Lab 08 |

## Technology stack

- **Python 3.12** (pinned via `.python-version`)
- **uv** for Python version selection, virtual environments and locked dependencies
- **pandas / NumPy** for data handling
- **scikit-learn** for ML and TF-IDF feature extraction
- **Matplotlib / Seaborn** for visualisations
- **Streamlit** for the frontend
- **joblib** for model serialization
- **pytest** for automated tests, **Ruff** for linting/formatting

## Getting started

```bash
# 1. Install the pinned Python and the locked dependencies
uv sync --locked
```

Training, evaluation and application commands are documented below as they are added.

## Project layout

```
src/spamshield/    core library (data, preprocessing, models, training, evaluation, inference)
app.py, pages/     Streamlit frontend
data/raw/          cached dataset (downloaded + checksum-verified, not committed)
artifacts/         trained models (regenerated) and machine-readable metrics
reports/figures/   generated comparison charts
tests/             pytest suite
```

## Dataset

SMS Spam Collection, UCI Machine Learning Repository (dataset id 228), licensed CC BY 4.0.
The dataset is downloaded on first use and verified with a SHA-256 checksum. Full citation
and attribution are added to this README once the data pipeline is implemented.

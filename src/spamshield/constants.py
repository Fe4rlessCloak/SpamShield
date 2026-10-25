"""Project-wide constants: paths, random seeds, dataset metadata and feature settings.

Keeping these in one module means every script, test and Streamlit page agrees on the
exact same seeds, paths and label mapping — a requirement for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_PATH = RAW_DATA_DIR / "SMSSpamCollection"
METADATA_PATH = RAW_DATA_DIR / "dataset_metadata.json"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
METRICS_CSV_PATH = METRICS_DIR / "metrics.csv"
METRICS_JSON_PATH = METRICS_DIR / "metrics.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
LABEL_MAP = {"ham": 0, "spam": 1}
INVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
SPAM_LABEL = LABEL_MAP["spam"]
HAM_LABEL = LABEL_MAP["ham"]
LABEL_NAMES = [INVERSE_LABEL_MAP[HAM_LABEL], INVERSE_LABEL_MAP[SPAM_LABEL]]

# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------
MAX_FEATURES = 4000
NGRAM_RANGE = (1, 2)
MIN_DF = 2

# --------------------------------------------------------------------------------------
# Dataset metadata (used for the metadata file and the README attribution)
# --------------------------------------------------------------------------------------
DATASET_NAME = "SMS Spam Collection"
DATASET_SOURCE = "UCI Machine Learning Repository"
DATASET_DATASET_URL = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DATASET_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_ARCHIVE_MEMBER = "SMSSpamCollection"
DATASET_LICENSE = "CC BY 4.0"
DATASET_CITATION = (
    "Almeida, T. A., Gomez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the "
    "Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM "
    "Symposium on Document Engineering (DocEng '11), Mountain View, CA, USA."
)
DATASET_EXPECTED_ROWS = 5574

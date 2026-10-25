"""Dataset acquisition, validation and descriptive statistics.

The *SMS Spam Collection* is downloaded from the UCI Machine Learning Repository the first
time it is needed, cached under ``data/raw`` and verified with a SHA-256 checksum. Every
later run is fully offline.

The raw file is a tab-separated, header-less collection of ``label<TAB>message`` records.
Some messages begin or end with a double quote, so quote processing must be disabled while
parsing; otherwise the message text is silently truncated and rows are lost.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .constants import (
    DATASET_ARCHIVE_MEMBER,
    DATASET_CITATION,
    DATASET_DATASET_URL,
    DATASET_DOWNLOAD_URL,
    DATASET_EXPECTED_ROWS,
    DATASET_LICENSE,
    DATASET_NAME,
    DATASET_PATH,
    DATASET_SOURCE,
    LABEL_MAP,
    METADATA_PATH,
    RAW_DATA_DIR,
)

COLUMNS = ["label", "message"]


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be downloaded, read or validated."""


# --------------------------------------------------------------------------------------
# Checksums and metadata
# --------------------------------------------------------------------------------------
def sha256_of_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of an in-memory payload."""
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path: str | Path | None = None) -> dict:
    """Return the recorded dataset metadata, or an empty dict when absent."""
    metadata_path = Path(path or METADATA_PATH)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def _write_metadata(data_checksum: str, archive_checksum: str) -> None:
    """Write a deterministic manifest describing the downloaded dataset."""
    metadata = {
        "dataset": DATASET_NAME,
        "source": DATASET_SOURCE,
        "dataset_url": DATASET_DATASET_URL,
        "download_url": DATASET_DOWNLOAD_URL,
        "license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "archive_member": DATASET_ARCHIVE_MEMBER,
        "archive_sha256": archive_checksum,
        "data_sha256": data_checksum,
        "rows": _count_rows(DATASET_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Download / load
# --------------------------------------------------------------------------------------
def _download_archive() -> bytes:
    try:
        with urllib.request.urlopen(DATASET_DOWNLOAD_URL, timeout=60) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DatasetError(
            f"Could not download the dataset from {DATASET_DOWNLOAD_URL}: {exc}"
        ) from exc


def download_dataset(*, force: bool = False) -> Path:
    """Download the archive, extract the collection and cache it locally.

    Subsequent calls are no-ops unless ``force`` is set.
    """
    if DATASET_PATH.exists() and not force:
        return DATASET_PATH

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_bytes = _download_archive()

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = archive.namelist()
            if DATASET_ARCHIVE_MEMBER not in members:
                raise DatasetError(
                    f"{DATASET_ARCHIVE_MEMBER!r} not found in the archive; members are {members}"
                )
            payload = archive.read(DATASET_ARCHIVE_MEMBER)
    except zipfile.BadZipFile as exc:
        raise DatasetError("The downloaded archive is not a valid zip file.") from exc

    DATASET_PATH.write_bytes(payload)
    _write_metadata(sha256_of_bytes(payload), sha256_of_bytes(archive_bytes))
    return DATASET_PATH


def verify_checksum(path: str | Path | None = None) -> bool:
    """Verify the cached dataset against the checksum in the metadata manifest."""
    dataset_path = Path(path or DATASET_PATH)
    if not dataset_path.exists():
        raise DatasetError(f"Dataset not found at {dataset_path}.")
    metadata = read_metadata()
    expected = metadata.get("data_sha256")
    if not expected:
        raise DatasetError(
            "No dataset metadata manifest found; cannot verify the checksum. "
            "Run `uv run python -m spamshield.data` to download and record it."
        )
    actual = sha256_of_file(dataset_path)
    if actual != expected:
        raise DatasetError(
            f"Checksum mismatch for {dataset_path}: expected {expected}, got {actual}."
        )
    return True


def load_dataset(
    path: str | Path | None = None,
    *,
    download: bool = True,
) -> pd.DataFrame:
    """Load the SMS Spam Collection into a ``[label, message]`` DataFrame.

    Downloads and caches the dataset first when it is missing and ``download`` is true.
    """
    dataset_path = Path(path or DATASET_PATH)
    if not dataset_path.exists():
        if not download:
            raise DatasetError(
                f"Dataset not found at {dataset_path}. Run "
                "`uv run python -m spamshield.data` to download it."
            )
        dataset_path = download_dataset()

    frame = pd.read_csv(
        dataset_path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    frame[COLUMNS] = frame[COLUMNS].fillna("")
    return frame


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
@dataclass
class DatasetReport:
    """Summary of the validation and cleaning applied to the raw dataset."""

    raw_rows: int = 0
    unknown_labels: int = 0
    empty_messages: int = 0
    duplicate_messages: int = 0
    duplicate_rows_removed: int = 0
    conflicting_label_messages: int = 0
    final_rows: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, DatasetReport]:
    """Validate, de-duplicate and label-encode the raw dataset.

    Cleaning happens *before* the train/test split so that identical messages can never
    leak across both splits. Steps:

    1. Normalise the label and drop records with an unknown label.
    2. Drop records whose message is missing or contains only whitespace.
    3. Detect identical messages carrying conflicting labels (reported, first kept).
    4. Drop exact duplicate messages, keeping the first occurrence.
    5. Add an integer ``label_id`` column using :data:`~spamshield.constants.LABEL_MAP`.
    """
    if not set(COLUMNS).issubset(frame.columns):
        raise DatasetError(f"Expected columns {COLUMNS}, got {list(frame.columns)}.")

    report = DatasetReport(raw_rows=len(frame))
    data = frame.loc[:, COLUMNS].copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str)

    known = data["label"].isin(LABEL_MAP)
    report.unknown_labels = int((~known).sum())
    data = data[known]

    empty = data["message"].str.strip().eq("")
    report.empty_messages = int(empty.sum())
    data = data[~empty].copy()

    label_counts_per_message = data.groupby("message")["label"].nunique()
    report.conflicting_label_messages = int((label_counts_per_message > 1).sum())

    occurrences = data["message"].value_counts()
    report.duplicate_messages = int((occurrences > 1).sum())

    before = len(data)
    data = data.drop_duplicates(subset="message", keep="first")
    report.duplicate_rows_removed = before - len(data)

    data["label_id"] = data["label"].map(LABEL_MAP).astype(int)
    data = data.reset_index(drop=True)

    report.final_rows = len(data)
    report.label_counts = {
        str(label): int(count) for label, count in data["label"].value_counts().items()
    }
    return data, report


def dataset_stats(frame: pd.DataFrame) -> dict:
    """Return headline statistics about a (cleaned) dataset."""
    labels = frame["label"]
    counts = labels.value_counts()
    lengths = frame["message"].str.len()
    words = frame["message"].str.split().str.len()
    total = len(frame)
    spam = int(counts.get("spam", 0))
    return {
        "total_messages": total,
        "ham_messages": int(counts.get("ham", 0)),
        "spam_messages": spam,
        "spam_ratio": round(spam / total, 4) if total else 0.0,
        "unique_messages": int(frame["message"].nunique()),
        "mean_characters": round(float(lengths.mean()), 2) if total else 0.0,
        "median_characters": float(lengths.median()) if total else 0.0,
        "max_characters": int(lengths.max()) if total else 0,
        "mean_words": round(float(words.mean()), 2) if total else 0.0,
    }


def text_length_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-label summary of message length in characters and words."""
    lengths = pd.DataFrame(
        {
            "label": frame["label"],
            "characters": frame["message"].str.len(),
            "words": frame["message"].str.split().str.len(),
        }
    )
    return lengths.groupby("label")[["characters", "words"]].agg(["mean", "median", "max"]).round(2)


def main() -> None:
    """Download (if needed), validate and report on the dataset."""
    path = download_dataset()
    verify_checksum(path)
    print(f"Dataset cached at {path} (SHA-256 verified)")

    frame = load_dataset()
    cleaned, report = clean_dataset(frame)
    print(f"Expected rows: {DATASET_EXPECTED_ROWS}")
    stats = dataset_stats(cleaned)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("Cleaning report:")
    for key, value in report.as_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

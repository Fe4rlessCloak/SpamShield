"""Tests for dataset loading, checksum verification and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from spamshield import data
from spamshield.data import (
    COLUMNS,
    DatasetError,
    clean_dataset,
    dataset_stats,
    load_dataset,
    read_metadata,
    sha256_of_bytes,
    text_length_table,
    verify_checksum,
)


class TestLoadDataset:
    def test_reads_tab_separated_records(self, tsv_dataset: Path) -> None:
        frame = load_dataset(tsv_dataset, download=False)

        assert list(frame.columns) == COLUMNS
        assert len(frame) == 4
        assert set(frame["label"]) == {"ham", "spam"}

    def test_preserves_quoted_message_text(self, tsv_dataset: Path) -> None:
        frame = load_dataset(tsv_dataset, download=False)

        assert frame.loc[1, "message"] == '"Free money now!" Text 80082 to claim'
        assert frame.loc[2, "message"] == 'This message ends with a quote"'

    def test_missing_file_without_download_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="Dataset not found"):
            load_dataset(tmp_path / "does-not-exist", download=False)


class TestCleanDataset:
    def test_drops_empty_and_duplicate_rows(self, sample_frame: pd.DataFrame) -> None:
        cleaned, report = clean_dataset(sample_frame)

        assert report.raw_rows == 6
        assert report.empty_messages == 1
        assert report.duplicate_messages == 1
        assert report.duplicate_rows_removed == 1
        assert report.final_rows == 4
        assert len(cleaned) == 4

    def test_encodes_labels_using_shared_mapping(self, sample_frame: pd.DataFrame) -> None:
        cleaned, _ = clean_dataset(sample_frame)

        assert set(cleaned["label_id"]) == {0, 1}
        assert cleaned.loc[cleaned["label"] == "spam", "label_id"].eq(1).all()
        assert cleaned.loc[cleaned["label"] == "ham", "label_id"].eq(0).all()

    def test_reports_conflicting_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "label": ["ham", "spam", "spam"],
                "message": ["Ambiguous", "Ambiguous", "A unique spam message"],
            }
        )
        cleaned, report = clean_dataset(frame)

        assert report.conflicting_label_messages == 1
        assert report.duplicate_rows_removed == 1
        assert len(cleaned) == 2

    def test_drops_records_with_unknown_labels(self) -> None:
        frame = pd.DataFrame({"label": ["ham", "HAM", "junk"], "message": ["one", "two", "three"]})
        cleaned, report = clean_dataset(frame)

        assert report.unknown_labels == 1
        assert report.final_rows == 2

    def test_rejects_frames_without_expected_columns(self) -> None:
        with pytest.raises(DatasetError, match="Expected columns"):
            clean_dataset(pd.DataFrame({"text": ["hello"]}))


class TestStatistics:
    def test_dataset_stats_reports_label_distribution(self, sample_frame: pd.DataFrame) -> None:
        cleaned, _ = clean_dataset(sample_frame)
        stats = dataset_stats(cleaned)

        assert stats["total_messages"] == 4
        assert stats["ham_messages"] == 2
        assert stats["spam_messages"] == 2
        assert 0.0 < stats["spam_ratio"] <= 1.0
        assert stats["mean_characters"] > 0

    def test_text_length_table_has_one_row_per_label(self, sample_frame: pd.DataFrame) -> None:
        cleaned, _ = clean_dataset(sample_frame)
        table = text_length_table(cleaned)

        assert set(table.index) == {"ham", "spam"}


class TestChecksums:
    def test_sha256_of_bytes_matches_known_digest(self) -> None:
        digest = "b5765abd4c8e0926191828e48f1a7560491b6f4faa6dabb832ff4d7a5bf2b88f"
        assert sha256_of_bytes(b"spamshield") == digest

    def test_sha256_of_file_matches_sha256_of_bytes(self, tsv_dataset: Path) -> None:
        assert data.sha256_of_file(tsv_dataset) == sha256_of_bytes(tsv_dataset.read_bytes())

    def test_read_metadata_returns_empty_dict_when_missing(self, tmp_path: Path) -> None:
        assert read_metadata(tmp_path / "missing.json") == {}

    def test_verify_checksum_accepts_matching_manifest(
        self, tsv_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "dataset_metadata.json"
        manifest.write_text(
            json.dumps({"data_sha256": data.sha256_of_file(tsv_dataset)}),
            encoding="utf-8",
        )
        monkeypatch.setattr(data, "METADATA_PATH", manifest)

        assert verify_checksum(tsv_dataset) is True

    def test_verify_checksum_rejects_mismatch(
        self, tsv_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = tmp_path / "dataset_metadata.json"
        manifest.write_text(json.dumps({"data_sha256": "0" * 64}), encoding="utf-8")
        monkeypatch.setattr(data, "METADATA_PATH", manifest)

        with pytest.raises(DatasetError, match="Checksum mismatch"):
            verify_checksum(tsv_dataset)

    def test_verify_checksum_without_manifest_raises(
        self, tsv_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(data, "METADATA_PATH", tmp_path / "missing.json")

        with pytest.raises(DatasetError, match="metadata manifest"):
            verify_checksum(tsv_dataset)

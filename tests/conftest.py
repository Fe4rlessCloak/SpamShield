"""Shared pytest fixtures.

All fixtures are tiny and fully local: the test suite never downloads the dataset and
never trains a full model, so it stays fast and works offline in CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A small raw dataframe with an empty message and one exact duplicate."""
    return pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "spam", "ham", "ham"],
            "message": [
                "Hey, are we still on for lunch tomorrow at 1pm?",
                "Win a FREE iPhone now! Text WIN to 80082 to claim your prize",
                "Ok, see you at 6pm.",
                "Congratulations! You have won 1000 cash. Call 09061701461 now",
                "   ",
                "Hey, are we still on for lunch tomorrow at 1pm?",
            ],
        }
    )


@pytest.fixture
def sample_messages() -> list[str]:
    return [
        "Free entry in a weekly competition, text WIN to 80082 now",
        "Are we still meeting for lunch tomorrow at one?",
        "Congratulations, you have won a 1000 cash prize, call now",
        "Ok, see you later at the library",
        "URGENT! Your mobile number has won a 2000 prize, claim 08712300220",
        "Can you pick up some milk on your way home please",
    ]


@pytest.fixture
def tsv_dataset(tmp_path: Path) -> Path:
    """A tiny tab-separated dataset file, including quoted message text."""
    path = tmp_path / "SMSSpamCollection"
    rows = [
        ("ham", "Just a normal message"),
        ("spam", '"Free money now!" Text 80082 to claim'),
        ("ham", 'This message ends with a quote"'),
        ("spam", "Win a prize, call 09061701461"),
    ]
    path.write_text(
        "".join(f"{label}\t{message}\n" for label, message in rows),
        encoding="utf-8",
    )
    return path

"""Page 3 — Dataset Explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from spamshield import ui
from spamshield.constants import (
    DATASET_CITATION,
    DATASET_DATASET_URL,
    DATASET_LICENSE,
    DATASET_NAME,
    DATASET_SOURCE,
)
from spamshield.data import DatasetError

st.title("Dataset Explorer")
st.write(f"Explore the **{DATASET_NAME}** used to train and evaluate every model.")

try:
    frame, report = ui.get_dataset()
except DatasetError as error:
    st.error(str(error))
    st.write("Download the dataset and train the models with:")
    st.code("uv run python -m spamshield.train", language="bash")
    st.stop()

counts = frame["label"].value_counts()
ham_count = int(counts.get("ham", 0))
spam_count = int(counts.get("spam", 0))
total = len(frame)

columns = st.columns(4)
columns[0].metric("Messages (after cleaning)", f"{total:,}")
columns[1].metric("Ham", f"{ham_count:,}")
columns[2].metric("Spam", f"{spam_count:,}")
columns[3].metric("Spam share", f"{spam_count / total:.1%}" if total else "n/a")

st.caption(
    f"{report.get('raw_rows', 'n/a')} raw rows were reduced to {total} unique messages; "
    f"{report.get('duplicate_rows_removed', 0)} exact duplicate rows were removed before "
    f"splitting, so no message appears in both the training and the test set."
)

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Class distribution")
    st.bar_chart(pd.DataFrame({"messages": counts}))

with right:
    st.subheader("Message length (characters)")
    lengths = frame["message"].str.len().to_numpy()
    bins = np.linspace(0, min(float(lengths.max()) if lengths.size else 160, 400), 31)
    histogram, edges = np.histogram(lengths, bins=bins)
    st.bar_chart(
        pd.DataFrame(
            {"messages": histogram},
            index=[f"{int(edge)}" for edge in edges[:-1]],
        ),
        x_label="Characters",
        y_label="Messages",
    )

st.subheader("Length statistics by label")
stats = (
    pd.DataFrame(
        {
            "label": frame["label"],
            "characters": frame["message"].str.len(),
            "words": frame["message"].str.split().str.len(),
        }
    )
    .groupby("label")
    .agg(
        messages=("characters", "size"),
        mean_characters=("characters", "mean"),
        median_characters=("characters", "median"),
        mean_words=("words", "mean"),
    )
    .round(1)
)
st.dataframe(stats, width="stretch")
st.caption(
    "Spam messages tend to be longer and to contain more numbers, phone numbers and "
    "currency amounts than ordinary messages."
)

st.subheader("Example messages")
label_filter = st.radio("Show", ["spam", "ham"], horizontal=True)
examples = frame.loc[frame["label"] == label_filter, "message"]
st.dataframe(
    examples.sample(min(10, len(examples)), random_state=42).rename("message"),
    width="stretch",
)

st.divider()
st.markdown(
    f"""
**Attribution** — {DATASET_NAME}, {DATASET_SOURCE}
([dataset page]({DATASET_DATASET_URL})), licensed **{DATASET_LICENSE}**.

> {DATASET_CITATION}
"""
)

"""Page 2 — Model Comparison.

Everything on this page is loaded from the saved evaluation artifacts; no model is ever
retrained when the page is opened.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from spamshield import ui
from spamshield.models import MODEL_NAMES

st.title("Model Comparison")
st.write(
    "All six algorithms were trained on the same data with the same TF-IDF features and "
    "evaluated on the same held-out test set, so the numbers are directly comparable."
)

ui.require_metrics()

frame = ui.get_metrics_frame()
payload = ui.get_metrics_payload()
selected_key = payload.get("selected_model", "")
models = payload.get("models", {})

if selected_key:
    selected = models.get(selected_key, {})
    st.success(
        f"**Selected model: {MODEL_NAMES.get(selected_key, selected_key)}** — chosen by "
        f"training-only cross-validated spam F1 "
        f"({selected.get('cv_f1_spam', float('nan')):.4f})."
    )
    st.caption(payload.get("selection_note", ""))

st.subheader("Metrics")
display = pd.DataFrame(
    {
        "Model": frame["model"],
        "Accuracy": frame["accuracy"],
        "Precision (spam)": frame["precision_spam"],
        "Recall (spam)": frame["recall_spam"],
        "F1 (spam)": frame["f1_spam"],
        "CV F1 (spam)": frame["cv_f1_spam"],
        "Train (s)": frame["train_seconds"],
        "Inference (ms/msg)": frame["inference_ms_per_message"],
        "Peak traced mem (MB)": frame["peak_memory_mb"],
    }
)
st.dataframe(display, hide_index=True, width="stretch")
st.caption(
    "Accuracy is misleading on this imbalanced dataset (spam is only ~13% of the "
    "messages), which is why the spam-class precision, recall and F1 drive the "
    "comparison."
)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Accuracy vs F1 (spam)")
    st.bar_chart(frame.set_index("model")[["accuracy", "f1_spam"]])
with right:
    st.subheader("Precision vs recall (spam)")
    st.bar_chart(frame.set_index("model")[["precision_spam", "recall_spam"]])

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Training time")
    st.bar_chart(frame.set_index("model")[["train_seconds"]], y_label="Seconds")
with right:
    st.subheader("Inference time per message")
    st.bar_chart(
        frame.set_index("model")[["inference_ms_per_message"]],
        y_label="Milliseconds",
    )

st.divider()
st.subheader("Confusion matrices")
st.caption(
    "Rows are the true labels and columns the predicted labels. The bottom-left cell is "
    "the spam that slipped through; the top-right cell is ham wrongly flagged as spam."
)
for _, row in frame.iterrows():
    key = row["model_key"]
    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.show_figure(f"confusion_matrix_{key}.png", caption=row["model"])
    with right:
        report = models.get(key, {}).get("report")
        if report:
            st.markdown(f"**Classification report — {row['model']}**")
            st.dataframe(
                pd.DataFrame(report).transpose().round(3),
                width="stretch",
            )

st.divider()
st.subheader("Generated comparison figures")
figure_columns = st.columns(2, gap="large")
with figure_columns[0]:
    ui.show_figure("accuracy_comparison.png")
with figure_columns[1]:
    ui.show_figure("f1_comparison.png")

with st.expander("Notes on how these results were produced"):
    st.markdown(
        f"""
- Selection metric: **{payload.get("selection_metric", "n/a")}**.
- Test rows: **{payload.get("test_rows", "n/a")}**.
- {payload.get("memory_note", "")}
- Figures are written to `reports/figures/` and metrics to `artifacts/metrics/` by
  `uv run python -m spamshield.train`.
"""
    )

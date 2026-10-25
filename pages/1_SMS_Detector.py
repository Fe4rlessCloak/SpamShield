"""Page 1 — SMS Spam Detector."""

from __future__ import annotations

import streamlit as st

from spamshield import predict
from spamshield.evaluate import ModelArtifactError
from spamshield.models import MODEL_NAMES
from spamshield.ui import require_trained_models

st.title("SMS Spam Detector")
st.write(
    "Type an SMS message, pick one of the six trained models and classify it as "
    "**spam** or **ham** (legitimate)."
)

model_keys = require_trained_models()

with st.form("detector"):
    message = st.text_area(
        "SMS message",
        height=150,
        placeholder=(
            "e.g. Congratulations! You have won a £1000 prize. Call 09061701461 to claim now"
        ),
    )
    left, right = st.columns([3, 2])
    with left:
        selected_key = st.selectbox(
            "Model",
            model_keys,
            format_func=lambda key: MODEL_NAMES[key],
            help="Every model was trained on exactly the same data and features.",
        )
    with right:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Analyze Message", type="primary")

if submitted:
    if not message.strip():
        st.warning("Please enter a message before analysing.")
    else:
        try:
            with st.spinner(f"Classifying with {MODEL_NAMES[selected_key]} ..."):
                result = predict.predict_message(message, selected_key)
        except ModelArtifactError as error:
            st.error(str(error))
        except ValueError as error:
            st.warning(str(error))
        else:
            st.divider()

            if result.is_spam:
                st.error("### 🚫 SPAM\nThis message looks like spam.")
            else:
                st.success("### ✅ HAM\nThis message looks legitimate.")

            columns = st.columns(3, gap="medium")
            with columns[0]:
                st.metric("Prediction", result.label.upper())
            with columns[1]:
                if result.has_probability:
                    st.metric("Confidence (spam)", f"{result.probability * 100:.1f}%")
                else:
                    st.metric("Confidence", "not available")
            with columns[2]:
                st.metric("Model used", result.model_name)

            if result.has_probability:
                assert result.probability is not None
                st.progress(min(max(result.probability, 0.0), 1.0))
                st.caption(
                    f"Probability of spam: **{result.probability:.3f}** "
                    f"(spam threshold 0.5, so anything above 50% is classified as spam)."
                )
            elif result.decision_score is not None:
                st.metric("Decision score", f"{result.decision_score:+.3f}")
                st.caption(
                    "LinearSVC has no probability output, so a **decision score** is shown "
                    "instead of a confidence: positive values mean spam, negative values "
                    "mean ham. This is a signed distance from the separating hyperplane, "
                    "**not** a probability."
                )
            else:  # pragma: no cover - every bundled model exposes one of the two
                st.caption("This model provides neither probabilities nor a decision score.")

            st.info(f"**{result.model_name}** — {result.explanation}\n\n_{result.notes}_")

            with st.expander("How this message was preprocessed"):
                st.write("Text actually seen by the TF-IDF vectorizer:")
                st.code(result.normalized_text or "(empty)", language=None)
                st.caption(
                    "Lower-casing and whitespace normalisation are applied, but numbers, "
                    "currency symbols and SMS wording are kept because they carry much of "
                    "the spam signal."
                )

st.divider()
st.caption(
    "New here? Start with the **Model Comparison** page to see how the six algorithms "
    "perform, or read **About the Project** for how the pipeline works."
)

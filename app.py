"""SpamShield — Streamlit entry point.

Run with::

    uv run streamlit run app.py

The application never trains anything on startup: it loads the models and metrics that
were produced by ``uv run python -m spamshield.train``.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SpamShield — SMS Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    {
        "Classifier": [
            st.Page(
                "pages/1_SMS_Detector.py",
                title="SMS Detector",
                icon="🔍",
                default=True,
            ),
        ],
        "Analysis": [
            st.Page("pages/2_Model_Comparison.py", title="Model Comparison", icon="📊"),
            st.Page("pages/3_Dataset_Explorer.py", title="Dataset Explorer", icon="📁"),
        ],
        "Reference": [
            st.Page("pages/4_About.py", title="About the Project", icon="ℹ️"),
        ],
    }
)

with st.sidebar:
    st.markdown("### 🛡️ SpamShield")
    st.caption("SMS spam detection with six classic machine learning algorithms.")
    st.divider()

navigation.run()

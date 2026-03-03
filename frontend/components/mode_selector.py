import streamlit as st


def select_mode() -> str:
    return st.selectbox(
        "Select Extraction Mode",
        ["Fast", "Balanced", "High Accuracy"],
        index=2,
        help="High Accuracy is the default because certificate OCR is dense and table-heavy.",
    )

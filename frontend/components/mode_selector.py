import streamlit as st

def select_mode():
    mode = st.selectbox(
        "Select Extraction Mode",
        ["Fast", "Balanced", "High Accuracy"]
    )
    return mode
import streamlit as st

def upload_file():
    uploaded_file = st.file_uploader(
        "Upload Certificate",
        type=["pdf", "png", "jpg", "jpeg", "webp", "tiff", "bmp"]
    )
    return uploaded_file

import streamlit as st

def upload_file(label: str = "Upload Certificate", key: str | None = None):
    uploaded_file = st.file_uploader(
        label,
        type=["pdf", "png", "jpg", "jpeg", "webp", "tiff", "bmp"],
        key=key,
    )
    return uploaded_file

import streamlit as st

from frontend.components.file_uploader import upload_file
from frontend.components.mode_selector import select_mode
from frontend.components.preview_panel import show_preview
from frontend.components.json_viewer import show_json
from frontend.components.confidence_display import show_confidence
from frontend.utils.file_handler import save_uploaded_file
from backend.pipeline import run_pipeline

st.set_page_config(page_title="Certificate Extractor", layout="wide")

st.title("📜 Certificate Extractor")

uploaded_file = upload_file()

if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    show_preview(file_path)

    mode = select_mode()

    if st.button("Extract Data"):
        with st.spinner("Processing..."):
            result = run_pipeline(file_path, mode)

        show_json(result)
        show_confidence(result)
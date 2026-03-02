import streamlit as st
from PIL import Image
from pdf2image import convert_from_path

def show_preview(file_path):
    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file_path)
        st.image(image, caption="Uploaded Certificate Preview", use_column_width=True)
    else:
        try:
            first_page = convert_from_path(file_path, first_page=1, last_page=1, dpi=150)[0]
            st.image(first_page, caption="PDF Preview (Page 1)", use_column_width=True)
        except Exception:
            st.info("PDF uploaded. Preview unavailable (Poppler may be missing).")

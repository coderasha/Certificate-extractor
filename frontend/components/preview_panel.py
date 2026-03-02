import streamlit as st
from PIL import Image

def show_preview(file_path):
    if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file_path)
        st.image(image, caption="Uploaded Certificate Preview", use_column_width=True)
    else:
        st.info("PDF uploaded. Preview not enabled yet.")
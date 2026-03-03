import streamlit as st

from frontend.components.file_uploader import upload_file
from frontend.components.mode_selector import select_mode
from frontend.components.preview_panel import show_preview
from frontend.components.json_viewer import show_json
from frontend.components.confidence_display import show_confidence
from frontend.utils.file_handler import save_uploaded_file
from backend.pipeline import run_pipeline


def main() -> None:
    st.set_page_config(page_title="Certificate Extractor", layout="wide")
    st.title("Certificate Extractor")
    st.caption("Upload a certificate (PDF/image) and send it directly to LLaMA Vision 3.2 for JSON extraction.")

    with st.sidebar:
        st.subheader("Model Settings")
        model = st.text_input("Model", value="llama3.2-vision")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434/api/chat")
        use_custom_timeout = st.checkbox("Use custom timeout", value=False)
        timeout = st.number_input("Timeout (seconds)", min_value=30, max_value=1200, value=150)

    uploaded_file = upload_file()
    if not uploaded_file:
        st.info("Upload a certificate file to begin.")
        return

    file_path = save_uploaded_file(uploaded_file)
    show_preview(file_path)

    mode = select_mode()

    if st.button("Extract Data", type="primary"):
        with st.spinner("Processing certificate..."):
            try:
                result = run_pipeline(
                    file_path=file_path,
                    mode=mode,
                    model=model,
                    ollama_url=ollama_url,
                    timeout_override=int(timeout) if use_custom_timeout else None,
                )
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                return

        show_json(result)
        show_confidence(result)


if __name__ == "__main__":
    main()

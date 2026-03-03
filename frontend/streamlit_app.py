import streamlit as st

from backend.pipeline import run_pipeline
from frontend.components.confidence_display import show_confidence
from frontend.components.file_uploader import upload_file
from frontend.components.json_viewer import show_json
from frontend.components.mode_selector import select_mode
from frontend.components.preview_panel import show_preview
from frontend.utils.file_handler import save_uploaded_file


def main() -> None:
    st.set_page_config(page_title="Certificate Extractor", layout="wide")
    st.title("Certificate Extractor")
    st.caption("Upload a certificate (PDF/image) and send it directly to LLaMA Vision 3.2 for JSON extraction.")

    with st.sidebar:
        st.subheader("Model Settings")
        model = st.text_input("Model", value="llama3.2-vision")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434/api/chat")
        use_custom_timeout = st.checkbox("Use custom timeout", value=False)
        timeout = st.number_input("Timeout (seconds)", min_value=30, max_value=1200, value=240)
        show_debug = st.checkbox("Show raw model debug", value=False)

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
                output = run_pipeline(
                    file_path=file_path,
                    mode=mode,
                    model=model,
                    ollama_url=ollama_url,
                    timeout_override=int(timeout) if use_custom_timeout else None,
                    include_debug=show_debug,
                )
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                return

        if show_debug:
            result, debug_info = output
        else:
            result = output
            debug_info = None

        show_json(result)
        show_confidence(result)

        if debug_info is not None:
            with st.expander("Debug: Ollama raw response"):
                st.caption(f"Status: {debug_info.get('status', 'unknown')}")
                st.caption(f"Images sent: {debug_info.get('images_count', 0)}")
                st.text_area("Text context preview", value=debug_info.get("text_context_preview", ""), height=180)
                st.text_area("Raw model content", value=debug_info.get("raw_model_content", ""), height=220)
                if debug_info.get("repair_model_content"):
                    st.text_area(
                        "Repair model content",
                        value=debug_info.get("repair_model_content", ""),
                        height=180,
                    )


if __name__ == "__main__":
    main()

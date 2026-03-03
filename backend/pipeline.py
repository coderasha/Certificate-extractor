from backend.speed_modes import MODE_CONFIG
from extractor import CertificateExtractor


def run_pipeline(
    file_path: str,
    mode: str,
    model: str = "llama3.2-vision",
    ollama_url: str = "http://localhost:11434/api/chat",
    timeout_override: int | None = None,
) -> dict:
    config = MODE_CONFIG.get(mode, MODE_CONFIG["Balanced"])
    timeout = timeout_override if timeout_override is not None else config["timeout"]

    extractor = CertificateExtractor(
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        max_pdf_pages=config["max_pdf_pages"],
        vision_dpi=config["vision_dpi"],
        max_image_dim=config["max_image_dim"],
        jpeg_quality=config["jpeg_quality"],
    )
    return extractor.extract(file_path)

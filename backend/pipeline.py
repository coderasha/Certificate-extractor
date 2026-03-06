from typing import Any

from backend.speed_modes import MODE_CONFIG
from backend.template_learning import TemplateLearningEngine
from extractor import CertificateExtractor


def run_pipeline(
    file_path: str,
    mode: str,
    model: str = "llama3.2-vision",
    ollama_url: str = "http://localhost:11434/api/chat",
    timeout_override: int | None = None,
    include_debug: bool = False,
) -> Any:
    config = MODE_CONFIG.get(mode, MODE_CONFIG["High Accuracy"])
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

    template_engine = TemplateLearningEngine()
    template_result, template_debug = template_engine.extract(file_path)

    if include_debug:
        model_result, model_debug = extractor.extract_with_debug(file_path)
    else:
        model_result = extractor.extract(file_path)
        model_debug = None

    final_result = model_result
    template_confidence = float(template_debug.get("template_confidence", 0.0))
    if template_result:
        if template_confidence >= 0.7:
            merged = CertificateExtractor._merge_candidate_data(template_result, model_result)
        else:
            merged = CertificateExtractor._merge_candidate_data(model_result, template_result)
        final_result = CertificateExtractor._normalize_result(merged)

    if include_debug:
        debug_payload = model_debug if isinstance(model_debug, dict) else {}
        debug_payload["template_learning"] = template_debug
        return final_result, debug_payload

    return final_result

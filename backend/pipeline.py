from typing import Any

from backend.speed_modes import MODE_CONFIG
from backend.template_learning import TemplateLearningEngine
from extractor import CertificateExtractor


def run_pipeline(
    file_path: str,
    mode: str,
    timeout_override: int | None = None,
    include_debug: bool = False,
) -> Any:
    config = MODE_CONFIG.get(mode, MODE_CONFIG["High Accuracy"])
    timeout = timeout_override if timeout_override is not None else config["timeout"]

    extractor = CertificateExtractor(
        timeout=timeout,
        max_pdf_pages=config["max_pdf_pages"],
        vision_dpi=config["vision_dpi"],
    )

    template_engine = TemplateLearningEngine()
    template_result, template_debug = template_engine.extract(file_path)

    if include_debug:
        ocr_result, ocr_debug = extractor.extract_with_debug(file_path)
    else:
        ocr_result = extractor.extract(file_path)
        ocr_debug = None

    final_result = ocr_result
    template_confidence = float(template_debug.get("template_confidence", 0.0))
    if template_result:
        if template_confidence >= 0.7:
            merged = CertificateExtractor._merge_candidate_data(template_result, ocr_result)
        else:
            merged = CertificateExtractor._merge_candidate_data(ocr_result, template_result)
        final_result = CertificateExtractor._normalize_result(merged)

    if include_debug:
        debug_payload = ocr_debug if isinstance(ocr_debug, dict) else {}
        debug_payload["template_learning"] = template_debug
        return final_result, debug_payload

    return final_result

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pdfplumber
import pytesseract
import requests
from pdf2image import convert_from_path
from PIL import Image


TARGET_FIELDS = [
    "NAME",
    "EXAMINATION",
    "HELD IN",
    "SEAT NUMBER",
    "SPECIALIZATION",
    "AICTE NUMBER",
    "TRIMESTER I",
    "TRIMESTER II",
    "TRIMESTER III",
    "TRIMESTER IV",
    "TRIMESTER TRIMESTER I",
    "TRIMESTER VI",
    "FINAL CGPA",
    "Total Credits",
    "Total Grade Points",
    "Total Marks Obtained",
    "Result Declared On",
]

ALIASES = {
    "NAME": ["STUDENT NAME", "CANDIDATE NAME"],
    "EXAMINATION": ["EXAM", "PROGRAMME", "COURSE"],
    "HELD IN": ["HELD_IN", "MONTH/YEAR", "SESSION"],
    "SEAT NUMBER": ["SEAT_NO", "SEAT NO", "ROLL NUMBER", "ROLL NO"],
    "SPECIALIZATION": ["SPECIALISATION", "BRANCH", "STREAM"],
    "AICTE NUMBER": ["AICTE NO", "AICTE", "AICTE_NUMBER"],
    "TRIMESTER I": ["TRIMESTER 1", "SEMESTER I", "SEMESTER 1"],
    "TRIMESTER II": ["TRIMESTER 2", "SEMESTER II", "SEMESTER 2"],
    "TRIMESTER III": ["TRIMESTER 3", "SEMESTER III", "SEMESTER 3"],
    "TRIMESTER IV": ["TRIMESTER 4", "SEMESTER IV", "SEMESTER 4"],
    "TRIMESTER TRIMESTER I": ["TRIMESTER V", "SEMESTER V", "SEMESTER 5", "TRIMESTER 5"],
    "TRIMESTER VI": ["TRIMESTER 6", "SEMESTER VI", "SEMESTER 6"],
    "FINAL CGPA": ["CGPA", "FINAL_GPA"],
    "Total Credits": ["TOTAL CREDITS", "TOTAL_CREDITS"],
    "Total Grade Points": ["TOTAL GRADE POINTS", "TOTAL_GRADE_POINTS"],
    "Total Marks Obtained": ["TOTAL MARKS", "TOTAL MARKS OBTAINED", "TOTAL_MARKS_OBTAINED"],
    "Result Declared On": ["RESULT DATE", "DECLARED ON", "RESULT_DECLARED_ON"],
}


class CertificateExtractor:
    def __init__(
        self,
        model: str = "llama3.2-vision",
        ollama_url: str = "http://localhost:11434/api/chat",
        timeout: int = 180,
        max_pdf_pages: int = 2,
        ocr_max_pages: int = 2,
        vision_dpi: int = 130,
        ocr_dpi: int = 170,
        max_image_dim: int = 1200,
        jpeg_quality: int = 62,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.max_pdf_pages = max_pdf_pages
        self.ocr_max_pages = ocr_max_pages
        self.vision_dpi = vision_dpi
        self.ocr_dpi = ocr_dpi
        self.max_image_dim = max_image_dim
        self.jpeg_quality = jpeg_quality

    def extract(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        text = self.extract_text(path)

        # Stage 1: text-only extraction (typically fastest and enough for many certificates)
        text_only = self.extract_structured_data(text=text, images=[])
        if not self._needs_refinement(text_only):
            return text_only

        images = self._prepare_visual_inputs(path)
        if not images:
            return text_only

        # Stage 2: single-image vision extraction
        one_image = self.extract_structured_data(text=text, images=[images[0]])
        best = one_image if self._result_score(one_image) >= self._result_score(text_only) else text_only

        # Stage 3: multi-image only when still uncertain and extra pages exist
        if len(images) > 1 and self._needs_refinement(best):
            multi = self.extract_structured_data(text=text, images=images)
            if self._result_score(multi) >= self._result_score(best):
                best = multi

        return best

    def extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_text_from_pdf(path)
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
            return self._extract_text_from_image(path)
        raise ValueError(
            "Unsupported file type. Supported types: PDF, PNG, JPG, JPEG, WEBP, TIFF, BMP"
        )

    def _extract_text_from_pdf(self, path: Path) -> str:
        direct_text_chunks: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = (page.extract_text() or "").strip()
                if txt:
                    direct_text_chunks.append(txt)

        direct_text = "\n\n".join(direct_text_chunks).strip()
        if len(direct_text) >= 120:
            return direct_text

        # OCR only on first pages when direct text is weak
        page_count = self._pdf_page_count(path)
        last_page = min(page_count, self.ocr_max_pages)
        if last_page <= 0:
            raise ValueError("PDF has no pages")

        ocr_chunks: list[str] = []
        page_images = convert_from_path(
            str(path),
            dpi=self.ocr_dpi,
            fmt="jpeg",
            first_page=1,
            last_page=last_page,
            thread_count=2,
        )
        for image in page_images:
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                ocr_chunks.append(ocr_text)

        merged = "\n\n".join([direct_text, *ocr_chunks]).strip()
        if not merged:
            raise ValueError("No text could be extracted from the PDF")
        return merged

    def _extract_text_from_image(self, path: Path) -> str:
        with Image.open(path) as image:
            text = pytesseract.image_to_string(image).strip()
        if not text:
            raise ValueError("No text could be extracted from the image")
        return text

    def extract_structured_data(self, text: str, images: list[str]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": "Extract certificate fields and return strict JSON only.",
                },
                {
                    "role": "user",
                    "content": self._build_prompt(text),
                    **({"images": images} if images else {}),
                },
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_predict": 120 if not images else (150 if len(images) == 1 else 200),
            },
        }

        response = self._post_with_fallback(payload, has_images=bool(images))

        body = response.json()
        raw_content = body.get("message", {}).get("content", "")
        if not raw_content:
            raise ValueError("Model response was empty")

        parsed = self._parse_model_json(raw_content)
        result = self._normalize_result(parsed)
        return result.to_dict()

    def _post_with_fallback(self, payload: dict[str, Any], has_images: bool) -> requests.Response:
        primary_timeout = self.timeout
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=primary_timeout)
        except requests.ReadTimeout:
            if has_images:
                fallback = self._make_lighter_payload(payload)
                try:
                    response = requests.post(
                        self.ollama_url,
                        json=fallback,
                        timeout=min(int(primary_timeout * 1.25), 600),
                    )
                except requests.RequestException as exc:
                    raise RuntimeError(f"Failed to call LLaMA Vision endpoint: {exc}") from exc
            else:
                raise RuntimeError(
                    f"Failed to call LLaMA Vision endpoint: Read timed out. (read timeout={primary_timeout})"
                )
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call LLaMA Vision endpoint: {exc}") from exc

        if response.status_code >= 400:
            body_preview = response.text.strip()[:500]
            raise RuntimeError(
                "Failed to call LLaMA Vision endpoint: "
                f"{response.status_code} {response.reason}. Response: {body_preview}"
            )
        return response

    @staticmethod
    def _make_lighter_payload(payload: dict[str, Any]) -> dict[str, Any]:
        lighter = json.loads(json.dumps(payload))
        user_msg = lighter["messages"][1]
        images = user_msg.get("images", [])
        if images:
            user_msg["images"] = images[:1]
        lighter["options"]["num_predict"] = 110
        return lighter

    def _build_prompt(self, text: str) -> str:
        keys = ", ".join([*TARGET_FIELDS, "confidence_score"])
        return (
            "Extract marksheet/certificate info and output STRICT JSON with keys: "
            f"{keys}.\n"
            "Use null if unknown. confidence_score must be float 0..1. No extra text.\n\n"
            f"OCR/Raw Text:\n{text[:2200]}"
        )

    def _prepare_visual_inputs(self, path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            page_count = self._pdf_page_count(path)
            last_page = min(page_count, self.max_pdf_pages)
            if last_page <= 0:
                return []

            page_images = convert_from_path(
                str(path),
                dpi=self.vision_dpi,
                fmt="jpeg",
                first_page=1,
                last_page=last_page,
                thread_count=2,
            )
            return [self._pil_image_to_b64(image) for image in page_images]

        with Image.open(path) as image:
            return [self._pil_image_to_b64(image)]

    @staticmethod
    def _pdf_page_count(path: Path) -> int:
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)

    def _pil_image_to_b64(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        image.thumbnail((self.max_image_dim, self.max_image_dim), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded

    @staticmethod
    def _parse_model_json(raw_content: str) -> dict[str, Any]:
        content = raw_content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Model did not return valid JSON")
            return json.loads(content[start : end + 1])

    @staticmethod
    def _normalize_result(data: dict[str, Any]) -> dict[str, Any]:
        def as_text(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        flattened = {str(k).strip(): v for k, v in data.items()}

        def get_value(key: str) -> str | None:
            direct = as_text(flattened.get(key))
            if direct is not None:
                return direct

            lookup = {k.lower(): v for k, v in flattened.items()}
            alias_candidates = [key, *ALIASES.get(key, [])]
            for alias in alias_candidates:
                candidate = as_text(lookup.get(alias.lower()))
                if candidate is not None:
                    return candidate
            return None

        raw_conf = data.get("confidence_score", 0.0)
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        normalized: dict[str, Any] = {field: get_value(field) for field in TARGET_FIELDS}
        normalized["confidence_score"] = confidence
        return normalized

    @staticmethod
    def _result_score(result: dict[str, Any]) -> float:
        fields = TARGET_FIELDS
        filled = sum(1 for field in fields if result.get(field))
        conf = float(result.get("confidence_score", 0.0) or 0.0)
        return float(filled) + (conf * 2.0)

    @staticmethod
    def _needs_refinement(result: dict[str, Any]) -> bool:
        fields = TARGET_FIELDS
        filled = sum(1 for field in fields if result.get(field))
        conf = float(result.get("confidence_score", 0.0) or 0.0)
        return filled < 8 or conf < 0.75

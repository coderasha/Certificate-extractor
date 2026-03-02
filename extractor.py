from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pdfplumber
import pytesseract
import requests
from pdf2image import convert_from_path
from PIL import Image


@dataclass
class ExtractionResult:
    student_name: str | None
    course_name: str | None
    issue_date: str | None
    certificate_id: str | None
    issuer: str | None
    confidence_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_name": self.student_name,
            "course_name": self.course_name,
            "issue_date": self.issue_date,
            "certificate_id": self.certificate_id,
            "issuer": self.issuer,
            "confidence_score": self.confidence_score,
        }


class CertificateExtractor:
    def __init__(
        self,
        model: str = "llama3.2-vision",
        ollama_url: str = "http://localhost:11434/api/chat",
        timeout: int = 240,
        max_pdf_pages: int = 2,
        ocr_max_pages: int = 2,
        vision_dpi: int = 140,
        ocr_dpi: int = 180,
        max_image_dim: int = 1400,
        jpeg_quality: int = 70,
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
        images = self._prepare_visual_inputs(path)
        if not images:
            raise ValueError("No visual inputs could be prepared for LLaMA Vision")

        fast_result = self.extract_structured_data(text=text, images=images[:1])
        if self._needs_refinement(fast_result) and len(images) > 1:
            refined_result = self.extract_structured_data(text=text, images=images)
            if self._result_score(refined_result) >= self._result_score(fast_result):
                return refined_result

        return fast_result

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
        if len(direct_text) >= 80:
            return direct_text

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
                    "content": "You extract structured fields from certificate documents.",
                },
                {
                    "role": "user",
                    "content": self._build_prompt(text),
                    "images": images,
                },
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_predict": 220,
            },
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
        except requests.ReadTimeout:
            fallback_payload = self._build_fallback_payload(text=text, image=images[0])
            try:
                response = requests.post(
                    self.ollama_url,
                    json=fallback_payload,
                    timeout=min(int(self.timeout * 1.5), 1200),
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"Failed to call LLaMA Vision endpoint: {exc}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call LLaMA Vision endpoint: {exc}") from exc

        if response.status_code >= 400:
            body_preview = response.text.strip()[:500]
            raise RuntimeError(
                "Failed to call LLaMA Vision endpoint: "
                f"{response.status_code} {response.reason}. Response: {body_preview}"
            )

        body = response.json()
        raw_content = body.get("message", {}).get("content", "")
        if not raw_content:
            raise ValueError("Model response was empty")

        parsed = self._parse_model_json(raw_content)
        result = self._normalize_result(parsed)
        return result.to_dict()

    def _build_prompt(self, text: str) -> str:
        return (
            "Extract certificate information and return STRICT JSON only with keys: "
            "student_name, course_name, issue_date, certificate_id, issuer, confidence_score.\n"
            "Rules:\n"
            "- Keep values as strings except confidence_score as float 0..1.\n"
            "- If unavailable, use null.\n"
            "- No extra keys or explanation.\n\n"
            f"OCR/Raw Text:\n{text[:3500]}"
        )

    def _build_fallback_payload(self, text: str, image: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": "Return only JSON for certificate extraction.",
                },
                {
                    "role": "user",
                    "content": self._build_prompt(text[:2000]),
                    "images": [image],
                },
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 160},
        }

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
    def _normalize_result(data: dict[str, Any]) -> ExtractionResult:
        def as_text(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        raw_conf = data.get("confidence_score", 0.0)
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return ExtractionResult(
            student_name=as_text(data.get("student_name")),
            course_name=as_text(data.get("course_name")),
            issue_date=as_text(data.get("issue_date")),
            certificate_id=as_text(data.get("certificate_id")),
            issuer=as_text(data.get("issuer")),
            confidence_score=confidence,
        )

    @staticmethod
    def _result_score(result: dict[str, Any]) -> float:
        fields = ["student_name", "course_name", "issue_date", "certificate_id", "issuer"]
        filled = sum(1 for field in fields if result.get(field))
        conf = float(result.get("confidence_score", 0.0) or 0.0)
        return float(filled) + conf

    @staticmethod
    def _needs_refinement(result: dict[str, Any]) -> bool:
        fields = ["student_name", "course_name", "issue_date", "certificate_id", "issuer"]
        filled = sum(1 for field in fields if result.get(field))
        conf = float(result.get("confidence_score", 0.0) or 0.0)
        return filled < 3 or conf < 0.7

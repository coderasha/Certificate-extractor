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
        timeout: int = 120,
        max_pdf_pages: int = 4,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.max_pdf_pages = max_pdf_pages

    def extract(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        text = self.extract_text(path)
        images = self._prepare_visual_inputs(path)
        parsed = self.extract_structured_data(text=text, images=images)
        return parsed

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
        if len(direct_text) >= 60:
            return direct_text

        ocr_chunks: list[str] = []
        for image in convert_from_path(str(path), dpi=300, fmt="jpeg"):
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
        if not images:
            raise ValueError("No visual inputs could be prepared for LLaMA Vision")

        prompt = (
            "Extract certificate information from the provided text and image. "
            "Return STRICT JSON only with keys: student_name, course_name, issue_date, "
            "certificate_id, issuer, confidence_score.\n"
            "Rules:\n"
            "- Keep values as strings except confidence_score which must be a float from 0 to 1.\n"
            "- If unavailable, use null for that field.\n"
            "- Do not include extra keys or explanations.\n\n"
            f"OCR/Raw Text:\n{text[:6000]}"
        )

        payload = {
            "model": self.model,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": "You extract structured fields from certificates.",
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": images,
                },
            ],
            "stream": False,
            "options": {"temperature": 0},
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
        except requests.ReadTimeout:
            retry_timeout = min(int(self.timeout * 2), 1200)
            try:
                response = requests.post(self.ollama_url, json=payload, timeout=retry_timeout)
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

    def _prepare_visual_inputs(self, path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            page_images = convert_from_path(str(path), dpi=170, fmt="jpeg")
            encoded_pages: list[str] = []
            for image in page_images[: self.max_pdf_pages]:
                encoded_pages.append(self._pil_image_to_b64(image))
            return encoded_pages
        return [self._image_b64(path)]

    @staticmethod
    def _image_b64(path: Path) -> str:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def _pil_image_to_b64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
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

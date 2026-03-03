from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from pdf2image import convert_from_path, pdfinfo_from_path
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
        vision_dpi: int = 130,
        max_image_dim: int = 1200,
        jpeg_quality: int = 62,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.max_pdf_pages = max_pdf_pages
        self.vision_dpi = vision_dpi
        self.max_image_dim = max_image_dim
        self.jpeg_quality = jpeg_quality

    def extract(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        images = self._prepare_visual_inputs(path)
        if not images:
            raise ValueError("Unable to prepare visual inputs from file")

        return self.extract_structured_data(images)

    def extract_structured_data(self, images: list[str]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": "You extract certificate data. Return strict JSON only.",
                },
                {
                    "role": "user",
                    "content": self._build_prompt(),
                    "images": images,
                },
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_predict": 180 if len(images) == 1 else 240,
            },
        }

        response = self._post_with_fallback(payload)
        body = response.json()
        raw_content = body.get("message", {}).get("content", "")

        if not raw_content:
            return self._normalize_result({})

        try:
            parsed = self._parse_model_json(raw_content)
            return self._normalize_result(parsed)
        except ValueError:
            repaired = self._repair_json_response(raw_content)
            if repaired is not None:
                return self._normalize_result(repaired)
            return self._normalize_result({})

    def _post_with_fallback(self, payload: dict[str, Any]) -> requests.Response:
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
        except requests.ReadTimeout:
            fallback = self._make_lighter_payload(payload)
            try:
                response = requests.post(
                    self.ollama_url,
                    json=fallback,
                    timeout=min(int(self.timeout * 1.25), 600),
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
        return response

    @staticmethod
    def _make_lighter_payload(payload: dict[str, Any]) -> dict[str, Any]:
        lighter = json.loads(json.dumps(payload))
        user_msg = lighter["messages"][1]
        images = user_msg.get("images", [])
        if images:
            user_msg["images"] = images[:1]
        lighter["options"]["num_predict"] = 140
        return lighter

    def _build_prompt(self) -> str:
        keys = ", ".join([*TARGET_FIELDS, "confidence_score"])
        return (
            "Extract marksheet/certificate info from the attached image(s) and output STRICT JSON with keys: "
            f"{keys}.\n"
            "Use null if unknown. confidence_score must be float 0..1. No extra text."
        )

    def _repair_json_response(self, raw_content: str) -> dict[str, Any] | None:
        repair_payload = {
            "model": self.model,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": "Convert the user's content to strict valid JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        "Normalize the following extraction output into valid JSON object with these keys only: "
                        + ", ".join([*TARGET_FIELDS, "confidence_score"])
                        + ". Use null for missing fields. Return JSON only.\n\n"
                        + raw_content[:3500]
                    ),
                },
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 140},
        }
        try:
            response = requests.post(
                self.ollama_url,
                json=repair_payload,
                timeout=min(max(45, int(self.timeout * 0.6)), 180),
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            if not content:
                return None
            return self._parse_model_json(content)
        except Exception:
            return None

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

        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
            raise ValueError(
                "Unsupported file type. Supported types: PDF, PNG, JPG, JPEG, WEBP, TIFF, BMP"
            )

        with Image.open(path) as image:
            return [self._pil_image_to_b64(image)]

    @staticmethod
    def _pdf_page_count(path: Path) -> int:
        info = pdfinfo_from_path(str(path))
        return int(info.get("Pages", 0))

    def _pil_image_to_b64(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        image.thumbnail((self.max_image_dim, self.max_image_dim), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_model_json(raw_content: str) -> dict[str, Any]:
        content = raw_content.strip()
        if not content:
            raise ValueError("Model did not return valid JSON")

        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        candidate = CertificateExtractor._extract_balanced_json_object(content)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        fallback = CertificateExtractor._parse_key_value_fallback(content)
        if fallback:
            return fallback

        raise ValueError("Model did not return valid JSON")

    @staticmethod
    def _extract_balanced_json_object(text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    @staticmethod
    def _parse_key_value_fallback(text: str) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip().strip(",")
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().strip('"').strip("'")
            value = value.strip().strip('"').strip("'")
            if key:
                parsed[key] = value if value and value.lower() != "null" else None

        if not parsed:
            return {}

        relevant_keys = {k.lower() for k in TARGET_FIELDS}
        alias_keys = {alias.lower() for aliases in ALIASES.values() for alias in aliases}
        out: dict[str, Any] = {}
        for key, value in parsed.items():
            lower_key = key.lower()
            if lower_key in relevant_keys or lower_key in alias_keys or lower_key == "confidence_score":
                out[key] = value
        return out

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

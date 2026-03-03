from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pdfplumber
import pytesseract
import requests
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image


TARGET_FIELDS = [
    "student_name",
    "course_name",
    "issue_date",
    "certificate_id",
    "issuer",
]

DETAILED_SECTION_DEFAULTS = {
    "institute_details": {
        "name": None,
        "address": None,
        "approval": None,
        "document_type": None,
    },
    "student_details": {
        "name": None,
        "examination": None,
        "held_in": None,
        "specialization": None,
        "seat_number": None,
        "aicte_number": None,
    },
    "course_details": [],
    "result_summary": {
        "total_marks_obtained": None,
        "total_maximum_marks": None,
        "percentage": None,
        "gpa": None,
        "overall_grade": None,
        "grade_range": None,
        "result": None,
    },
    "trimester_wise_performance": [],
    "final_summary": {
        "final_cgpa": None,
        "total_credits": None,
        "total_grade_points": None,
        "total_marks_obtained": None,
        "total_maximum_marks": None,
    },
    "result_declaration": {
        "result_declared_on": None,
        "signed_by": None,
    },
}

ALIASES = {
    "student_name": ["student name", "candidate name", "learner_name"],
    "course_name": [
        "course",
        "course name",
        "program",
        "programme",
        "examination",
        "specialization",
        "specialisation",
    ],
    "issue_date": [
        "date",
        "issue date",
        "issued on",
        "awarded on",
        "result declared on",
        "declared on",
        "held in",
    ],
    "certificate_id": [
        "certificate number",
        "certificate no",
        "certificate id",
        "id",
        "credential id",
        "serial number",
        "seat number",
        "seat no",
        "roll number",
        "roll no",
        "aicte number",
        "aicte no",
    ],
    "issuer": [
        "issuing authority",
        "issued by",
        "organization",
        "organisation",
        "institute",
        "institution",
        "university",
        "board",
    ],
}


class CertificateExtractor:
    def __init__(
        self,
        model: str = "llama3.2-vision",
        ollama_url: str = "http://localhost:11434/api/chat",
        timeout: int = 180,
        max_pdf_pages: int = 3,
        vision_dpi: int = 160,
        max_image_dim: int = 1600,
        jpeg_quality: int = 80,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.max_pdf_pages = max_pdf_pages
        self.vision_dpi = vision_dpi
        self.max_image_dim = max_image_dim
        self.jpeg_quality = jpeg_quality

    def extract(self, file_path: str) -> dict[str, Any]:
        result, _ = self._extract(file_path)
        return result

    def extract_with_debug(self, file_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._extract(file_path)

    def _extract(self, file_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        images = self._prepare_visual_inputs(path)
        if not images:
            raise ValueError("Unable to prepare visual inputs from file")

        text_context = self._extract_text_context(path)
        return self.extract_structured_data(images=images, text_context=text_context)

    def extract_structured_data(self, images: list[str], text_context: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.model,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": "You extract certificate data accurately. Return strict JSON only.",
                },
                {
                    "role": "user",
                    "content": self._build_prompt(text_context),
                    "images": images,
                },
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_predict": 420,
            },
        }

        debug_info: dict[str, Any] = {
            "model": self.model,
            "images_count": len(images),
            "text_context_preview": text_context[:1800],
            "raw_model_content": "",
            "repair_model_content": "",
            "status": "started",
        }

        response = self._post_with_fallback(payload)
        body = response.json()
        raw_content = body.get("message", {}).get("content", "")
        debug_info["raw_model_content"] = raw_content

        if not raw_content:
            debug_info["status"] = "empty_model_content"
            return self._normalize_result({}), debug_info

        try:
            parsed = self._parse_model_json(raw_content)
            debug_info["status"] = "parsed_direct"
            return self._normalize_result(parsed), debug_info
        except ValueError:
            repaired, repair_raw = self._repair_json_response(raw_content)
            debug_info["repair_model_content"] = repair_raw
            if repaired is not None:
                debug_info["status"] = "parsed_repair"
                return self._normalize_result(repaired), debug_info

            debug_info["status"] = "parse_failed"
            return self._normalize_result({}), debug_info

    def _post_with_fallback(self, payload: dict[str, Any]) -> requests.Response:
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
        except requests.ReadTimeout:
            fallback = self._make_lighter_payload(payload)
            try:
                response = requests.post(
                    self.ollama_url,
                    json=fallback,
                    timeout=min(int(self.timeout * 1.3), 600),
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"Failed to call LLaMA Vision endpoint: {exc}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call LLaMA Vision endpoint: {exc}") from exc

        if response.status_code >= 400:
            body_preview = response.text.strip()[:700]
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
        lighter["options"]["num_predict"] = 260
        return lighter

    def _build_prompt(self, text_context: str) -> str:
        keys = ", ".join([*TARGET_FIELDS, *DETAILED_SECTION_DEFAULTS.keys(), "confidence_score"])
        context = text_context.strip()
        context_block = (
            f"\n\nOCR_TEXT_CONTEXT (may be noisy, use image as source of truth):\n{context[:5000]}"
            if context
            else ""
        )

        return (
            "Extract all relevant certificate details from the attached image(s) and output STRICT JSON with keys: "
            f"{keys}.\n"
            "For list sections (course_details, trimester_wise_performance), include all visible entries.\n"
            "Use null for unknown scalar values and [] for unknown list sections.\n"
            "Do not invent values. Do not output any text outside JSON."
            + context_block
        )

    def _repair_json_response(self, raw_content: str) -> tuple[dict[str, Any] | None, str]:
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
                        + ", ".join([*TARGET_FIELDS, *DETAILED_SECTION_DEFAULTS.keys(), "confidence_score"])
                        + ". Use null for missing scalar fields and [] for missing list fields. Return JSON only.\n\n"
                        + raw_content[:5000]
                    ),
                },
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 300},
        }
        try:
            response = requests.post(
                self.ollama_url,
                json=repair_payload,
                timeout=min(max(60, int(self.timeout * 0.7)), 240),
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            if not content:
                return None, ""
            return self._parse_model_json(content), content
        except Exception:
            return None, ""

    def _extract_text_context(self, path: Path) -> str:
        suffix = path.suffix.lower()
        chunks: list[str] = []

        if suffix == ".pdf":
            # Direct text first (best quality when embedded).
            try:
                with pdfplumber.open(path) as pdf:
                    upto = min(len(pdf.pages), self.max_pdf_pages)
                    for idx in range(upto):
                        text = (pdf.pages[idx].extract_text() or "").strip()
                        if text:
                            chunks.append(f"[PDF_TEXT_PAGE_{idx + 1}]\n{text}")
            except Exception:
                pass

            # OCR helps for scanned PDFs.
            try:
                ocr_last = min(self.max_pdf_pages, 2)
                if ocr_last > 0:
                    ocr_images = convert_from_path(
                        str(path),
                        dpi=max(190, self.vision_dpi),
                        fmt="jpeg",
                        first_page=1,
                        last_page=ocr_last,
                        thread_count=2,
                    )
                    for idx, image in enumerate(ocr_images, start=1):
                        text = pytesseract.image_to_string(image).strip()
                        if text:
                            chunks.append(f"[OCR_PAGE_{idx}]\n{text}")
            except Exception:
                pass

            return "\n\n".join(chunks)[:6000]

        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
            try:
                with Image.open(path) as image:
                    text = pytesseract.image_to_string(image).strip()
                return text[:4500] if text else ""
            except Exception:
                return ""

        return ""

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
            raise ValueError("Unsupported file type. Supported types: PDF, PNG, JPG, JPEG, WEBP, TIFF, BMP")

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
        lookup = CertificateExtractor._build_lookup(data)

        def get_value(key: str) -> str | None:
            direct = as_text(flattened.get(key))
            if direct is not None:
                return direct

            alias_candidates = [key, *ALIASES.get(key, [])]
            for alias in alias_candidates:
                candidate = as_text(lookup.get(alias.lower()))
                if candidate is not None:
                    return candidate
            return None

        normalized: dict[str, Any] = {field: get_value(field) for field in TARGET_FIELDS}
        normalized["institute_details"] = CertificateExtractor._normalize_section_dict(
            data.get("institute_details"),
            DETAILED_SECTION_DEFAULTS["institute_details"],
        )
        normalized["student_details"] = CertificateExtractor._normalize_section_dict(
            data.get("student_details"),
            DETAILED_SECTION_DEFAULTS["student_details"],
        )
        normalized["course_details"] = CertificateExtractor._normalize_list_of_dicts(data.get("course_details"))
        normalized["result_summary"] = CertificateExtractor._normalize_section_dict(
            data.get("result_summary"),
            DETAILED_SECTION_DEFAULTS["result_summary"],
        )
        normalized["trimester_wise_performance"] = CertificateExtractor._normalize_list_of_dicts(
            data.get("trimester_wise_performance")
        )
        normalized["final_summary"] = CertificateExtractor._normalize_section_dict(
            data.get("final_summary"),
            DETAILED_SECTION_DEFAULTS["final_summary"],
        )
        normalized["result_declaration"] = CertificateExtractor._normalize_section_dict(
            data.get("result_declaration"),
            DETAILED_SECTION_DEFAULTS["result_declaration"],
        )

        if not normalized["student_details"].get("name"):
            normalized["student_details"]["name"] = normalized["student_name"]
        if not normalized["student_details"].get("examination"):
            normalized["student_details"]["examination"] = normalized["course_name"]
        if not normalized["student_details"].get("seat_number"):
            normalized["student_details"]["seat_number"] = normalized["certificate_id"]
        if not normalized["result_declaration"].get("result_declared_on"):
            normalized["result_declaration"]["result_declared_on"] = normalized["issue_date"]
        if not normalized["institute_details"].get("name"):
            normalized["institute_details"]["name"] = normalized["issuer"]

        if not normalized["student_name"]:
            normalized["student_name"] = normalized["student_details"].get("name")
        if not normalized["course_name"]:
            normalized["course_name"] = normalized["student_details"].get("examination")
        if not normalized["certificate_id"]:
            normalized["certificate_id"] = normalized["student_details"].get("seat_number")
        if not normalized["issue_date"]:
            normalized["issue_date"] = normalized["result_declaration"].get("result_declared_on")
        if not normalized["issuer"]:
            normalized["issuer"] = normalized["institute_details"].get("name")

        raw_conf = data.get("confidence_score", None)
        if raw_conf is not None:
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                confidence = 0.0
        else:
            filled = sum(1 for field in TARGET_FIELDS if normalized.get(field))
            confidence = 0.0 if filled == 0 else min(0.95, 0.45 + (0.1 * filled))
        confidence = max(0.0, min(1.0, confidence))
        normalized["confidence_score"] = confidence
        return normalized

    @staticmethod
    def _build_lookup(data: Any) -> dict[str, Any]:
        lookup: dict[str, Any] = {}

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_name = str(key).strip().lower()
                    if not isinstance(value, (dict, list)):
                        if key_name not in lookup:
                            lookup[key_name] = value
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data if isinstance(data, dict) else {})
        return lookup

    @staticmethod
    def _normalize_section_dict(value: Any, defaults: dict[str, Any]) -> dict[str, Any]:
        out = dict(defaults)
        if not isinstance(value, dict):
            return out
        for key in defaults.keys():
            raw = value.get(key, None)
            if raw is None:
                out[key] = None
            elif isinstance(raw, str):
                text = raw.strip()
                out[key] = text or None
            else:
                out[key] = raw
        return out

    @staticmethod
    def _normalize_list_of_dicts(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                normalized_item: dict[str, Any] = {}
                for key, raw in item.items():
                    if raw is None:
                        normalized_item[str(key)] = None
                    elif isinstance(raw, str):
                        text = raw.strip()
                        normalized_item[str(key)] = text or None
                    else:
                        normalized_item[str(key)] = raw
                if normalized_item:
                    out.append(normalized_item)
        return out

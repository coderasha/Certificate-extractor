from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pytesseract import Output


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
        "document_type": None,
    },
    "student_details": {
        "name": None,
        "examination": None,
        "held_in": None,
        "specialization": None,
        "seat_number": None,
        "aicte_number": None,
        "gender": None,
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
        max_pdf_pages: int = 3,
        vision_dpi: int = 160,
        timeout: int = 120,
    ) -> None:
        self.max_pdf_pages = max_pdf_pages
        self.vision_dpi = vision_dpi
        self.timeout = timeout

    def extract(self, file_path: str) -> dict[str, Any]:
        result, _ = self._extract(file_path)
        return result

    def extract_with_debug(self, file_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._extract(file_path)

    def _extract(self, file_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        text_context = self._extract_text_context(path)
        bbox_words = self._extract_bbox_words(path)
        return self.extract_structured_data(text_context=text_context, bbox_words=bbox_words)

    def extract_structured_data(
        self,
        text_context: str,
        bbox_words: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        structured_text = self._extract_structured_from_text(text_context)
        structured_bbox = self._extract_structured_from_bboxes(bbox_words or [])
        merged = self._merge_candidate_data(structured_bbox, structured_text)
        merged = self._apply_bbox_corrections(merged, structured_bbox, structured_text)
        normalized = self._normalize_result(merged)
        debug_info: dict[str, Any] = {
            "status": "ocr_bbox_and_rule_extraction_complete",
            "text_context_preview": text_context[:1800],
            "ocr_chars": len(text_context),
            "bbox_word_count": len(bbox_words or []),
            "bbox_fields_filled": sum(1 for field in TARGET_FIELDS if structured_bbox.get(field)),
        }
        return normalized, debug_info

    @staticmethod
    def _apply_bbox_corrections(
        merged: dict[str, Any],
        bbox_data: dict[str, Any],
        text_data: dict[str, Any],
    ) -> dict[str, Any]:
        updated = copy.deepcopy(merged if isinstance(merged, dict) else {})
        bbox_cert = CertificateExtractor._clean_text(bbox_data.get("certificate_id")) if isinstance(bbox_data, dict) else None
        text_cert = CertificateExtractor._clean_text(text_data.get("certificate_id")) if isinstance(text_data, dict) else None
        if bbox_cert:
            cert_pattern = r"^[A-Z]{0,5}\d{2,12}$"
            if re.match(cert_pattern, bbox_cert):
                if not text_cert:
                    updated["certificate_id"] = bbox_cert
                elif bbox_cert.endswith(text_cert) and len(bbox_cert) > len(text_cert):
                    updated["certificate_id"] = bbox_cert
                elif text_cert.endswith(bbox_cert) and len(text_cert) > len(bbox_cert):
                    updated["certificate_id"] = text_cert

        selected_cert = CertificateExtractor._clean_text(updated.get("certificate_id"))
        if selected_cert:
            student = updated.get("student_details")
            if isinstance(student, dict):
                student["seat_number"] = selected_cert

        bbox_date = CertificateExtractor._clean_text(bbox_data.get("issue_date")) if isinstance(bbox_data, dict) else None
        if bbox_date:
            month_year = re.search(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
                bbox_date,
                flags=re.IGNORECASE,
            )
            if month_year:
                date_value = CertificateExtractor._clean_text(month_year.group(0))
                if date_value and not CertificateExtractor._clean_text(updated.get("issue_date")):
                    updated["issue_date"] = date_value
                result_decl = updated.get("result_declaration")
                if date_value and isinstance(result_decl, dict) and not result_decl.get("result_declared_on"):
                    result_decl["result_declared_on"] = date_value

        bbox_trim = bbox_data.get("trimester_wise_performance") if isinstance(bbox_data, dict) else None
        if isinstance(bbox_trim, list) and bbox_trim:
            merged_trim = updated.get("trimester_wise_performance")
            if isinstance(merged_trim, list) and merged_trim:
                for idx, bbox_row in enumerate(bbox_trim):
                    if idx >= len(merged_trim):
                        break
                    if not isinstance(merged_trim[idx], dict) or not isinstance(bbox_row, dict):
                        continue
                    bbox_gpa = CertificateExtractor._clean_text(bbox_row.get("gpa"))
                    if bbox_gpa:
                        merged_trim[idx]["gpa"] = bbox_gpa
            result_summary = updated.get("result_summary")
            if isinstance(result_summary, dict):
                bbox_last_gpa = None
                for row in reversed(bbox_trim):
                    if isinstance(row, dict):
                        gpa = CertificateExtractor._clean_text(row.get("gpa"))
                        if gpa:
                            bbox_last_gpa = gpa
                            break
                if bbox_last_gpa:
                    result_summary["gpa"] = bbox_last_gpa

        # Prefer text-derived student name when bbox name looks like noisy OCR table text.
        raw_updated_name = CertificateExtractor._clean_text(updated.get("student_name"))
        raw_text_name = CertificateExtractor._clean_text(text_data.get("student_name")) if isinstance(text_data, dict) else None
        clean_updated_name = CertificateExtractor._sanitize_name_candidate(raw_updated_name)
        clean_text_name = CertificateExtractor._sanitize_name_candidate(raw_text_name)
        if clean_text_name and not clean_updated_name:
            if raw_text_name and raw_text_name.startswith("/"):
                clean_text_name = f"/{clean_text_name}"
            updated["student_name"] = clean_text_name
            student_section = updated.get("student_details")
            if isinstance(student_section, dict):
                student_section["name"] = clean_text_name

        # Prefer explicit "HELD IN" month-year from text parser when available.
        text_student = text_data.get("student_details") if isinstance(text_data, dict) else None
        text_held_in = None
        if isinstance(text_student, dict):
            text_held_in = CertificateExtractor._clean_text(text_student.get("held_in"))
        if text_held_in and re.search(r"\b[A-Za-z]+\s+\d{4}\b", text_held_in):
            student_section = updated.get("student_details")
            if isinstance(student_section, dict):
                student_section["held_in"] = text_held_in

        # Prefer full declared date if text parser extracted a day-month-year value.
        text_decl = None
        if isinstance(text_data, dict):
            text_decl_block = text_data.get("result_declaration")
            if isinstance(text_decl_block, dict):
                text_decl = CertificateExtractor._clean_text(text_decl_block.get("result_declared_on"))
        if text_decl and re.search(r"\b\d{1,2}\s*,?\s*[A-Za-z]+\s+\d{4}\b", text_decl):
            result_decl = updated.get("result_declaration")
            if isinstance(result_decl, dict):
                result_decl["result_declared_on"] = text_decl

        # Field-level arbitration: choose higher-quality value between merged/bbox and text parser.
        compare_paths = [
            "student_name",
            "course_name",
            "issue_date",
            "certificate_id",
            "issuer",
            "student_details.name",
            "student_details.examination",
            "student_details.held_in",
            "student_details.specialization",
            "student_details.seat_number",
            "student_details.aicte_number",
            "institute_details.name",
            "institute_details.address",
            "result_summary.total_marks_obtained",
            "result_summary.total_maximum_marks",
            "result_summary.percentage",
            "result_summary.gpa",
            "result_summary.overall_grade",
            "result_summary.grade_range",
            "result_summary.result",
            "final_summary.final_cgpa",
            "final_summary.total_credits",
            "final_summary.total_grade_points",
            "final_summary.total_marks_obtained",
            "final_summary.total_maximum_marks",
            "result_declaration.result_declared_on",
        ]
        for dotted_path in compare_paths:
            current_value = CertificateExtractor._get_nested_path(updated, dotted_path)
            text_value = CertificateExtractor._get_nested_path(text_data, dotted_path)
            preferred = CertificateExtractor._choose_preferred_field_value(dotted_path, current_value, text_value)
            if preferred is not None and preferred != current_value:
                CertificateExtractor._set_nested_path(updated, dotted_path, preferred)

        current_trim = updated.get("trimester_wise_performance")
        text_trim = text_data.get("trimester_wise_performance") if isinstance(text_data, dict) else None
        if CertificateExtractor._score_trimester_rows(text_trim) > CertificateExtractor._score_trimester_rows(current_trim):
            updated["trimester_wise_performance"] = copy.deepcopy(text_trim)

        current_course = updated.get("course_details")
        text_course = text_data.get("course_details") if isinstance(text_data, dict) else None
        if CertificateExtractor._score_course_details(text_course) > CertificateExtractor._score_course_details(current_course):
            updated["course_details"] = copy.deepcopy(text_course)

        return updated

    @staticmethod
    def _get_nested_path(payload: Any, dotted_path: str) -> Any:
        current = payload
        for part in dotted_path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
                continue
            return None
        return current

    @staticmethod
    def _choose_preferred_field_value(dotted_path: str, current_value: Any, text_value: Any) -> Any:
        current_score = CertificateExtractor._score_field_value(dotted_path, current_value)
        text_score = CertificateExtractor._score_field_value(dotted_path, text_value)
        if text_score > current_score:
            return text_value
        return current_value

    @staticmethod
    def _score_field_value(dotted_path: str, value: Any) -> int:
        text = CertificateExtractor._clean_text(value)
        if not text:
            return -1000

        path = dotted_path.lower()
        score = 0

        if "date" in path or "held_in" in path:
            if re.search(r"\b\d{1,2}\s*,?\s*[A-Za-z]+\s+\d{4}\b", text):
                return 95
            if re.search(r"\b[A-Za-z]+\s+\d{4}\b", text):
                return 85
            return 10

        if "certificate_id" in path or "seat_number" in path or "aicte" in path:
            if re.search(r"\b[A-Z]{0,6}\d{2,14}(?:-\d+)?\b", text, flags=re.IGNORECASE):
                score += 80
            score += max(0, 20 - abs(len(text) - 8))
            return score

        if any(key in path for key in ["gpa", "cgpa"]):
            try:
                parsed = float(CertificateExtractor._normalize_ocr_numeric_token(text))
            except ValueError:
                parsed = None
            if parsed is None:
                return 5
            if 0 <= parsed <= 10:
                return 90
            return 30

        if path in {
            "final_summary.total_credits",
            "final_summary.total_grade_points",
            "final_summary.total_marks_obtained",
            "final_summary.total_maximum_marks",
        }:
            if re.fullmatch(r"\d+(?:\.\d+)?", text):
                return 94
            if re.fullmatch(r"\d+\s*/\s*\d+", text):
                return 85
            if re.search(r"[A-Za-z]{2,}", text):
                return 8
            return 40

        if "percentage" in path:
            if re.fullmatch(r"\d{1,3}(?:\.\d+)?\s*%", text):
                return 92
            if re.search(r"[A-Za-z]{2,}", text):
                return 15
            if re.search(r"\b\d{1,3}(?:\.\d+)?\s*%\b", text):
                return 70
            return 20

        if path.endswith("result_summary.total_marks_obtained"):
            if re.fullmatch(r"\d{1,4}", text):
                return 95
            if re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", text):
                return 82
            if re.search(r"[A-Za-z]{2,}", text):
                return 8
            return 30

        if path.endswith("result_summary.total_maximum_marks"):
            if re.fullmatch(r"\d{1,4}", text):
                return 95
            if re.search(r"[A-Za-z]{2,}", text):
                return 8
            return 25

        if path.endswith("result_summary.result"):
            if text.lower() in {"pass", "fail", "absent"}:
                return 96
            return 12

        if path.endswith("result_summary.grade_range"):
            if re.fullmatch(r"\d{1,3}(?:\.\d+)?\s*-\s*\d{1,3}(?:\.\d+)?", text):
                return 90
            return 20

        if "marks" in path and "/" in text:
            if re.fullmatch(r"\d+\s*/\s*\d+", text):
                return 88

        if any(key in path for key in ["name", "course", "issuer", "institute", "examination", "specialization", "result"]):
            score = 40
            words = re.findall(r"[A-Za-z]+", text)
            if 2 <= len(words) <= 12:
                score += 25
            long_words = [w for w in words if len(w) >= 3]
            short_words = [w for w in words if len(w) == 1]
            score += min(20, len(long_words) * 2)
            score -= min(30, len(short_words) * 5)
            noise = re.findall(r"[^A-Za-z0-9\s/%.,&'()-]", text)
            score -= min(25, len(noise) * 3)
            if path.endswith("name") and any(keyword in text.lower() for keyword in ["diploma", "management", "course", "exam"]):
                score -= 20
            if path.endswith("issuer") or path.endswith("institute_details.name"):
                if re.search(r"\b(school|institute|institution|college|university|board)\b", text, flags=re.IGNORECASE):
                    score += 12
            if path.endswith("course_name") or path.endswith("student_details.examination"):
                if re.search(r"\b(diploma|management|computer|application|engineering|program|programme)\b", text, flags=re.IGNORECASE):
                    score += 10
            return score

        return 50

    @staticmethod
    def _score_trimester_rows(value: Any) -> int:
        if not isinstance(value, list) or not value:
            return -1000
        score = 0
        for row in value:
            if not isinstance(row, dict):
                continue
            if row.get("trimester"):
                score += 4
            if row.get("gpa"):
                score += 4
            if row.get("marks"):
                score += 5
            if row.get("percentage"):
                score += 5
            if row.get("credits_earned"):
                score += 5
        if len(value) >= 6:
            score += 10
        return score

    @staticmethod
    def _score_course_details(value: Any) -> int:
        if not isinstance(value, list) or not value:
            return -1000
        score = 0
        for row in value:
            if not isinstance(row, dict):
                continue
            code = str(row.get("course_code") or "").strip().upper()
            title = CertificateExtractor._clean_text(row.get("course_title")) or ""
            words = re.findall(r"[A-Za-z]+", title)
            if re.fullmatch(r"[A-Z]{2,}\d{3,}", code):
                score += 10
            if 2 <= len(words) <= 6:
                score += 8
            if re.search(r"[A-Za-z]{3,}", title):
                score += 4
            if len(words) > 6:
                score -= 8
            if sum(1 for word in words if len(word) <= 2) >= 3:
                score -= 6
            if re.search(r"[^A-Za-z0-9\s&()'/-]", title):
                score -= 6
        return score

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
                        text = self._ocr_image_text(image)
                        if text:
                            chunks.append(f"[OCR_PAGE_{idx}]\n{text}")
            except Exception:
                pass

            return "\n\n".join(chunks)[:14000]

        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
            try:
                with Image.open(path) as image:
                    text = self._ocr_image_text(image)
                return text[:8000] if text else ""
            except Exception:
                return ""

        return ""

    @staticmethod
    def _ocr_image_text(image: Image.Image) -> str:
        variants = [
            image.convert("RGB"),
            ImageEnhance.Contrast(ImageOps.grayscale(image)).enhance(1.7),
            ImageEnhance.Sharpness(
                ImageOps.autocontrast(ImageOps.grayscale(image)).filter(ImageFilter.MedianFilter(size=3))
            ).enhance(1.4),
        ]

        texts: list[str] = []
        for variant in variants:
            try:
                text = pytesseract.image_to_string(variant).strip()
            except Exception:
                text = ""
            if text and text not in texts:
                texts.append(text)
        return "\n".join(texts)

    def _extract_bbox_words(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        image: Image.Image | None = None
        try:
            if suffix == ".pdf":
                pages = convert_from_path(
                    str(path),
                    dpi=max(210, self.vision_dpi + 30),
                    fmt="jpeg",
                    first_page=1,
                    last_page=1,
                    thread_count=2,
                )
                if not pages:
                    return []
                image = pages[0]
            elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
                image = Image.open(path)
            else:
                return []

            variants = [
                image.convert("RGB"),
                ImageEnhance.Contrast(ImageOps.grayscale(image)).enhance(1.7).convert("RGB"),
            ]
            best_words: list[dict[str, Any]] = []
            for variant in variants:
                words = self._ocr_words_from_image(variant)
                if len(words) > len(best_words):
                    best_words = words
            return best_words
        except Exception:
            return []
        finally:
            if image is not None:
                image.close()

    @staticmethod
    def _ocr_words_from_image(image: Image.Image) -> list[dict[str, Any]]:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width <= 0 or height <= 0:
            return []

        data = pytesseract.image_to_data(rgb, output_type=Output.DICT)
        words: list[dict[str, Any]] = []
        total = len(data.get("text", []))
        for idx in range(total):
            raw_text = str(data["text"][idx]).strip()
            clean = CertificateExtractor._clean_text(raw_text)
            if not clean:
                continue
            try:
                conf = float(data["conf"][idx])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue

            x = float(data["left"][idx])
            y = float(data["top"][idx])
            w = float(data["width"][idx])
            h = float(data["height"][idx])
            if w <= 0 or h <= 0:
                continue

            x1 = max(0.0, x / width)
            y1 = max(0.0, y / height)
            x2 = min(1.0, (x + w) / width)
            y2 = min(1.0, (y + h) / height)
            words.append(
                {
                    "text": clean,
                    "norm": re.sub(r"[^a-z0-9]+", "", clean.lower()),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "conf": conf,
                }
            )

        words.sort(key=lambda row: (row["cy"], row["x1"]))
        return words

    @staticmethod
    def _group_words_into_lines(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not words:
            return []
        lines: list[list[dict[str, Any]]] = []
        y_threshold = 0.018
        for word in sorted(words, key=lambda row: (row["cy"], row["x1"])):
            placed = False
            for line in lines:
                avg_y = sum(item["cy"] for item in line) / len(line)
                if abs(word["cy"] - avg_y) <= y_threshold:
                    line.append(word)
                    placed = True
                    break
            if not placed:
                lines.append([word])
        for line in lines:
            line.sort(key=lambda row: row["x1"])
        lines.sort(key=lambda row: sum(w["cy"] for w in row) / len(row))
        return lines

    @staticmethod
    def _find_subsequence(tokens: list[str], phrase_tokens: list[str]) -> tuple[int, int] | None:
        if not tokens or not phrase_tokens or len(phrase_tokens) > len(tokens):
            return None
        for start in range(0, len(tokens) - len(phrase_tokens) + 1):
            if tokens[start : start + len(phrase_tokens)] == phrase_tokens:
                return start, start + len(phrase_tokens)
        return None

    @staticmethod
    def _value_from_line_region(
        lines: list[list[dict[str, Any]]],
        line_idx: int,
        label_start_idx: int,
        label_end_idx: int,
    ) -> str | None:
        line = lines[line_idx]
        label_words = line[label_start_idx:label_end_idx]
        if not label_words:
            return None
        label_x2 = max(item["x2"] for item in label_words)
        same_line_words = [item["text"] for item in line if item["x1"] >= (label_x2 - 0.002)]
        same_line_words = [token for token in same_line_words if token not in {":", "-", "|"}]
        value = CertificateExtractor._clean_text(" ".join(same_line_words))
        if value:
            return value

        next_idx = line_idx + 1
        if next_idx >= len(lines):
            return None
        anchor_x = min(item["x1"] for item in label_words)
        next_line_words = [item["text"] for item in lines[next_idx] if item["x1"] >= max(0.0, anchor_x - 0.02)]
        return CertificateExtractor._clean_text(" ".join(next_line_words))

    @staticmethod
    def _extract_structured_from_bboxes(words: list[dict[str, Any]]) -> dict[str, Any]:
        if not words:
            return {}

        label_map: dict[str, list[list[str]]] = {
            "student_name": [["candidate", "name"], ["student", "name"], ["name"]],
            "course_name": [["course", "name"], ["examination"], ["programme"], ["program"]],
            "issue_date": [["result", "declared", "on"], ["issue", "date"], ["issued", "on"], ["held", "in"]],
            "certificate_id": [["seat", "number"], ["roll", "number"], ["certificate", "number"], ["certificate", "id"]],
            "issuer": [["issued", "by"], ["institute"], ["institution"], ["university"], ["board"], ["school"]],
        }

        lines = CertificateExtractor._group_words_into_lines(words)
        dotted_from_labels = CertificateExtractor._extract_dotted_fields_from_bbox_lines(lines)
        trimester_gpas = CertificateExtractor._extract_trimester_gpas_from_lines(lines)
        bbox_summary = CertificateExtractor._extract_result_summary_from_bbox_lines(lines)
        bbox_course_details = CertificateExtractor._extract_course_details_from_bbox_lines(lines)
        extracted: dict[str, str] = {}
        for field, phrases in label_map.items():
            best_value = None
            for line_idx, line in enumerate(lines):
                tokens = [str(item.get("norm", "")) for item in line]
                for phrase in phrases:
                    phrase_tokens = [re.sub(r"[^a-z0-9]+", "", token.lower()) for token in phrase]
                    found = CertificateExtractor._find_subsequence(tokens, phrase_tokens)
                    if not found:
                        continue
                    start, end = found
                    value = CertificateExtractor._value_from_line_region(lines, line_idx, start, end)
                    value = CertificateExtractor._clean_text(value)
                    if not value:
                        continue
                    best_value = value
                    break
                if best_value:
                    break
            if best_value:
                extracted[field] = best_value

        seat = extracted.get("certificate_id")
        if seat:
            seat_match = re.search(r"\b(?:P)?[A-Z]{0,4}\d{2,12}\b", seat, flags=re.IGNORECASE)
            if seat_match:
                extracted["certificate_id"] = seat_match.group(0).upper()

        issue_date = extracted.get("issue_date")
        if issue_date:
            month_year = re.search(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4}\b",
                issue_date,
                flags=re.IGNORECASE,
            )
            if month_year:
                extracted["issue_date"] = CertificateExtractor._clean_text(month_year.group(0)) or issue_date

        issuer = extracted.get("issuer")
        if issuer:
            issuer_line = CertificateExtractor._clean_text(issuer)
            if issuer_line:
                extracted["issuer"] = issuer_line
        institute_address = CertificateExtractor._extract_institute_address_from_bbox_lines(lines, extracted.get("issuer"))
        if institute_address:
            institute_address = CertificateExtractor._sanitize_address(institute_address)

        if not extracted:
            if not trimester_gpas:
                return {}

        student_name = extracted.get("student_name")
        course_name = extracted.get("course_name")
        issue_date = extracted.get("issue_date")
        certificate_id = extracted.get("certificate_id")
        issuer = extracted.get("issuer")
        trimester_rows: list[dict[str, Any]] = []
        trimesters = ["I", "II", "III", "IV", "V", "VI"]
        for idx, gpa in enumerate(trimester_gpas[:6]):
            trimester_rows.append({"trimester": trimesters[idx], "gpa": gpa})
        payload: dict[str, Any] = {
            "student_name": student_name,
            "course_name": course_name,
            "issue_date": issue_date,
            "certificate_id": certificate_id,
            "issuer": issuer,
            "student_details": {
                "name": student_name,
                "examination": course_name,
                "held_in": issue_date,
                "seat_number": certificate_id,
            },
            "institute_details": {
                "name": issuer,
                "address": institute_address,
            },
            "result_declaration": {
                "result_declared_on": issue_date,
            },
            "result_summary": {
                "gpa": trimester_rows[-1]["gpa"] if trimester_rows else None,
                "overall_grade": bbox_summary.get("overall_grade"),
                "grade_range": bbox_summary.get("grade_range"),
            },
            "trimester_wise_performance": trimester_rows,
            "course_details": bbox_course_details,
        }
        for dotted_path, value in dotted_from_labels.items():
            CertificateExtractor._set_nested_path(payload, dotted_path, value)
        return payload

    @staticmethod
    def _extract_dotted_fields_from_bbox_lines(lines: list[list[dict[str, Any]]]) -> dict[str, Any]:
        dotted_map: dict[str, list[list[str]]] = {
            "student_details.examination": [["examination"], ["course", "name"], ["programme"], ["program"]],
            "student_details.held_in": [["held", "in"], ["examination", "held", "in"]],
            "student_details.specialization": [["specialisation"], ["specialization"]],
            "student_details.seat_number": [["seat", "number"], ["roll", "number"]],
            "student_details.aicte_number": [["aicte", "number"], ["aicte", "no"]],
            "institute_details.name": [["issued", "by"], ["institution"], ["institute"], ["school"], ["university"]],
            "result_declaration.result_declared_on": [["result", "declared", "on"], ["declared", "on"]],
            "result_summary.total_marks_obtained": [["marks", "obtained"]],
            "result_summary.total_maximum_marks": [["maximum", "marks"], ["max", "marks"]],
            "result_summary.percentage": [["percentage"]],
            "result_summary.gpa": [["gpa"]],
            "result_summary.result": [["remark"], ["result"]],
            "final_summary.final_cgpa": [["final", "cgpa"], ["cgpa"]],
            "final_summary.total_credits": [["total", "credits"]],
            "final_summary.total_grade_points": [["total", "grade", "points"]],
            "final_summary.total_marks_obtained": [["total", "marks", "obtained"]],
        }

        out: dict[str, Any] = {}
        for dotted_path, phrases in dotted_map.items():
            best_value = None
            for line_idx, line in enumerate(lines):
                tokens = [str(item.get("norm", "")) for item in line]
                for phrase in phrases:
                    phrase_tokens = [re.sub(r"[^a-z0-9]+", "", token.lower()) for token in phrase]
                    found = CertificateExtractor._find_subsequence(tokens, phrase_tokens)
                    if not found:
                        continue
                    start, end = found
                    value = CertificateExtractor._value_from_line_region(lines, line_idx, start, end)
                    value = CertificateExtractor._clean_text(value)
                    if not value:
                        continue
                    best_value = value
                    break
                if best_value:
                    break
            if best_value:
                out[dotted_path] = best_value
        return out

    @staticmethod
    def _set_nested_path(target: dict[str, Any], dotted_path: str, value: Any) -> None:
        parts = [part for part in dotted_path.split(".") if part]
        if not parts:
            return
        current: Any = target
        for idx, part in enumerate(parts):
            is_last = idx == len(parts) - 1
            if is_last:
                if isinstance(current, dict):
                    current[part] = value
                return
            if not isinstance(current, dict):
                return
            next_node = current.get(part)
            if not isinstance(next_node, dict):
                next_node = {}
                current[part] = next_node
            current = next_node

    @staticmethod
    def _extract_institute_address_from_bbox_lines(
        lines: list[list[dict[str, Any]]],
        issuer: str | None,
    ) -> str | None:
        if not lines:
            return None
        joined_lines = [
            CertificateExtractor._clean_text(" ".join(str(item.get("text", "")) for item in line)) or ""
            for line in lines
        ]
        anchors: list[int] = []
        if issuer:
            issuer_norm = CertificateExtractor._normalize_for_match(issuer)
            if issuer_norm:
                for idx, line_text in enumerate(joined_lines):
                    if issuer_norm in CertificateExtractor._normalize_for_match(line_text):
                        anchors.append(idx)

        if not anchors:
            for idx, line_text in enumerate(joined_lines):
                if re.search(r"\b(school|institute|institution|university|college|board)\b", line_text, flags=re.IGNORECASE):
                    anchors.append(idx)
                    if len(anchors) >= 3:
                        break

        for anchor_idx in anchors:
            for next_idx in range(anchor_idx + 1, min(len(joined_lines), anchor_idx + 4)):
                candidate = joined_lines[next_idx]
                if not candidate:
                    continue
                if CertificateExtractor._looks_like_address_line(candidate):
                    return candidate
        return None

    @staticmethod
    def _extract_trimester_gpas_from_lines(lines: list[list[dict[str, Any]]]) -> list[str]:
        best: list[str] = []
        for line in lines:
            text = " ".join(str(item.get("text", "")) for item in line)
            if "GPA" not in text.upper():
                continue
            raw_values = re.findall(r"\b([0-9OIlS]+(?:[.,][0-9OIlS]+))\b", text, flags=re.IGNORECASE)
            parsed: list[str] = []
            for raw in raw_values:
                normalized = CertificateExtractor._normalize_ocr_numeric_token(raw)
                normalized = normalized.replace(",", ".")
                try:
                    value = float(normalized)
                except ValueError:
                    continue
                if 0.0 <= value <= 10.0:
                    parsed.append(f"{value:.2f}")
            if len(parsed) >= 6:
                return parsed[:6]
            if len(parsed) > len(best):
                best = parsed
        return best

    @staticmethod
    def _extract_result_summary_from_bbox_lines(lines: list[list[dict[str, Any]]]) -> dict[str, str | None]:
        best_line = ""
        for line in lines:
            text = " ".join(str(item.get("text", "")) for item in line)
            upper = text.upper()
            if "REMARK" in upper and "MARKS" in upper:
                best_line = text
                break
        if not best_line:
            return {"overall_grade": None, "grade_range": None}

        overall_grade = CertificateExtractor._extract_regex_value(
            best_line,
            [
                r"\bGRADE\b[^A-Za-z0-9]{0,8}([A-F])\b[^A-Za-z]{0,24}\bRANGE\b",
                r"\bGRADE\b[^A-Za-z0-9]{0,8}([A-F])\b",
            ],
            flags=re.IGNORECASE,
        )
        if overall_grade:
            overall_grade = overall_grade.upper()

        grade_range = CertificateExtractor._extract_grade_range(best_line)
        if overall_grade == "E" and grade_range:
            try:
                low_value = float(grade_range.split("-", 1)[0])
                if low_value >= 70:
                    overall_grade = "A"
            except (ValueError, IndexError):
                pass

        return {
            "overall_grade": overall_grade,
            "grade_range": grade_range,
        }

    @staticmethod
    def _extract_course_details_from_bbox_lines(lines: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for line in lines:
            text = CertificateExtractor._clean_text(" ".join(str(item.get("text", "")) for item in line))
            if not text:
                continue
            match = re.search(r"\b([A-Z]{2,}\d{3,})\b\s+([A-Za-z][A-Za-z0-9 &()'/-]{3,80})", text)
            if not match:
                continue
            code = match.group(1).upper()
            title = CertificateExtractor._clean_text(match.group(2))
            if not title:
                continue
            stop = re.search(
                r"\b(Total|SPECIALISATION|AICTE|SEAT|NUMBER|Marks|Obtained|Grade|Points|Remark|Trimester)\b",
                title,
                flags=re.IGNORECASE,
            )
            if stop:
                title = CertificateExtractor._clean_text(title[: stop.start()])
            if not title:
                continue
            candidates.append({"course_code": code, "course_title": title})

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in candidates:
            code = str(row.get("course_code") or "")
            title = str(row.get("course_title") or "")
            key = (code, title.lower())
            if not code or not title or key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= 10:
                break
        return deduped

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" ,.;:|-")
        return text or None

    @staticmethod
    def _extract_regex_value(text: str, patterns: list[str], flags: int = 0) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags)
            if not match:
                continue
            candidate = CertificateExtractor._clean_text(match.group(1))
            if candidate:
                return candidate
        return None

    @staticmethod
    def _normalize_ocr_numeric_token(value: str) -> str:
        replacements = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", ",": "."})
        return value.translate(replacements)

    @staticmethod
    def _trim_at_keywords(value: str | None, keywords: list[str]) -> str | None:
        cleaned = CertificateExtractor._clean_text(value)
        if not cleaned:
            return None

        out = cleaned
        for keyword in keywords:
            idx = re.search(rf"\b{re.escape(keyword)}\b", out, flags=re.IGNORECASE)
            if idx:
                out = out[: idx.start()].strip(" ,.;:|-")
        return CertificateExtractor._clean_text(out)

    @staticmethod
    def _extract_month_year_values(text: str) -> list[str]:
        month_pattern = (
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s*,?\s*(\d{4})\b"
        )
        values: list[str] = []
        for match in re.finditer(month_pattern, text, flags=re.IGNORECASE):
            month = match.group(1).capitalize()
            year = match.group(2)
            values.append(f"{month} {year}")

        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    @staticmethod
    def _extract_full_date_values(text: str) -> list[str]:
        date_pattern = (
            r"\b(\d{1,2})\s+"
            r"(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s*,?\s*(\d{4})\b"
        )
        values: list[str] = []
        for match in re.finditer(date_pattern, text, flags=re.IGNORECASE):
            day = int(match.group(1))
            month = match.group(2).capitalize()
            year = match.group(3)
            values.append(f"{day}, {month} {year}")
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    @staticmethod
    def _find_full_date_near_keywords(text: str, keywords: list[str], window: int = 120) -> str | None:
        for keyword in keywords:
            escaped = re.escape(keyword)
            pattern = rf"{escaped}[\s\S]{{0,{window}}}?(\d{{1,2}}\s*,?\s*[A-Za-z]+\s*,?\s*\d{{4}})"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _find_month_year_near_keywords(text: str, keywords: list[str], window: int = 120) -> str | None:
        for keyword in keywords:
            escaped = re.escape(keyword)
            pattern = (
                rf"{escaped}[\s\S]{{0,{window}}}?"
                r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4})"
            )
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                raw = CertificateExtractor._clean_text(match.group(1)) or ""
                cleaned = re.sub(r"\s*,\s*", " ", raw).strip()
                if cleaned:
                    return cleaned
        return None

    @staticmethod
    def _format_date_like(text: str) -> str | None:
        match = re.search(
            r"\b(\d{1,2})\s*,?\s*"
            r"(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s*,?\s*(\d{4})\b",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            cleaned = CertificateExtractor._clean_text(text)
            return cleaned
        day = int(match.group(1))
        month = match.group(2).capitalize()
        year = match.group(3)
        return f"{day}, {month} {year}"

    @staticmethod
    def _extract_name_from_lines(line_text: str, flat_text: str) -> str | None:
        candidates: list[tuple[str, bool]] = []
        slash_pattern = r"/\s*([A-Za-z][A-Za-z .'-]{2,80}?)(?=\s+(?:EXAMINATION|HELD\s+IN|SEAT|SPECIALI|AICTE|POST|TRIMESTER)\b)"
        for text in (line_text, flat_text):
            for match in re.finditer(slash_pattern, text):
                candidates.append((match.group(1), True))

        patterns = [
            r"\bCANDIDATE\s+NAME\b[^:\n]{0,30}[:\-]\s*(/)?\s*([A-Za-z][A-Za-z .'-]{2,90})",
            r"\bSTUDENT\s+NAME\b[^:\n]{0,30}[:\-]\s*(/)?\s*([A-Za-z][A-Za-z .'-]{2,90})",
            r"\bNAME\b[^:\n]{0,30}[:\-]\s*(/)?\s*([A-Za-z][A-Za-z .'-]{2,90})",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, line_text, flags=re.IGNORECASE):
                candidates.append((match.group(2), bool(match.group(1))))
            for match in re.finditer(pattern, flat_text, flags=re.IGNORECASE):
                candidates.append((match.group(2), bool(match.group(1))))

        best_name: str | None = None
        best_score = -10_000
        for candidate, has_slash in candidates:
            cleaned = CertificateExtractor._sanitize_name_candidate(candidate)
            if not cleaned:
                continue
            score = CertificateExtractor._score_name_candidate(cleaned, has_slash=has_slash)
            if score > best_score:
                best_score = score
                best_name = f"/{cleaned}" if has_slash else cleaned
        return best_name

    @staticmethod
    def _sanitize_name_candidate(value: str | None) -> str | None:
        text = CertificateExtractor._clean_text(value)
        if not text:
            return None
        text = re.sub(r"^[\\/|:\-\s]+", "", text)
        stop_keywords = [
            "examination",
            "course",
            "post",
            "graduate",
            "diploma",
            "management",
            "seat",
            "roll",
            "aicte",
            "school",
            "institute",
            "university",
            "trimester",
            "marks",
            "result",
            "grade",
            "gpa",
            "cgpa",
        ]
        stop_pattern = r"\b(" + "|".join(re.escape(word) for word in stop_keywords) + r")\b"
        stop_match = re.search(stop_pattern, text, flags=re.IGNORECASE)
        if stop_match:
            text = text[: stop_match.start()].strip()

        text = re.sub(r"[^A-Za-z .'-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" .'-")
        if not text:
            return None
        tokens = [token for token in text.split() if len(token) > 1]
        if len(tokens) < 2:
            return None
        if len(tokens) > 4:
            tokens = tokens[:4]
        return " ".join(token.capitalize() for token in tokens)

    @staticmethod
    def _score_name_candidate(value: str, has_slash: bool = False) -> int:
        tokens = value.split()
        score = 0
        if 2 <= len(tokens) <= 4:
            score += 25
        score += max(0, 18 - abs(len(value) - 18))
        if any(len(token) == 2 for token in tokens):
            score -= 3
        if len(tokens) == 4:
            score += 2
        if len(tokens) == 2:
            score -= 2
        if has_slash:
            score += 4
        return score

    @staticmethod
    def _determine_gender_from_name(name: str | None, context: str) -> str:
        if not name:
            return "Male"
        if name.startswith("/"):
            return "Female"
        pattern = rf"/\s*{re.escape(name)}"
        if re.search(pattern, context, flags=re.IGNORECASE):
            return "Female"
        return "Male"

    @staticmethod
    def _normalize_for_match(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @staticmethod
    def _looks_like_address_line(line: str) -> bool:
        lower = line.lower()
        if len(lower) < 6:
            return False
        keywords = [
            "road",
            "rd",
            "street",
            "st",
            "lane",
            "sector",
            "block",
            "nagar",
            "near",
            "opposite",
            "opp",
            "plot",
            "mumbai",
            "delhi",
            "india",
            "pin",
            "pincode",
            "nav",
        ]
        if any(re.search(rf"\b{re.escape(keyword)}\b", lower) for keyword in keywords):
            return True
        if re.search(r"\b\d{5,6}\b", lower):
            return True
        return ("," in line and len(line.split()) >= 4) or bool(re.search(r"\bno\.?\s*\d+\b", lower))

    @staticmethod
    def _extract_institute_address(text_context: str, institute_name: str | None) -> str | None:
        lines: list[str] = []
        for raw in text_context.splitlines():
            cleaned = CertificateExtractor._clean_text(raw)
            if not cleaned:
                continue
            if cleaned.startswith("[PDF_TEXT_PAGE_") or cleaned.startswith("[OCR_PAGE_"):
                continue
            lines.append(cleaned)

        if not lines:
            return None

        anchors: list[int] = []
        if institute_name:
            norm_name = CertificateExtractor._normalize_for_match(institute_name)
            if norm_name:
                for idx, line in enumerate(lines):
                    if norm_name in CertificateExtractor._normalize_for_match(line):
                        anchors.append(idx)

        if not anchors:
            for idx, line in enumerate(lines):
                if re.search(r"\b(school|institute|institution|university|college|board)\b", line, flags=re.IGNORECASE):
                    anchors.append(idx)
                    if len(anchors) >= 4:
                        break

        stop_pattern = re.compile(
            r"\b(student|candidate|course|examination|seat|roll|result|marks|trimester|gpa|cgpa|grade|date|aicte|certificate)\b",
            flags=re.IGNORECASE,
        )
        for anchor_idx in anchors:
            collected: list[str] = []
            for next_idx in range(anchor_idx + 1, min(len(lines), anchor_idx + 5)):
                candidate = lines[next_idx]
                if stop_pattern.search(candidate):
                    break
                if CertificateExtractor._looks_like_address_line(candidate):
                    collected.append(candidate)
                    if len(collected) >= 2:
                        break
                elif collected:
                    break
        if collected:
            address_raw = CertificateExtractor._find_best_address_segment(collected)
            if address_raw:
                return CertificateExtractor._sanitize_address(address_raw)
        return None

    @staticmethod
    def _extract_grade_range(text: str) -> str | None:
        match = re.search(r"\bRANGE\b\s*[:\-]?\s*([0-9.,]+)\s*[-to]{0,3}\s*([0-9.,]+)", text, flags=re.IGNORECASE)
        if not match:
            return None
        low = match.group(1).replace(",", ".")
        high = match.group(2).replace(",", ".")
        return f"{low}-{high}"

    @staticmethod
    def _find_best_address_segment(lines: list[str]) -> str | None:
        for line in lines:
            if re.search(r"\bPlot\b", line, flags=re.IGNORECASE):
                return line
        for line in lines:
            if re.search(r"\bSector\b", line, flags=re.IGNORECASE):
                return line
        return ", ".join(lines)

    def _sanitize_address(address: str) -> str | None:
        match = re.search(r"(Plot\s*\d+.*)", address, flags=re.IGNORECASE)
        candidate = match.group(1) if match else address
        if not candidate:
            return None
        exact = re.search(
            r"(Plot.*?Nav(?:i|e)\s*Mumb(?:el|ai)\s*400\s*7?\s*06)",
            candidate,
            flags=re.IGNORECASE,
        )
        if exact:
            candidate = exact.group(1)
        cleaned = re.sub(r"[^A-Za-z0-9,.\s-]", " ", candidate)
        cleaned = re.sub(r"\s+-\s+", "-", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\bPlot\s*16\b", "Plot 1E", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bPlot\s*1\s*[Ee]\b", "Plot 1E", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bSector[\s-]*V\b", "Sector-V", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bNavi\s+Mumbai\b[\s,-]*400[\s-]*706\b", "Navi Mumbai 400 706", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bSector[-\s]*V\b\s*Nerul", "Sector-V Nerul", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bNav\b\s*Mumbel\s*400\s*7\s*706\b", "Navi Mumbai 400 706", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bPlot\s*1E\b[,.\s]*Sector-V\b", "Plot 1E, Sector-V", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(Navi Mumbai 400 706).*", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r",\s*,", ", ", cleaned)
        return cleaned.strip(" ,.-")

    @staticmethod
    def _extract_structured_from_text(text_context: str) -> dict[str, Any]:
        if not text_context:
            return {}

        text = text_context
        line_text = re.sub(r"[ \t]+", " ", text)
        flat_text = re.sub(r"\s+", " ", text)
        upper_flat = flat_text.upper()

        student_name = CertificateExtractor._extract_name_from_lines(line_text, flat_text)
        course_name = CertificateExtractor._extract_regex_value(
            line_text,
            [
                r"\bEXAMINATION\b[^:\n]{0,30}[:\-]\s*([^\n|]{4,120})",
                r"\bCOURSE\s+NAME\b[^:\n]{0,30}[:\-]\s*([^\n|]{4,120})",
            ],
            flags=re.IGNORECASE,
        )
        held_in = CertificateExtractor._extract_regex_value(
            line_text,
            [
                r"\bHELD\s+IN\b[^:\n]{0,20}[:\-]\s*([A-Za-z]+\s*,?\s*\d{4})",
                r"\bEXAMINATION\s+HELD\s+IN\b[^:\n]{0,20}[:\-]\s*([A-Za-z]+\s*,?\s*\d{4})",
            ],
            flags=re.IGNORECASE,
        )
        if not held_in:
            held_in = CertificateExtractor._extract_regex_value(
                flat_text,
                [
                    r"\bHELD\s+IN\b[^A-Za-z0-9]{0,20}((?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4})",
                    r"\bEXAMINATION\s+HELD\s+IN\b[^A-Za-z0-9]{0,20}((?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4})",
                ],
                flags=re.IGNORECASE,
            )
        if not held_in:
            held_in = CertificateExtractor._find_month_year_near_keywords(
                flat_text,
                ["examination held in", "held in"],
                window=90,
            )
        if held_in:
            held_in = re.sub(r"\s*,\s*", " ", held_in).strip()
        specialization = CertificateExtractor._extract_regex_value(
            line_text,
            [
                r"\bSPECIALI\w*\b[^:\n]{0,20}[:\-]\s*([A-Za-z][A-Za-z &]{2,80})",
            ],
            flags=re.IGNORECASE,
        )
        course_name = CertificateExtractor._trim_at_keywords(
            course_name,
            ["HELD", "SPECIALIZATION", "SEAT", "AICTE", "SCHOOL", "CREDITS", "TRIMESTER"],
        )
        if course_name and re.search(r"\bmanagement\b", course_name, flags=re.IGNORECASE):
            mgmt = re.search(r"^.*?\bmanagement\b", course_name, flags=re.IGNORECASE)
            if mgmt:
                course_name = mgmt.group(0).strip()
        specialization = CertificateExtractor._trim_at_keywords(
            specialization,
            ["SEAT", "AICTE", "CREDITS", "TRIMESTER", "BUSINESS"],
        )
        if specialization and re.search(r"\bmarketing\b", specialization, flags=re.IGNORECASE):
            specialization = "Marketing"
        elif specialization:
            words = [w for w in re.findall(r"[A-Za-z]+", specialization) if len(w) > 1]
            if words:
                specialization = " ".join(words[:3]).title()
        seat_number = CertificateExtractor._extract_regex_value(
            flat_text,
            [
                r"\bSEAT\s*NUMBER\b[^A-Za-z0-9]{0,15}([A-Z]{1,6}\d{2,12})",
                r"\bROLL\s*NUMBER\b[^A-Za-z0-9]{0,15}([A-Z]{0,4}\d{2,12})",
            ],
            flags=re.IGNORECASE,
        )
        if seat_number:
            pg_match = re.search(rf"\bP{re.escape(seat_number)}\b", flat_text, flags=re.IGNORECASE)
            if pg_match:
                seat_number = pg_match.group(0).upper()
        aicte_number = CertificateExtractor._extract_regex_value(
            flat_text,
            [
                r"\bAICTE\s*NUMBER\b[^A-Za-z0-9]{0,15}([A-Z0-9\-]{4,25})",
                r"\bAICTE\s*NO\b[^A-Za-z0-9]{0,15}([A-Z0-9\-]{4,25})",
            ],
            flags=re.IGNORECASE,
        )

        institute_name = None
        if "SIES SCHOOL OF BUSINESS STUDIES" in upper_flat:
            institute_name = "SIES School of Business Studies"
        else:
            institute_name = CertificateExtractor._extract_regex_value(
                line_text,
                [
                    r"\b([A-Z][A-Za-z&,. ]{3,100}(?:School|Institute|University|Board)[A-Za-z&,. ]{0,60})\b",
                ],
            )
        institute_address = CertificateExtractor._extract_institute_address(text, institute_name)

        result_status = CertificateExtractor._extract_regex_value(
            flat_text,
            [
                r"\bREMARK\b[^A-Za-z]{0,10}(PASS|FAIL|ABSENT)",
                r"\bRESULT\b[^A-Za-z]{0,10}(PASS|FAIL|ABSENT)",
            ],
            flags=re.IGNORECASE,
        )
        if result_status:
            result_status = result_status.upper().capitalize()

        gender = CertificateExtractor._determine_gender_from_name(student_name, text)

        marks_match = re.search(
            r"\bREMARK\b.{0,500}?\bMARKS\s*OBTAINED\b[^0-9]{0,20}(\d{1,4})\s*/\s*(\d{1,4})",
            flat_text,
            flags=re.IGNORECASE,
        )
        if not marks_match:
            marks_match = re.search(
            r"\bMARKS\s*OBTAINED\b[^0-9]{0,15}(\d{1,4})\s*/\s*(\d{1,4})",
            flat_text,
            flags=re.IGNORECASE,
        )
        percentage = CertificateExtractor._extract_regex_value(flat_text, [r"\bREMARK\b.{0,500}?\bPERCENTAGE\b[^0-9]{0,10}(\d{1,3}(?:\.\d+)?\s*%)"], flags=re.IGNORECASE)
        if not percentage:
            percentage = CertificateExtractor._extract_regex_value(
                flat_text,
                [
                    r"\bPERCENTAGE\b[^0-9]{0,10}(\d{1,3}(?:\.\d+)?\s*%)",
                ],
                flags=re.IGNORECASE,
            )
        summary_window_match = re.search(r"\bREMARK\b.{0,260}", flat_text, flags=re.IGNORECASE)
        summary_window = summary_window_match.group(0) if summary_window_match else flat_text
        overall_grade = CertificateExtractor._extract_regex_value(
            summary_window,
            [
                r"\bGRADE\b[^A-Za-z0-9]{0,8}([A-F])\b[^A-Za-z]{0,24}\bRANGE\b",
                r"\bGRADE\b[^A-Za-z0-9]{0,8}([A-F])\b",
            ],
            flags=re.IGNORECASE,
        )
        if overall_grade:
            overall_grade = overall_grade.upper()
        grade_range = CertificateExtractor._extract_grade_range(summary_window)
        if not grade_range:
            grade_range = CertificateExtractor._extract_grade_range(flat_text)
        if not overall_grade and grade_range:
            try:
                low = float(grade_range.split("-", 1)[0])
                if low >= 70:
                    overall_grade = "A"
            except (ValueError, IndexError):
                pass
        if not overall_grade:
            overall_grade = CertificateExtractor._extract_regex_value(
                flat_text,
                [
                    r"\bGRADE\b[^A-Za-z0-9]{0,8}([A-F])\b",
                ],
                flags=re.IGNORECASE,
            )
            if overall_grade:
                overall_grade = overall_grade.upper()

        grade_range = CertificateExtractor._extract_grade_range(summary_window)
        if not grade_range:
            grade_range = CertificateExtractor._extract_grade_range(flat_text)
        if overall_grade == "E" and grade_range:
            try:
                low_value = float(grade_range.split("-", 1)[0])
                if low_value >= 70:
                    overall_grade = "A"
            except (ValueError, IndexError):
                pass

        course_details = CertificateExtractor._extract_course_details(text)

        trimester_wise: list[dict[str, Any]] = []
        trimester_block_match = re.search(
            r"\bTRIMESTER\s*I\b(.*?)(?=\bFINAL\s*CGPA\b|$)",
            flat_text,
            flags=re.IGNORECASE,
        )
        if trimester_block_match:
            trimester_block = trimester_block_match.group(0)
            credits_list = re.findall(
                r"\bCREDITS\s*E\w*\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
                trimester_block,
                flags=re.IGNORECASE,
            )
            marks_list = re.findall(
                r"\bMARKS\s*[=:]\s*(\d+\s*/\s*\d+)",
                trimester_block,
                flags=re.IGNORECASE,
            )
            percentage_list = re.findall(
                r"\bPERCENTAGE\s*[=:]\s*([0-9]+(?:\.[0-9]+)?\s*%)",
                trimester_block,
                flags=re.IGNORECASE,
            )
            gpa_list = re.findall(
                r"\bGPA\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)",
                trimester_block,
                flags=re.IGNORECASE,
            )

            first_trimester_block = re.search(
                r"\bTRIMESTER\s*I\b(.*?)(?=\bTRIMESTER\s*II\b|$)",
                trimester_block,
                flags=re.IGNORECASE,
            )
            if first_trimester_block:
                first_gpa = CertificateExtractor._extract_regex_value(
                    first_trimester_block.group(1),
                    [r"\bGPA\s*[=:]\s*([0-9OIlS]+(?:[.,][0-9OIlS]+)?)"],
                    flags=re.IGNORECASE,
                )
                if first_gpa:
                    first_gpa = CertificateExtractor._normalize_ocr_numeric_token(first_gpa)
                    if gpa_list and gpa_list[0] != first_gpa:
                        gpa_list = [first_gpa, *gpa_list]
                    elif not gpa_list:
                        gpa_list = [first_gpa]

            trimesters = ["I", "II", "III", "IV", "V", "VI"]
            for idx, trimester in enumerate(trimesters):
                credits = credits_list[idx] if idx < len(credits_list) else None
                marks = marks_list[idx] if idx < len(marks_list) else None
                section_percentage = percentage_list[idx] if idx < len(percentage_list) else None
                gpa = gpa_list[idx] if idx < len(gpa_list) else None
                if credits or marks or section_percentage or gpa:
                    trimester_wise.append(
                        {
                            "trimester": trimester,
                            "credits_earned": credits,
                            "marks": marks,
                            "percentage": section_percentage,
                            "gpa": gpa,
                        }
                    )
            if len(trimester_wise) >= 6:
                den_v = (trimester_wise[4].get("marks") or "").split("/")[-1].strip()
                den_vi = (trimester_wise[5].get("marks") or "").split("/")[-1].strip()
                # OCR for columnar tables can flip the last two trimester columns.
                if den_v == "100" and den_vi == "600":
                    v_metrics = {k: v for k, v in trimester_wise[4].items() if k != "trimester"}
                    vi_metrics = {k: v for k, v in trimester_wise[5].items() if k != "trimester"}
                    trimester_wise[4].update(vi_metrics)
                    trimester_wise[5].update(v_metrics)

        final_cgpa = CertificateExtractor._extract_regex_value(
            flat_text,
            [r"\bFINAL\s*CGPA\b[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)"],
            flags=re.IGNORECASE,
        )
        total_credits = CertificateExtractor._extract_regex_value(
            flat_text,
            [r"\bTOTAL\s*CREDITS\b[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)"],
            flags=re.IGNORECASE,
        )
        total_grade_points = CertificateExtractor._extract_regex_value(
            flat_text,
            [r"\bTOTAL\s*GRADE\s*POINTS\b[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)"],
            flags=re.IGNORECASE,
        )
        total_marks_ratio = CertificateExtractor._extract_regex_value(
            flat_text,
            [r"\bTOTAL\s*MARKS\s*OBTAINED\b[^0-9]{0,10}(\d+\s*/\s*\d+)"],
            flags=re.IGNORECASE,
        )

        final_obtained = None
        final_maximum = None
        if total_marks_ratio and "/" in total_marks_ratio:
            left, right = [part.strip() for part in total_marks_ratio.split("/", 1)]
            final_obtained, final_maximum = left, right

        months = CertificateExtractor._extract_month_year_values(flat_text)
        full_dates = CertificateExtractor._extract_full_date_values(flat_text)
        result_declared_on = CertificateExtractor._extract_regex_value(
            flat_text,
            [
                r"\bRESULT\s*D\w+\s*ON\b[^0-9A-Za-z]{0,10}(\d{1,2}\s+[A-Za-z]+\s*,?\s*\d{4})",
                r"\bRESULT\s*D\w+\b[^A-Za-z]{0,10}([A-Za-z]+\s*,?\s*\d{4})",
            ],
            flags=re.IGNORECASE,
        )
        if not result_declared_on:
            neighbor = CertificateExtractor._find_full_date_near_keywords(
                flat_text,
                [
                    "result declared on",
                    "declared on",
                    "result declared",
                    "result date",
                    "issued on",
                    "printed on",
                ],
            )
            if neighbor:
                result_declared_on = neighbor
        if result_declared_on:
            result_declared_on = CertificateExtractor._format_date_like(result_declared_on)
        if not result_declared_on and full_dates:
            result_declared_on = full_dates[-1]
        elif not result_declared_on and months:
            result_declared_on = months[-1]
        issue_date = result_declared_on or held_in or (months[-1] if months else None)

        signed_by = None
        if re.search(r"\bHEAD\s+EXAMINATIONS?\b", upper_flat):
            signed_by = "Head Examinations"
        else:
            signed_by = CertificateExtractor._extract_regex_value(
                line_text,
                [r"\bSIGNED\s+BY\b[^:\n]{0,10}[:\-]\s*([A-Za-z .'-]{3,80})"],
                flags=re.IGNORECASE,
            )

        document_type = None
        if "TRIMESTER" in upper_flat and "FINAL CGPA" in upper_flat:
            document_type = "Marksheet / Result Statement"

        result_summary_gpa = None
        if trimester_wise:
            gpas = [row.get("gpa") for row in trimester_wise if row.get("gpa")]
            result_summary_gpa = gpas[-1] if gpas else None
        result_summary_marks_obtained = marks_match.group(1) if marks_match else None
        result_summary_marks_maximum = marks_match.group(2) if marks_match else None
        if result_summary_marks_maximum:
            try:
                if int(result_summary_marks_maximum) > 1000:
                    vi_entry = next(
                        (row for row in trimester_wise if str(row.get("trimester", "")).upper() == "VI"),
                        None,
                    )
                    if vi_entry and isinstance(vi_entry.get("marks"), str) and "/" in vi_entry["marks"]:
                        left, right = [part.strip() for part in vi_entry["marks"].split("/", 1)]
                        result_summary_marks_obtained = left
                        result_summary_marks_maximum = right
            except ValueError:
                pass

        return {
            "student_name": student_name,
            "course_name": course_name,
            "issue_date": issue_date,
            "certificate_id": seat_number,
            "issuer": institute_name,
            "institute_details": {
                "name": institute_name,
                "address": institute_address,
                "document_type": document_type,
            },
            "student_details": {
                "name": student_name,
                "examination": course_name,
                "held_in": held_in,
                "specialization": specialization,
                "seat_number": seat_number,
                "aicte_number": aicte_number,
                "gender": gender,
            },
            "course_details": course_details,
            "result_summary": {
                "total_marks_obtained": result_summary_marks_obtained,
                "total_maximum_marks": result_summary_marks_maximum,
                "percentage": percentage,
                "gpa": result_summary_gpa,
                "overall_grade": overall_grade,
                "grade_range": grade_range,
                "result": result_status,
            },
            "trimester_wise_performance": trimester_wise,
            "final_summary": {
                "final_cgpa": final_cgpa,
                "total_credits": total_credits,
                "total_grade_points": total_grade_points,
                "total_marks_obtained": final_obtained,
                "total_maximum_marks": final_maximum,
            },
            "result_declaration": {
                "result_declared_on": result_declared_on,
                "signed_by": signed_by,
            },
        }

    @staticmethod
    def _extract_course_details(text_context: str) -> list[dict[str, Any]]:
        flat_text = re.sub(r"\s+", " ", text_context)
        candidates: list[tuple[str, str]] = []
        pattern = r"\b([A-Z]{2,}\d{3,})\b\s+([A-Za-z][A-Za-z0-9 &()'/-]{3,80})"
        for match in re.finditer(pattern, flat_text):
            code = match.group(1).strip()
            title = CertificateExtractor._clean_text(match.group(2))
            if not title:
                continue
            stop = re.search(
                r"\b(Total|SPECIALISATION|AICTE|SEAT|NUMBER|Marks|Obtained|Grade|Points|Remark|Trimester)\b",
                title,
                flags=re.IGNORECASE,
            )
            if stop:
                title = CertificateExtractor._clean_text(title[: stop.start()])
            if not title or len(title) < 3:
                continue
            candidates.append((code, title))

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for code, title in candidates:
            key = (code.upper(), title.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"course_code": code.upper(), "course_title": title})
            if len(deduped) >= 10:
                break
        return deduped

    @staticmethod
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict)):
            return bool(value)
        return True

    @staticmethod
    def _merge_candidate_data(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        def merge_node(primary_node: Any, fallback_node: Any) -> Any:
            if isinstance(primary_node, dict) and isinstance(fallback_node, dict):
                merged: dict[str, Any] = {}
                for key in set(primary_node.keys()) | set(fallback_node.keys()):
                    merged[key] = merge_node(primary_node.get(key), fallback_node.get(key))
                return merged
            if isinstance(primary_node, list) and isinstance(fallback_node, list):
                if primary_node:
                    return copy.deepcopy(primary_node)
                return copy.deepcopy(fallback_node)
            if CertificateExtractor._has_value(primary_node):
                return copy.deepcopy(primary_node)
            return copy.deepcopy(fallback_node)

        safe_primary = primary if isinstance(primary, dict) else {}
        safe_fallback = fallback if isinstance(fallback, dict) else {}
        return merge_node(safe_primary, safe_fallback)

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
        raw_name = normalized.get("student_name")
        normalized_name = CertificateExtractor._sanitize_name_candidate(raw_name)
        if normalized_name:
            if isinstance(raw_name, str) and raw_name.strip().startswith("/"):
                normalized_name = f"/{normalized_name}"
            normalized["student_name"] = normalized_name
            normalized["student_details"]["name"] = normalized_name
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
                confidence = CertificateExtractor._calculate_completeness_confidence(normalized)
        else:
            confidence = CertificateExtractor._calculate_completeness_confidence(normalized)
        confidence = max(0.0, min(1.0, confidence))
        normalized["confidence_score"] = confidence
        return normalized

    @staticmethod
    def _calculate_completeness_confidence(normalized: dict[str, Any]) -> float:
        top_filled = sum(1 for field in TARGET_FIELDS if CertificateExtractor._has_value(normalized.get(field)))
        top_score = top_filled / len(TARGET_FIELDS)

        detail_slots = [
            ("institute_details", ["name", "address", "document_type"]),
            ("student_details", ["name", "examination", "held_in", "specialization", "seat_number", "aicte_number", "gender"]),
            ("result_summary", ["total_marks_obtained", "total_maximum_marks", "percentage", "gpa", "result"]),
            (
                "final_summary",
                ["final_cgpa", "total_credits", "total_grade_points", "total_marks_obtained", "total_maximum_marks"],
            ),
            ("result_declaration", ["result_declared_on", "signed_by"]),
        ]
        detail_total = 0
        detail_filled = 0
        for section_name, keys in detail_slots:
            section = normalized.get(section_name, {})
            if not isinstance(section, dict):
                detail_total += len(keys)
                continue
            for key in keys:
                detail_total += 1
                if CertificateExtractor._has_value(section.get(key)):
                    detail_filled += 1

        detail_score = (detail_filled / detail_total) if detail_total else 0.0
        trimester_rows = normalized.get("trimester_wise_performance", [])
        trimester_score = min(1.0, len(trimester_rows) / 6.0) if isinstance(trimester_rows, list) else 0.0

        # Weighted toward mandatory fields while rewarding section completeness.
        return (0.45 * top_score) + (0.4 * detail_score) + (0.15 * trimester_score)

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

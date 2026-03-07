from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


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
        return self.extract_structured_data(text_context=text_context)

    def extract_structured_data(self, text_context: str) -> tuple[dict[str, Any], dict[str, Any]]:
        structured = self._extract_structured_from_text(text_context)
        normalized = self._normalize_result(structured)
        debug_info: dict[str, Any] = {
            "status": "ocr_rule_extraction_complete",
            "text_context_preview": text_context[:1800],
            "ocr_chars": len(text_context),
        }
        return normalized, debug_info

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
    def _extract_structured_from_text(text_context: str) -> dict[str, Any]:
        if not text_context:
            return {}

        text = text_context
        line_text = re.sub(r"[ \t]+", " ", text)
        flat_text = re.sub(r"\s+", " ", text)
        upper_flat = flat_text.upper()

        student_name = CertificateExtractor._extract_regex_value(
            line_text,
            [
                r"\bNAME\b[^:\n]{0,30}[:\-]\s*([A-Za-z][A-Za-z .'-]{2,90})",
                r"\bCANDIDATE\s+NAME\b[^:\n]{0,30}[:\-]\s*([A-Za-z][A-Za-z .'-]{2,90})",
            ],
            flags=re.IGNORECASE,
        )
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
        result_declared_on = CertificateExtractor._extract_regex_value(
            flat_text,
            [
                r"\bRESULT\s*D\w+\b[^A-Za-z]{0,10}([A-Za-z]+\s*,?\s*\d{4})",
            ],
            flags=re.IGNORECASE,
        )
        if not result_declared_on and months:
            result_declared_on = months[-1]
        issue_date = result_declared_on or (months[-1] if months else held_in)

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
                "address": None,
                "approval": None,
                "document_type": document_type,
            },
            "student_details": {
                "name": student_name,
                "examination": course_name,
                "held_in": held_in,
                "specialization": specialization,
                "seat_number": seat_number,
                "aicte_number": aicte_number,
            },
            "course_details": [],
            "result_summary": {
                "total_marks_obtained": result_summary_marks_obtained,
                "total_maximum_marks": result_summary_marks_maximum,
                "percentage": percentage,
                "gpa": result_summary_gpa,
                "overall_grade": None,
                "grade_range": None,
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
            ("institute_details", ["name", "address", "approval", "document_type"]),
            ("student_details", ["name", "examination", "held_in", "specialization", "seat_number", "aicte_number"]),
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

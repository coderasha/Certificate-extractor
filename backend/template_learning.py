from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageOps
from pytesseract import Output


LEARNABLE_FIELDS = [
    "name",
    "examination",
    "held_in",
    "seat_number",
    "specialisation",
    "aicte_number",
    "course_code",
    "course_title",
    "maximum_marks",
    "minimum_marks",
    "marks_obtained",
    "course_credits",
    "grade",
    "credits_earned_c",
    "grade_points_g",
    "cxg",
    "remark",
    "percentage",
    "gpa",
    "overall_grade",
    "range",
    "trimester_i",
    "trimester_ii",
    "trimester_iii",
    "trimester_iv",
    "trimester_v",
    "trimester_vi",
    "final_cgpa",
    "total_credits",
    "total_grade_points",
    "total_marks_obtained",
    "result_declared_on",
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "certificate",
    "name",
    "course",
    "issued",
    "issue",
    "date",
    "number",
    "result",
    "marks",
    "grade",
    "student",
}


@dataclass
class OcrWord:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


class TemplateLearningEngine:
    """Learns layout/value associations from user-labeled certificates."""

    def __init__(self, storage_dir: str = "storage/template_learning") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / "templates.json"

    def list_templates(self) -> list[dict[str, Any]]:
        db = self._load_db()
        templates = db.get("templates", [])
        out: list[dict[str, Any]] = []
        for template in templates:
            out.append(
                {
                    "template_id": template.get("template_id"),
                    "template_name": template.get("template_name"),
                    "examples_count": int(template.get("examples_count", 0)),
                    "updated_at": template.get("updated_at"),
                }
            )
        out.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
        return out

    def add_training_example(
        self,
        file_path: str,
        annotations: dict[str, Any],
        template_name: str | None = None,
        template_id: str | None = None,
        include_all_fields: bool = False,
        full_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document = self._build_document(file_path)
        normalized_annotations = self._normalize_annotations(annotations)
        training_annotations: dict[str, str | None] = dict(normalized_annotations)
        if include_all_fields and isinstance(full_payload, dict):
            training_annotations.update(self._flatten_payload_for_training(full_payload))

        db = self._load_db()
        template = self._find_or_create_template(
            db=db,
            template_name=(template_name or "").strip() or None,
            template_id=template_id,
            document_keywords=document["keywords"],
        )

        field_profiles = template.setdefault("field_profiles", {})
        learned_fields = 0
        unresolved_fields: list[str] = []

        for field, value in training_annotations.items():
            if not value:
                continue

            profile = field_profiles.setdefault(field, {"samples": [], "avg_bbox": None})
            bbox = self._locate_value_bbox(document["words"], value)

            sample = {
                "value": value,
                "tokens": self._tokenize(value),
                "bbox": bbox,
            }
            samples = profile.setdefault("samples", [])
            samples.append(sample)
            if len(samples) > 30:
                profile["samples"] = samples[-30:]

            profile["avg_bbox"] = self._average_bbox(profile["samples"])
            if bbox:
                learned_fields += 1
            else:
                unresolved_fields.append(field)

        template["examples_count"] = int(template.get("examples_count", 0)) + 1
        template["updated_at"] = self._now_iso()
        template_keywords = set(template.get("keywords", []))
        template_keywords.update(document["keywords"][:40])
        template["keywords"] = sorted(template_keywords)[:120]

        self._save_db(db)

        return {
            "template_id": template.get("template_id"),
            "template_name": template.get("template_name"),
            "learned_fields": learned_fields,
            "total_annotated_fields": sum(1 for value in training_annotations.values() if value),
            "unresolved_fields": unresolved_fields,
        }

    def extract(self, file_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        db = self._load_db()
        templates = db.get("templates", [])
        if not templates:
            return {}, {"status": "no_templates"}

        document = self._build_document(file_path)
        best_template, score = self._match_template(document=document, templates=templates)
        if not best_template or score < 0.2:
            return {}, {"status": "no_template_match", "best_score": round(score, 3)}

        extracted: dict[str, Any] = {}
        field_profiles = best_template.get("field_profiles", {})
        field_confidences: dict[str, float] = {}

        extracted_flat: dict[str, str] = {}
        for field, profile in field_profiles.items():
            if not isinstance(profile, dict):
                continue
            value, confidence = self._extract_field_with_profile(document["words"], profile)
            if value:
                extracted_flat[str(field)] = value
                field_confidences[field] = confidence

        extracted = self._unflatten_to_nested(extracted_flat)
        denominator = max(1, len(field_profiles))
        coverage = len(extracted_flat) / denominator
        avg_conf = (sum(field_confidences.values()) / len(field_confidences)) if field_confidences else 0.0
        template_confidence = min(1.0, (0.65 * score) + (0.35 * avg_conf))

        debug = {
            "status": "matched",
            "template_id": best_template.get("template_id"),
            "template_name": best_template.get("template_name"),
            "template_match_score": round(score, 4),
            "template_coverage": round(coverage, 4),
            "template_confidence": round(template_confidence, 4),
            "extracted_fields_count": len(extracted_flat),
            "field_confidences": {k: round(v, 4) for k, v in field_confidences.items()},
        }
        return extracted, debug

    def _find_or_create_template(
        self,
        db: dict[str, Any],
        template_name: str | None,
        template_id: str | None,
        document_keywords: list[str],
    ) -> dict[str, Any]:
        templates = db.setdefault("templates", [])

        if template_id:
            for template in templates:
                if template.get("template_id") == template_id:
                    return template

        if template_name:
            for template in templates:
                if str(template.get("template_name", "")).strip().lower() == template_name.lower():
                    return template

        new_template = {
            "template_id": str(uuid4()),
            "template_name": template_name or f"Template {len(templates) + 1}",
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "examples_count": 0,
            "keywords": document_keywords[:40],
            "field_profiles": {},
        }
        templates.append(new_template)
        return new_template

    def _match_template(
        self,
        document: dict[str, Any],
        templates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, float]:
        best_template: dict[str, Any] | None = None
        best_score = 0.0
        doc_keywords = set(document["keywords"])

        for template in templates:
            template_keywords = set(template.get("keywords", []))
            keyword_similarity = self._jaccard(doc_keywords, template_keywords)

            field_profiles = template.get("field_profiles", {})
            profile_count = 0
            profile_hits = 0
            for field, profile in field_profiles.items():
                if not isinstance(profile, dict):
                    continue
                profile_count += 1
                value, conf = self._extract_field_with_profile(document["words"], profile)
                if value and conf >= 0.35:
                    profile_hits += 1

            field_similarity = (profile_hits / profile_count) if profile_count else 0.0
            score = (0.55 * keyword_similarity) + (0.45 * field_similarity)

            if score > best_score:
                best_score = score
                best_template = template

        return best_template, best_score

    def _extract_field_with_profile(self, words: list[OcrWord], profile: dict[str, Any]) -> tuple[str | None, float]:
        avg_bbox = profile.get("avg_bbox")
        samples = profile.get("samples", []) if isinstance(profile.get("samples"), list) else []

        if not avg_bbox or len(avg_bbox) != 4:
            return None, 0.0

        x1, y1, x2, y2 = avg_bbox
        # Expand region to absorb scan shifts and slight template changes.
        expand_x = 0.08
        expand_y = 0.05
        box = (
            max(0.0, x1 - expand_x),
            max(0.0, y1 - expand_y),
            min(1.0, x2 + expand_x),
            min(1.0, y2 + expand_y),
        )

        region_words = self._words_in_normalized_box(words, box)
        if not region_words:
            return None, 0.0

        region_text = self._join_words(region_words)
        cleaned_text = self._clean_extracted_text(region_text)
        if not cleaned_text:
            return None, 0.0

        sample_tokens: list[list[str]] = []
        for sample in samples[-8:]:
            tokens = sample.get("tokens")
            if isinstance(tokens, list) and tokens:
                sample_tokens.append([str(token) for token in tokens])

        predicted_tokens = self._tokenize(cleaned_text)
        if not predicted_tokens:
            return None, 0.0

        confidence = 0.35 + min(0.35, len(predicted_tokens) * 0.03)
        if sample_tokens:
            best_sim = 0.0
            for tokens in sample_tokens:
                sim = self._token_sequence_similarity(predicted_tokens, tokens)
                if sim > best_sim:
                    best_sim = sim
            confidence = min(1.0, confidence + (0.35 * best_sim))

        return cleaned_text, confidence

    def _words_in_normalized_box(self, words: list[OcrWord], box: tuple[float, float, float, float]) -> list[OcrWord]:
        x1, y1, x2, y2 = box
        selected: list[OcrWord] = []
        for word in words:
            cx = (word.x1 + word.x2) / 2
            cy = (word.y1 + word.y2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                selected.append(word)
        selected.sort(key=lambda w: (w.y1, w.x1))
        return selected

    def _build_document(self, file_path: str) -> dict[str, Any]:
        image = self._load_image(file_path)
        words, width, height = self._ocr_with_orientation_robustness(image)
        words = [word for word in words if word.text]
        full_text = " ".join(word.text for word in words)
        keywords = self._extract_keywords(full_text)

        return {
            "words": self._normalize_word_boxes(words, width, height),
            "keywords": keywords,
            "full_text": full_text,
        }

    def _load_image(self, file_path: str) -> Image.Image:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = convert_from_path(str(path), dpi=220, first_page=1, last_page=1)
            if not pages:
                raise ValueError("Could not render PDF for template learning")
            return pages[0].convert("RGB")

        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
            raise ValueError("Unsupported file type for template learning")

        return Image.open(path).convert("RGB")

    def _ocr_with_orientation_robustness(self, image: Image.Image) -> tuple[list[OcrWord], int, int]:
        variants = self._build_image_variants(image)
        best_words: list[OcrWord] = []
        best_score = -1.0
        best_size = image.size

        for variant in variants:
            words = self._ocr_words(variant)
            if not words:
                continue
            score = sum(max(0.0, word.conf) for word in words) + (6.0 * len(words))
            if score > best_score:
                best_score = score
                best_words = words
                best_size = variant.size

        if best_words:
            return best_words, best_size[0], best_size[1]

        fallback = self._ocr_words(image)
        return fallback, image.size[0], image.size[1]

    def _build_image_variants(self, image: Image.Image) -> list[Image.Image]:
        base = image.convert("RGB")
        variants = [base]

        gray = ImageOps.grayscale(base)
        auto = ImageOps.autocontrast(gray)
        sharp = ImageEnhance.Sharpness(auto).enhance(1.7)
        variants.append(sharp.convert("RGB"))

        # Orientation robustness for rotated scans.
        for angle in (90, 180, 270):
            variants.append(base.rotate(angle, expand=True))

        return variants

    def _ocr_words(self, image: Image.Image) -> list[OcrWord]:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        words: list[OcrWord] = []

        for idx in range(len(data.get("text", []))):
            text = str(data["text"][idx]).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][idx])
            except (TypeError, ValueError):
                conf = -1.0

            x = float(data["left"][idx])
            y = float(data["top"][idx])
            w = float(data["width"][idx])
            h = float(data["height"][idx])
            if w <= 0 or h <= 0:
                continue

            words.append(OcrWord(text=text, x1=x, y1=y, x2=x + w, y2=y + h, conf=conf))

        words.sort(key=lambda row: (row.y1, row.x1))
        return words

    def _normalize_word_boxes(self, words: list[OcrWord], width: int, height: int) -> list[OcrWord]:
        if width <= 0 or height <= 0:
            return words
        normalized: list[OcrWord] = []
        for word in words:
            normalized.append(
                OcrWord(
                    text=word.text,
                    x1=(word.x1 / width),
                    y1=(word.y1 / height),
                    x2=(word.x2 / width),
                    y2=(word.y2 / height),
                    conf=word.conf,
                )
            )
        return normalized

    def _locate_value_bbox(self, words: list[OcrWord], value: str) -> list[float] | None:
        target_tokens = self._tokenize(value)
        if not target_tokens:
            return None

        word_tokens = [self._normalize_token(word.text) for word in words]
        best_match: tuple[int, int, float] | None = None
        n = len(words)
        m = len(target_tokens)

        for start in range(0, max(1, n - m + 1)):
            end = min(n, start + m)
            window = word_tokens[start:end]
            if len(window) != m:
                continue

            sims = [SequenceMatcher(None, window[i], target_tokens[i]).ratio() for i in range(m)]
            avg_sim = sum(sims) / m
            if best_match is None or avg_sim > best_match[2]:
                best_match = (start, end, avg_sim)

        if best_match and best_match[2] >= 0.72:
            return self._bbox_from_words(words[best_match[0] : best_match[1]])

        token_set = set(target_tokens)
        hits = [idx for idx, token in enumerate(word_tokens) if token and token in token_set]
        if hits:
            grouped = [words[idx] for idx in hits]
            return self._bbox_from_words(grouped)

        return None

    def _bbox_from_words(self, words: list[OcrWord]) -> list[float] | None:
        if not words:
            return None
        x1 = min(word.x1 for word in words)
        y1 = min(word.y1 for word in words)
        x2 = max(word.x2 for word in words)
        y2 = max(word.y2 for word in words)
        return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]

    def _average_bbox(self, samples: list[dict[str, Any]]) -> list[float] | None:
        boxes: list[list[float]] = []
        for sample in samples:
            bbox = sample.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
        if not boxes:
            return None

        avg = [sum(item[idx] for item in boxes) / len(boxes) for idx in range(4)]
        return [round(value, 6) for value in avg]

    def _normalize_annotations(self, annotations: dict[str, Any]) -> dict[str, str | None]:
        normalized: dict[str, str | None] = {}
        for field in LEARNABLE_FIELDS:
            raw = annotations.get(field)
            if raw is None:
                normalized[field] = None
                continue
            text = str(raw).strip()
            normalized[field] = text or None
        return normalized

    def _flatten_payload_for_training(self, payload: dict[str, Any]) -> dict[str, str | None]:
        flat: dict[str, str | None] = {}

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    next_path = f"{path}.{key}" if path else str(key)
                    walk(value, next_path)
                return

            if isinstance(node, list):
                for idx, item in enumerate(node):
                    next_path = f"{path}.{idx}" if path else str(idx)
                    walk(item, next_path)
                return

            if node is None:
                return
            if isinstance(node, (str, int, float, bool)):
                text = str(node).strip()
                if text:
                    flat[path] = text

        walk(payload, "")
        # Exclude generated signal from training.
        flat.pop("confidence_score", None)
        return flat

    def _unflatten_to_nested(self, flat: dict[str, str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in flat.items():
            self._set_nested_value(out, key, value)
        return out

    def _set_nested_value(self, target: dict[str, Any], dotted_path: str, value: Any) -> None:
        if not dotted_path:
            return

        parts = dotted_path.split(".")
        current: Any = target

        for idx, part in enumerate(parts):
            is_last = idx == len(parts) - 1
            next_part = parts[idx + 1] if not is_last else None
            next_is_index = next_part.isdigit() if next_part is not None else False

            if part.isdigit():
                # Index segment should only appear after container creation.
                list_index = int(part)
                if not isinstance(current, list):
                    return
                while len(current) <= list_index:
                    current.append({} if not is_last else None)
                if is_last:
                    current[list_index] = value
                    return
                if not isinstance(current[list_index], (dict, list)):
                    current[list_index] = [] if next_is_index else {}
                current = current[list_index]
                continue

            if is_last:
                if isinstance(current, dict):
                    current[part] = value
                return

            if not isinstance(current, dict):
                return

            if part not in current or not isinstance(current[part], (dict, list)):
                current[part] = [] if next_is_index else {}
            current = current[part]

    def _extract_keywords(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        deduped: list[str] = []
        for token in tokens:
            if len(token) < 4 or token in STOPWORDS:
                continue
            if token not in deduped:
                deduped.append(token)
            if len(deduped) >= 120:
                break
        return deduped

    @staticmethod
    def _token_sequence_similarity(a: list[str], b: list[str]) -> float:
        if not a or not b:
            return 0.0
        seq_a = " ".join(a)
        seq_b = " ".join(b)
        return SequenceMatcher(None, seq_a, seq_b).ratio()

    @staticmethod
    def _join_words(words: list[OcrWord]) -> str:
        text = " ".join(word.text.strip() for word in words if word.text.strip())
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _clean_extracted_text(text: str) -> str | None:
        out = re.sub(r"\s+", " ", text).strip(" ,.;:|-")
        return out or None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.split(r"[^A-Za-z0-9]+", text.lower()) if token]

    @staticmethod
    def _normalize_token(token: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", token.lower())

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return inter / union

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _load_db(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"templates": []}
        try:
            return json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception:
            return {"templates": []}

    def _save_db(self, db: dict[str, Any]) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")

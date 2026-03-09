from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.template_learning import LEARNABLE_FIELDS

COLLEGE_DB = Path("storage/colleges.json")


def _default_colleges() -> list[dict[str, Any]]:
    return [
        {"name": "SIES School of Business Studies", "fields": list(LEARNABLE_FIELDS)},
        {"name": "Inmantec College", "fields": list(LEARNABLE_FIELDS)},
    ]


def _normalize_fields(fields: list[str] | None) -> list[str]:
    if not fields:
        return list(LEARNABLE_FIELDS)
    normalized: list[str] = []
    seen: set[str] = set()
    for field in fields:
        candidate = str(field or "").strip()
        if not candidate:
            continue
        candidate = candidate.replace(" ", "_").lower()
        if candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized or list(LEARNABLE_FIELDS)


def _load_db() -> dict[str, Any]:
    if not COLLEGE_DB.exists():
        return {"colleges": []}
    try:
        return json.loads(COLLEGE_DB.read_text(encoding="utf-8"))
    except Exception:
        return {"colleges": []}


def _save_db(db: dict[str, Any]) -> None:
    COLLEGE_DB.parent.mkdir(parents=True, exist_ok=True)
    COLLEGE_DB.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")


def load_colleges() -> list[dict[str, Any]]:
    db = _load_db()
    colleges = db.get("colleges")
    if not isinstance(colleges, list):
        colleges = []
    if not colleges:
        colleges = _default_colleges()
        _save_db({"colleges": colleges})
    return [
        {"name": str(entry.get("name", "")).strip(), "fields": _normalize_fields(entry.get("fields"))}
        for entry in colleges
        if str(entry.get("name", "")).strip()
    ]


def save_colleges(colleges: list[dict[str, Any]]) -> None:
    normalized = []
    seen: set[str] = set()
    for entry in colleges:
        name = str(entry.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append({"name": name, "fields": _normalize_fields(entry.get("fields"))})
    _save_db({"colleges": normalized})


def get_college(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    normalized_name = name.strip().lower()
    for entry in load_colleges():
        if entry["name"].strip().lower() == normalized_name:
            return entry
    return None


def upsert_college(name: str, fields: list[str]) -> dict[str, Any]:
    if not name.strip():
        raise ValueError("College name cannot be empty")
    colleges = load_colleges()
    normalized_fields = _normalize_fields(fields)
    for entry in colleges:
        if entry["name"].strip().lower() == name.strip().lower():
            entry["fields"] = normalized_fields
            save_colleges(colleges)
            return entry
    new_entry = {"name": name.strip(), "fields": normalized_fields}
    colleges.append(new_entry)
    save_colleges(colleges)
    return new_entry

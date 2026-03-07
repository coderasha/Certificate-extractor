#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extractor import CertificateExtractor


IGNORE_PATHS_DEFAULT = {"confidence_score"}


@dataclass
class FieldResult:
    path: str
    expected: Any
    predicted: Any
    matched: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate extraction accuracy against ground-truth JSON files.",
    )
    parser.add_argument("--input-file", help="Single certificate path (PDF/image).")
    parser.add_argument("--input-dir", help="Directory containing certificate files.")
    parser.add_argument("--ground-truth-file", help="Single ground-truth JSON file.")
    parser.add_argument(
        "--ground-truth-dir",
        default="evaluation/ground_truth",
        help="Directory with ground-truth JSON files named <input_stem>.json",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--include-confidence",
        action="store_true",
        help="Include confidence_score in accuracy calculation.",
    )
    parser.add_argument(
        "--report-file",
        help="Optional output report JSON path.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=30,
        help="Maximum mismatches printed per file.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(",", "")
    return text


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value).strip()
    if isinstance(value, str):
        return normalize_text(value)
    return value


def is_effectively_equal(expected: Any, predicted: Any) -> bool:
    exp = normalize_value(expected)
    pred = normalize_value(predicted)

    if exp is None:
        return pred is None
    if pred is None:
        return False
    if exp == pred:
        return True

    if isinstance(exp, str) and isinstance(pred, str):
        if re.fullmatch(r"\d+(\.\d+)?", exp) or re.fullmatch(r"\d+\s*/\s*\d+", exp):
            return exp == pred
        similarity = SequenceMatcher(None, exp, pred).ratio()
        return similarity >= 0.94
    return False


def flatten_expected(
    node: Any,
    path: list[str],
    out: list[tuple[list[str], Any]],
    ignored_leaf_names: set[str],
) -> None:
    if path and path[-1] in ignored_leaf_names:
        return

    if isinstance(node, dict):
        for key in sorted(node.keys()):
            flatten_expected(node[key], [*path, str(key)], out, ignored_leaf_names)
        return

    if isinstance(node, list):
        for idx, value in enumerate(node):
            flatten_expected(value, [*path, str(idx)], out, ignored_leaf_names)
        return

    out.append((path, node))


def get_by_path(node: Any, path: list[str]) -> Any:
    cur = node
    for token in path:
        if isinstance(cur, dict):
            if token not in cur:
                return None
            cur = cur[token]
            continue
        if isinstance(cur, list):
            try:
                idx = int(token)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        return None
    return cur


def compare(expected: dict[str, Any], predicted: dict[str, Any], include_confidence: bool) -> tuple[float, list[FieldResult]]:
    ignored = set() if include_confidence else set(IGNORE_PATHS_DEFAULT)
    flat_expected: list[tuple[list[str], Any]] = []
    flatten_expected(expected, [], flat_expected, ignored)

    results: list[FieldResult] = []
    scored = 0
    matched = 0
    for path_tokens, expected_value in flat_expected:
        # Skip expected nulls from scoring denominator.
        if expected_value is None:
            continue
        scored += 1
        predicted_value = get_by_path(predicted, path_tokens)
        ok = is_effectively_equal(expected_value, predicted_value)
        if ok:
            matched += 1
        results.append(
            FieldResult(
                path=".".join(path_tokens),
                expected=expected_value,
                predicted=predicted_value,
                matched=ok,
            )
        )

    accuracy = (matched / scored) if scored else 0.0
    return accuracy, results


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    if args.input_file:
        files.append(Path(args.input_file))
    if args.input_dir:
        for path in sorted(Path(args.input_dir).iterdir()):
            if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}:
                files.append(path)
    if not files:
        raise ValueError("Provide --input-file or --input-dir.")
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(missing))
    return files


def resolve_ground_truth(input_path: Path, args: argparse.Namespace) -> Path:
    if args.ground_truth_file:
        gt_path = Path(args.ground_truth_file)
    else:
        gt_path = Path(args.ground_truth_dir) / f"{input_path.stem}.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found for {input_path.name}: {gt_path}")
    return gt_path


def main() -> int:
    args = parse_args()
    files = resolve_inputs(args)

    extractor = CertificateExtractor(
        timeout=args.timeout,
    )

    aggregate_scored = 0
    aggregate_matched = 0
    report: dict[str, Any] = {"files": []}

    for path in files:
        gt_path = resolve_ground_truth(path, args)
        expected = json.loads(gt_path.read_text(encoding="utf-8"))
        predicted = extractor.extract(str(path))

        accuracy, field_results = compare(expected, predicted, args.include_confidence)
        scored = sum(1 for item in field_results if item.expected is not None)
        matched = sum(1 for item in field_results if item.matched)
        aggregate_scored += scored
        aggregate_matched += matched

        mismatches = [item for item in field_results if not item.matched]
        print(f"\nFile: {path}")
        print(f"Ground Truth: {gt_path}")
        print(f"Accuracy: {accuracy * 100:.2f}% ({matched}/{scored})")
        if mismatches:
            print("Mismatches:")
            for item in mismatches[: args.max_mismatches]:
                print(
                    f"  - {item.path}: expected={json.dumps(item.expected, ensure_ascii=False)} "
                    f"predicted={json.dumps(item.predicted, ensure_ascii=False)}"
                )

        report["files"].append(
            {
                "input_file": str(path),
                "ground_truth_file": str(gt_path),
                "accuracy_percent": round(accuracy * 100, 2),
                "matched_fields": matched,
                "scored_fields": scored,
                "mismatches": [
                    {
                        "path": item.path,
                        "expected": item.expected,
                        "predicted": item.predicted,
                    }
                    for item in mismatches
                ],
            }
        )

    overall = (aggregate_matched / aggregate_scored) if aggregate_scored else 0.0
    print(f"\nOverall Accuracy: {overall * 100:.2f}% ({aggregate_matched}/{aggregate_scored})")
    report["overall_accuracy_percent"] = round(overall * 100, 2)
    report["overall_matched_fields"] = aggregate_matched
    report["overall_scored_fields"] = aggregate_scored

    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

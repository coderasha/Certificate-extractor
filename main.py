import argparse
import json
import sys

from extractor import CertificateExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract structured certificate data from a PDF or image file."
    )
    parser.add_argument("input_file", help="Path to certificate file (PDF/image)")
    parser.add_argument(
        "--model",
        default="llama3.2-vision",
        help="LLaMA Vision model name served by Ollama",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/chat",
        help="Ollama chat endpoint URL",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    extractor = CertificateExtractor(
        model=args.model,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
    )

    try:
        result = extractor.extract(args.input_file)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

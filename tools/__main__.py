"""CLI for day-12 structured tools."""

from __future__ import annotations

import argparse
import json

from tools.core import run_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a StudyAbroad structured tool.")
    parser.add_argument(
        "tool",
        choices=["permit_renewal", "document_checklist", "office_lookup"],
    )
    parser.add_argument(
        "--json",
        required=True,
        help='JSON object payload, e.g. {"expiry_date":"2026-12-31"}',
    )
    args = parser.parse_args()
    payload = json.loads(args.json)
    result = run_tool(args.tool, payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

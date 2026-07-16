#!/usr/bin/env python
"""Write a ground-truth-free v5/v6 ingest diagnostic (never hardval metrics)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.pipeline.vocab import load_vocabulary  # noqa: E402
from backend.tools.ingest_compare import compare_ingests  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--baseline-log")
    parser.add_argument("--candidate-log")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = compare_ingests(
        baseline_root=args.baseline_root,
        candidate_root=args.candidate_root,
        vocab=load_vocabulary(args.vocab),
        baseline_log=args.baseline_log,
        candidate_log=args.candidate_log,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["deltas"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

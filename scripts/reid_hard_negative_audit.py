#!/usr/bin/env python
"""Generate a machine-readable, truth-status-aware S3 hard-negative audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.reid.hard_negative_audit import audit_hard_negatives, write_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = audit_hard_negatives(
        template_path=args.template,
        review_path=args.review,
        result_dir=args.result_dir,
    )
    write_audit(summary, args.out)
    print(
        json.dumps(
            {
                "review_status": summary["review_status"],
                "hard_negative_evaluated": summary["hard_negative_evaluated"],
                "g2_evaluated": summary["g2_evaluated"],
                "opposite_merge_groups": summary["opposite_merge_groups"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

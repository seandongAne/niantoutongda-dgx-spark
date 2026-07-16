#!/usr/bin/env python
"""按冻结公式对 prompt 候选的预测 JSON 做确定性排序。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.detection_eval import (  # noqa: E402
    SCHEMA_VERSION,
    EvaluationInputError,
    load_json_document,
    rank_prompt_candidates,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank prompt candidates by recall - lambda*FP/frame - mu*fragmentation."
        )
    )
    parser.add_argument("ground_truth", help="ground-truth JSON file")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="ID=PREDICTIONS.json",
        help="candidate ID and its prediction JSON; repeat for every candidate",
    )
    parser.add_argument("--lambda-fp", required=True, type=float)
    parser.add_argument("--mu-fragmentation", required=True, type=float)
    parser.add_argument("--output", help="write ranking JSON here instead of stdout")
    args = parser.parse_args(argv)

    try:
        ground_truth = load_json_document(args.ground_truth)
        candidates = _load_candidates(args.candidate)
        ranked = rank_prompt_candidates(
            ground_truth,
            candidates,
            lambda_fp=args.lambda_fp,
            mu_fragmentation=args.mu_fragmentation,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": ranked[0].evaluation.dataset_id,
            "formula": (
                "visible_instance_recall - lambda_fp * false_positives_per_frame "
                "- mu_fragmentation * fragmentation_rate"
            ),
            "weights": {
                "lambda_fp": args.lambda_fp,
                "mu_fragmentation": args.mu_fragmentation,
            },
            "ranking": [item.to_dict(rank) for rank, item in enumerate(ranked, start=1)],
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (EvaluationInputError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


def _load_candidates(specifications: list[str]) -> dict[str, Mapping[str, Any]]:
    candidates: dict[str, Mapping[str, Any]] = {}
    for specification in specifications:
        if "=" not in specification:
            raise EvaluationInputError(
                f"candidate must use ID=PREDICTIONS.json syntax: {specification!r}"
            )
        candidate_id, path = specification.split("=", 1)
        if not candidate_id.strip() or not path:
            raise EvaluationInputError(
                f"candidate must use non-empty ID=PREDICTIONS.json: {specification!r}"
            )
        if candidate_id in candidates:
            raise EvaluationInputError(f"duplicate candidate_id: {candidate_id!r}")
        candidates[candidate_id] = load_json_document(path)
    return candidates


if __name__ == "__main__":
    raise SystemExit(main())

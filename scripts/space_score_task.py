#!/usr/bin/env python
"""Score automatic spatial regions against semantic-only frozen truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.spatial.scoring import (  # noqa: E402
    load_frozen_spatial_truth,
    load_region_manifest,
    score_spatial_regions,
    write_spatial_scoring_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        required=True,
        type=Path,
        help="automatic RegionManifest regions.json",
    )
    parser.add_argument(
        "--truth",
        required=True,
        type=Path,
        help="frozen semantic-only truth manifest",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction = load_region_manifest(args.regions)
    truth = load_frozen_spatial_truth(args.truth)
    result = score_spatial_regions(
        prediction,
        truth,
        required_expected_anchor_count=args.expected_count,
    )
    outputs = write_spatial_scoring_outputs(result, args.out_dir)
    print(
        json.dumps(
            {
                "score": result.metrics.score,
                "accepted": result.metrics.acceptance_passed,
                "gate_reasons": result.metrics.gate_reasons,
                "normalized_hash": result.normalized_hash,
                "outputs": outputs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.metrics.acceptance_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

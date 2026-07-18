#!/usr/bin/env python
"""Produce trusted new-home regions from automatic observation JSONL.

This command has no manual-region-manifest input.  It writes a candidate
manifest and diagnostics on every valid run, but writes the solver-compatible
``regions.json`` only when the configured automatic coverage gate passes.
``--shadow-only`` keeps a failed gate as a successful diagnostic stage so a
separately configured, explicit manual fallback may continue downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.spatial import (  # noqa: E402
    SpatialProducerConfig,
    load_observations_jsonl,
    produce_spatial_regions,
    write_spatial_outputs,
)


def _expected_anchors(values: Sequence[str]) -> list[str]:
    return [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="new-home video identifier")
    parser.add_argument(
        "--observations", required=True, type=Path, help="automatic observation JSONL"
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-regions", type=int, default=5)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--min-hard-field-confidence", type=float, default=0.70)
    parser.add_argument("--min-power-confidence", type=float, default=0.70)
    parser.add_argument("--min-field-consensus", type=float, default=0.67)
    parser.add_argument("--dedupe-iou", type=float, default=0.35)
    parser.add_argument(
        "--expected-anchor",
        action="append",
        default=[],
        help="required anchor label; repeat or pass a comma-separated list",
    )
    parser.add_argument(
        "--allow-partial-expected-coverage",
        action="store_true",
        help="report NOT_OBSERVED expected anchors without blocking the global gate",
    )
    parser.add_argument(
        "--shadow-only",
        action="store_true",
        help=(
            "return success after writing diagnostics even when the automatic "
            "gate fails; never creates regions.json on failure"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = SpatialProducerConfig(
        min_regions=args.min_regions,
        min_observations_per_region=args.min_observations,
        min_model_confidence=args.min_confidence,
        min_hard_field_confidence=args.min_hard_field_confidence,
        min_power_confidence=args.min_power_confidence,
        min_field_consensus=args.min_field_consensus,
        dedupe_iou_threshold=args.dedupe_iou,
        expected_anchor_labels=_expected_anchors(args.expected_anchor),
        require_expected_coverage=not args.allow_partial_expected_coverage,
    )
    observations = load_observations_jsonl(args.observations, video_id=args.video_id)
    result = produce_spatial_regions(args.video_id, observations, config)
    outputs = write_spatial_outputs(result, args.out_dir)
    print(
        json.dumps(
            {
                "video_id": args.video_id,
                "gate_status": result.metrics.gate_status.value,
                "gate_reasons": result.metrics.gate_reasons,
                "observations": result.metrics.observation_count,
                "auto_accepted": result.metrics.auto_accepted_count,
                "projected_regions": result.metrics.projected_region_count,
                "normalized_hash": result.normalized_hash,
                "outputs": outputs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.gate_passed or args.shadow_only else 2


if __name__ == "__main__":
    raise SystemExit(main())

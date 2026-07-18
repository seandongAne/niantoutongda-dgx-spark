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
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.spatial import (  # noqa: E402
    AnchorAssignmentConfig,
    SpatialProducerConfig,
    load_automatic_anchor_candidates,
    load_observations_jsonl,
    produce_assigned_spatial_regions,
    produce_spatial_regions,
    write_spatial_outputs,
)
from backend.tools.spatial.assignment import (  # noqa: E402
    load_expected_anchor_contract_manifest,
)

ASSIGNMENT_FILENAME = "assignment.json"


def _expected_anchors(values: Sequence[str]) -> list[str]:
    return [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hashed_input(path: Path, hashes_path: Path, output_name: str) -> None:
    try:
        payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{hashes_path}: invalid JSON: {exc.msg}") from exc
    expected = (payload.get("outputs") or {}).get(output_name)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{hashes_path}: missing output hash for {output_name}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{path}: sha256 mismatch against {hashes_path}; "
            f"expected={expected}, actual={actual}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="new-home video identifier")
    parser.add_argument(
        "--observations", required=True, type=Path, help="automatic observation JSONL"
    )
    parser.add_argument(
        "--observation-hashes",
        type=Path,
        help="adapter hashes.json used to authenticate auto_observations.jsonl",
    )
    parser.add_argument(
        "--anchor-candidates",
        type=Path,
        help="Spark-local VLM automatic candidate array; enables global assignment mode",
    )
    parser.add_argument(
        "--anchor-hashes",
        type=Path,
        help="classifier hashes.json used to authenticate anchor_candidates.json",
    )
    parser.add_argument(
        "--anchor-contract",
        type=Path,
        help=(
            "strict production anchor/support/capacity contract; required with "
            "--anchor-candidates and intentionally separate from scorer truth"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-regions", type=int, default=5)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--min-hard-field-confidence", type=float, default=0.70)
    parser.add_argument("--min-power-confidence", type=float, default=0.70)
    parser.add_argument("--min-field-consensus", type=float, default=0.67)
    parser.add_argument("--dedupe-iou", type=float, default=0.35)
    parser.add_argument("--min-anchor-vote-share", type=float, default=0.60)
    parser.add_argument("--min-vlm-mean-confidence", type=float, default=0.50)
    parser.add_argument("--min-assignment-score", type=float, default=0.70)
    parser.add_argument("--min-assignment-margin", type=float, default=0.05)
    parser.add_argument(
        "--support-saturation-observations",
        type=int,
        default=5,
        help=(
            "detector observation count that saturates the assignment support "
            "component"
        ),
    )
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
    # A malformed or mismatched rerun must not leave an earlier trusted region
    # manifest reachable in the same output directory.
    stale_region_path = args.out_dir / "regions.json"
    if stale_region_path.exists():
        stale_region_path.unlink()
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
    if args.observation_hashes is not None:
        _verify_hashed_input(
            args.observations,
            args.observation_hashes,
            "auto_observations.jsonl",
        )
    observations = load_observations_jsonl(args.observations, video_id=args.video_id)
    assignment = None
    if args.anchor_candidates is not None:
        if not config.expected_anchor_labels:
            parser.error("--anchor-candidates requires at least one --expected-anchor")
        if args.anchor_contract is None:
            parser.error("--anchor-candidates requires --anchor-contract")
        if args.anchor_hashes is not None:
            _verify_hashed_input(
                args.anchor_candidates,
                args.anchor_hashes,
                "anchor_candidates.json",
            )
        candidates = load_automatic_anchor_candidates(args.anchor_candidates)
        contract_manifest = load_expected_anchor_contract_manifest(
            args.anchor_contract
        )
        result, assignment = produce_assigned_spatial_regions(
            args.video_id,
            observations,
            candidates,
            config,
            AnchorAssignmentConfig(
                min_candidate_observations=args.min_observations,
                min_label_vote_count=2,
                min_label_vote_share=args.min_anchor_vote_share,
                min_mean_confidence=args.min_vlm_mean_confidence,
                min_hard_field_confidence=args.min_hard_field_confidence,
                min_power_confidence=args.min_power_confidence,
                min_assignment_score=args.min_assignment_score,
                min_runner_up_margin=args.min_assignment_margin,
                support_saturation_observations=(
                    args.support_saturation_observations
                ),
            ),
            anchor_contracts=contract_manifest.anchors,
        )
    else:
        if args.anchor_hashes is not None:
            parser.error("--anchor-hashes requires --anchor-candidates")
        if args.anchor_contract is not None:
            parser.error("--anchor-contract requires --anchor-candidates")
        result = produce_spatial_regions(args.video_id, observations, config)
    outputs = write_spatial_outputs(result, args.out_dir)
    if assignment is not None:
        assignment_path = args.out_dir / ASSIGNMENT_FILENAME
        assignment_path.write_text(
            assignment.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        outputs["assignment"] = str(assignment_path)
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
                "assignment_hash": (
                    assignment.normalized_hash if assignment is not None else None
                ),
                "outputs": outputs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.gate_passed or args.shadow_only else 2


if __name__ == "__main__":
    raise SystemExit(main())

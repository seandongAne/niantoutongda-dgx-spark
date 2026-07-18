#!/usr/bin/env python
"""Apply an auditable visual overlay to automatic spatial candidates.

The review is bound to the automatic spatial ``normalized.sha256`` and to
project-relative evidence frames.  Diagnostics are written for every valid
review, while solver-compatible ``regions.json`` is emitted only when all five
frozen anchors are visually adjudicated and no decision remains unresolved.
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
    SpatialCandidateManifest,
    adjudicate_spatial_regions,
    load_visual_adjudication,
    remove_stale_adjudicated_regions,
    write_spatial_adjudication_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="automatic candidate_manifest.json",
    )
    parser.add_argument(
        "--source-hash",
        required=True,
        type=Path,
        help="automatic spatial normalized.sha256",
    )
    parser.add_argument(
        "--review",
        required=True,
        type=Path,
        help="visual adjudication review JSON",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Never leave a previous trusted regions.json reachable after a failed or
    # malformed rerun.  Validation happens only after this targeted cleanup.
    remove_stale_adjudicated_regions(args.out_dir)

    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidate_manifest = SpatialCandidateManifest.model_validate(candidate_payload)
    source_hash = args.source_hash.read_text(encoding="ascii").strip()
    review = load_visual_adjudication(args.review)
    result = adjudicate_spatial_regions(
        candidate_manifest,
        source_hash,
        review,
        project_root=PROJ,
    )
    outputs = write_spatial_adjudication_outputs(result, args.out_dir)
    print(
        json.dumps(
            {
                "review_id": review.review_id,
                "gate_status": result.metrics.gate_status.value,
                "gate_reasons": result.metrics.gate_reasons,
                "visually_adjudicated": result.metrics.visually_adjudicated_count,
                "projected_regions": result.metrics.projected_region_count,
                "normalized_hash": result.normalized_hash,
                "outputs": outputs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

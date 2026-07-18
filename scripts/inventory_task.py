#!/usr/bin/env python
"""Build the strict 20-row, tracklet-audited hero inventory projection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.core import AgentHandoff, AgentRole  # noqa: E402
from backend.tools.inventory import (  # noqa: E402
    project_inventory_files,
    write_inventory_projection,
)
from backend.tools.trace import finalize_message, write_fragment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entities",
        type=Path,
        default=Path("results/hero_s1/reid-final/entities.jsonl"),
    )
    parser.add_argument(
        "--items", type=Path, default=Path("fixtures/hero_s1/items.json")
    )
    parser.add_argument(
        "--anchor-review",
        type=Path,
        default=Path(
            "fixtures/hero_s1/annotations/anchor_review.confirmed.json"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-clarifications", type=int, default=4)
    parser.add_argument("--trace-id")
    parser.add_argument("--trace-out", type=Path)
    args = parser.parse_args()
    if bool(args.trace_id) != bool(args.trace_out):
        parser.error("--trace-id and --trace-out must be provided together")

    projection = project_inventory_files(
        entities_path=args.entities,
        items_path=args.items,
        anchor_review_path=args.anchor_review,
        max_clarifications=args.max_clarifications,
    )
    write_inventory_projection(projection, args.out_dir)
    if args.trace_out:
        message = finalize_message(
            AgentHandoff(
                message_id=f"{args.trace_id}-entities-ready",
                correlation_id=args.trace_id,
                producer=AgentRole.MEM,
                target=AgentRole.GROUP,
                action="ENTITIES_READY",
                item_ids=[row["entity_id"] for row in projection.trusted_entities],
                artifact_refs=[
                    str(args.out_dir / "inventory.jsonl"),
                    str(args.out_dir / "trusted_entities.jsonl"),
                    str(args.out_dir / "display.jsonl"),
                    str(args.out_dir / "clarifications.jsonl"),
                    str(args.out_dir / "metrics.json"),
                ],
                summary={
                    "raw_entities": projection.metrics["raw_entity_count"],
                    "trusted_inventory": projection.metrics[
                        "trusted_inventory_count"
                    ],
                    "clarifications": len(projection.clarifications),
                    "raw_unresolved": projection.metrics[
                        "raw_link_unresolved_count"
                    ],
                },
            )
        )
        write_fragment(args.trace_out, [message])
    print(
        json.dumps(
            {
                "raw_entities": projection.metrics["raw_entity_count"],
                "trusted_inventory": projection.metrics[
                    "trusted_inventory_count"
                ],
                "downstream_eligible": projection.metrics[
                    "downstream_eligible_count"
                ],
                "raw_links_complete": projection.metrics[
                    "raw_link_complete_count"
                ],
                "raw_links_unresolved": projection.metrics[
                    "raw_link_unresolved_count"
                ],
                "clarifications": len(projection.clarifications),
                "deferred": projection.metrics["deferred_unresolved_count"],
                "projection_hash": projection.projection_hash,
                "out_dir": args.out_dir.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

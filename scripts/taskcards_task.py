#!/usr/bin/env python
"""任务卡阶段 — 布局结果 → 结构化任务卡 JSONL + 可读 markdown。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.hero_bundle import HeroGroup, RegionManifest  # noqa: E402
from backend.schemas.core import AgentHandoff, AgentRole  # noqa: E402
from backend.tools.solver.layout_solver import LayoutResult  # noqa: E402
from backend.tools.taskcards import build_task_cards, task_card_markdown  # noqa: E402
from backend.tools.trace import (  # noqa: E402
    finalize_message,
    require_handoff,
    write_fragment,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", required=True, type=Path)
    ap.add_argument("--layout", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--display", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trace-id")
    ap.add_argument("--trace-parent", type=Path)
    ap.add_argument("--trace-out", type=Path)
    args = ap.parse_args()
    trace_args = (args.trace_id, args.trace_parent, args.trace_out)
    if any(trace_args) and not all(trace_args):
        ap.error("--trace-id, --trace-parent and --trace-out must be provided together")

    groups = [HeroGroup.model_validate(row) for row in load_jsonl(args.groups)]
    layout_data = json.loads(args.layout.read_text(encoding="utf-8"))
    layout = LayoutResult(
        status=layout_data["status"],
        assignments=layout_data["assignments"],
        alternatives=layout_data["alternatives"],
        objective=layout_data["objective"],
        conflicts=layout_data["conflicts"],
    )
    manifest = RegionManifest.model_validate(
        json.loads(args.regions.read_text(encoding="utf-8"))
    )
    entity_display = {row["entity_id"]: row for row in load_jsonl(args.display)}

    cards = build_task_cards(groups, layout, manifest, entity_display)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "taskcards.jsonl").open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(card.model_dump_json() + "\n")
    (out / "taskcards.md").write_text(
        "# 搬家任务卡\n\n" + "\n".join(task_card_markdown(c) for c in cards),
        encoding="utf-8",
    )
    if args.trace_out:
        parent = require_handoff(args.trace_parent, "PLACEMENT_READY")
        message = finalize_message(
            AgentHandoff(
                message_id=f"{args.trace_id}-tasks-ready",
                correlation_id=parent.correlation_id,
                causation_id=parent.message_id,
                producer=AgentRole.EXEC,
                target=AgentRole.USER,
                action="TASKS_READY",
                item_ids=[card.card_id for card in cards],
                artifact_refs=[
                    str(out / "taskcards.jsonl"),
                    str(out / "taskcards.md"),
                ],
                summary={"taskcards": len(cards)},
            )
        )
        write_fragment(args.trace_out, [message])
    print(json.dumps({"cards": len(cards)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

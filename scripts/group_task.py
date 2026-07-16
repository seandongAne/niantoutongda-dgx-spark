#!/usr/bin/env python
"""GROUP 生活组合阶段 — 旁白主证据,共现只佐证。

输入:实体展示表(entity_naming 产物)+ narration.jsonl,
可选轻确认 / 模板规则 / 共现统计。
输出:groups.jsonl(HeroGroup)、life_groups.jsonl(core LifeGroup)、
clarifications.jsonl、conflicts.json、audit-events.jsonl(追加式)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.hero_bundle import (  # noqa: E402
    GroupConfirmation,
    NarrationItem,
)
from backend.schemas.core import AgentHandoff, AgentRole  # noqa: E402
from backend.tools.audit.store import append_event  # noqa: E402
from backend.tools.grouping import build_groups, resolve_narration  # noqa: E402
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
    ap.add_argument("--display", required=True, type=Path)
    ap.add_argument("--narration", required=True, type=Path)
    ap.add_argument("--confirmations", type=Path, default=None)
    ap.add_argument("--template-rules", type=Path, default=None)
    ap.add_argument("--cooccurrence", type=Path, default=None)
    ap.add_argument("--config-version", default="group-v1")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trace-id")
    ap.add_argument("--trace-parent", type=Path)
    ap.add_argument("--trace-out", type=Path)
    args = ap.parse_args()
    trace_args = (args.trace_id, args.trace_parent, args.trace_out)
    if any(trace_args) and not all(trace_args):
        ap.error("--trace-id, --trace-parent and --trace-out must be provided together")

    entity_display = {
        row["entity_id"]: row for row in load_jsonl(args.display)
    }
    items = [
        NarrationItem.model_validate(row) for row in load_jsonl(args.narration)
    ]
    confirmations = (
        [
            GroupConfirmation.model_validate(row)
            for row in json.loads(args.confirmations.read_text(encoding="utf-8"))
        ]
        if args.confirmations
        else []
    )
    template_rules = (
        json.loads(args.template_rules.read_text(encoding="utf-8"))
        if args.template_rules
        else {}
    )
    cooccurrence = (
        {
            (pair["a"], pair["b"]): pair["count"]
            for pair in json.loads(args.cooccurrence.read_text(encoding="utf-8"))
        }
        if args.cooccurrence
        else {}
    )

    resolutions = resolve_narration(items, entity_display)
    build = build_groups(
        entity_display,
        items,
        resolutions,
        confirmations,
        template_rules,
        cooccurrence,
        config_version=args.config_version,
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "groups.jsonl").open("w", encoding="utf-8") as f:
        for group in build.groups:
            f.write(group.model_dump_json() + "\n")
    with (out / "life_groups.jsonl").open("w", encoding="utf-8") as f:
        for group in build.groups:
            f.write(group.to_life_group().model_dump_json() + "\n")
    with (out / "resolutions.jsonl").open("w", encoding="utf-8") as f:
        for res in resolutions:
            f.write(res.model_dump_json() + "\n")
    with (out / "clarifications.jsonl").open("w", encoding="utf-8") as f:
        for clar in build.clarifications:
            f.write(clar.model_dump_json() + "\n")
    (out / "conflicts.json").write_text(
        json.dumps(build.conflicts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for event in build.audit_events:
        append_event(out / "audit-events.jsonl", event)

    if args.trace_out:
        parent = require_handoff(args.trace_parent, "ENTITIES_READY")
        message = finalize_message(
            AgentHandoff(
                message_id=f"{args.trace_id}-groups-ready",
                correlation_id=parent.correlation_id,
                causation_id=parent.message_id,
                producer=AgentRole.GROUP,
                target=AgentRole.SPACE,
                action="GROUPS_READY",
                item_ids=[group.group_id for group in build.groups],
                artifact_refs=[
                    str(out / "groups.jsonl"),
                    str(out / "life_groups.jsonl"),
                ],
                summary={
                    "groups": len(build.groups),
                    "unassigned": len(build.unassigned_entity_ids),
                    "clarifications": len(build.clarifications),
                },
            )
        )
        write_fragment(args.trace_out, [message])

    print(
        json.dumps(
            {
                "groups": len(build.groups),
                "grouped_entities": sum(len(g.entity_ids) for g in build.groups),
                "unassigned": len(build.unassigned_entity_ids),
                "clarifications": len(build.clarifications),
                "conflicts": len(build.conflicts),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

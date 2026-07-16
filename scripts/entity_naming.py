#!/usr/bin/env python
"""实体级展示名 — 命名权威原则落地。

用户可见名字只来自 S5 本地 VLM 对 hero 图的命名;GDINO raw_label 是内部
召回键,绝不外显。实体展示名取"最像完整物体"的成员轨:优先用 tracklet
质量分(--tracklets-dir 提供时),否则按 S5 confidence(high>medium>low)
+ 证据图数量 + tracklet_id 字典序,全程确定性。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.grouping.narration import COLOR_ZH  # noqa: E402
from backend.schemas.core import (  # noqa: E402
    AgentHandoff,
    AgentRole,
    ClarificationDecision,
    ClarificationRequest,
)
from backend.tools.trace import finalize_message, write_fragment  # noqa: E402

CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}
QUALITY_FIELDS = ("hero_score", "quality", "max_quality")


def disambiguate_duplicates(rows: list[dict]) -> None:
    """同款不同色重名消歧:展示名相同但主色不同 → 后缀中文颜色词。

    这是演示"分得清同类两件"的用户可见面;颜色也相同的重名保持原样
    (旁白/轻确认层再区分)。
    """
    by_name: dict[str, list[dict]] = {}
    for row in rows:
        by_name.setdefault(row["display_name_zh"], []).append(row)
    for name, dupes in by_name.items():
        if len(dupes) < 2:
            continue
        colors = {r["color_primary"] for r in dupes}
        if len(colors) < 2:
            continue
        for row in dupes:
            zh = COLOR_ZH.get(row["color_primary"], "")
            if zh:
                row["display_name_zh"] = f"{name}({zh})"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_tracklet_quality(tracklets_dir: Path | None) -> dict[str, float]:
    quality: dict[str, float] = {}
    if tracklets_dir is None:
        return quality
    for path in sorted(tracklets_dir.glob("*/tracklets.jsonl")):
        for row in load_jsonl(path):
            attrs = row.get("attributes") or {}
            for field in QUALITY_FIELDS:
                value = row.get(field, attrs.get(field))
                if value is not None:
                    quality[row["tracklet_id"]] = float(value)
                    break
    return quality


def member_sort_key(tid: str, attr_row: dict | None, quality: dict[str, float]):
    attrs = (attr_row or {}).get("attributes", {})
    return (
        -quality.get(tid, 0.0),
        CONFIDENCE_RANK.get(attrs.get("confidence", ""), 3),
        -len((attr_row or {}).get("sources", [])),
        tid,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entities", required=True, type=Path)
    ap.add_argument("--attributes", required=True, type=Path)
    ap.add_argument("--tracklets-dir", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clarifications", type=Path)
    ap.add_argument("--trace-id")
    ap.add_argument("--trace-out", type=Path)
    args = ap.parse_args()
    if bool(args.trace_id) != bool(args.trace_out):
        ap.error("--trace-id and --trace-out must be provided together")

    entities = load_jsonl(args.entities)
    attr_by_tid = {row["tracklet_id"]: row for row in load_jsonl(args.attributes)}
    quality = load_tracklet_quality(args.tracklets_dir)

    named = missing = 0
    rows: list[dict] = []
    for entity in sorted(entities, key=lambda e: e["entity_id"]):
        members = sorted(
            entity["tracklet_ids"],
            key=lambda tid: member_sort_key(tid, attr_by_tid.get(tid), quality),
        )
        chosen = members[0]
        attrs = attr_by_tid.get(chosen, {}).get("attributes", {})
        display = attrs.get("label_zh", "").strip()
        if display and display != "unknown":
            source = "vlm"
            named += 1
        else:
            display = "未命名物品"
            source = "missing"
            missing += 1
        sources = attr_by_tid.get(chosen, {}).get("sources", [])
        rows.append(
            {
                "entity_id": entity["entity_id"],
                "display_name_zh": display,
                "display_source": source,
                "color_primary": attrs.get("color_primary", ""),
                "hero_crop_ref": sources[0] if sources else "",
                "source_tracklet_id": chosen,
                "internal_recall_label": entity.get("label", ""),
            }
        )
    disambiguate_duplicates(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    if args.trace_out:
        messages = []
        if args.clarifications:
            for raw in load_jsonl(args.clarifications):
                legacy_decision = raw.get("decision")
                request = ClarificationRequest.model_validate(raw).model_copy(
                    update={"decision": None}
                )
                finalize_message(request)
                messages.append(request)
                if legacy_decision:
                    messages.append(
                        finalize_message(
                            ClarificationDecision(
                                message_id=f"{request.message_id}-decision",
                                correlation_id=request.correlation_id,
                                causation_id=request.message_id,
                                producer=AgentRole.USER,
                                request_id=request.message_id,
                                decision=legacy_decision,
                            )
                        )
                    )
        messages.append(
            finalize_message(
                AgentHandoff(
                    message_id=f"{args.trace_id}-mem-ready",
                    correlation_id=args.trace_id,
                    producer=AgentRole.MEM,
                    target=AgentRole.GROUP,
                    action="ENTITIES_READY",
                    item_ids=[row["entity_id"] for row in rows],
                    artifact_refs=[str(args.entities), str(args.out)],
                    summary={"entities": len(rows), "clarifications": len(messages)},
                )
            )
        )
        write_fragment(args.trace_out, messages)
    print(
        json.dumps(
            {"entities": len(entities), "vlm_named": named, "missing_name": missing},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

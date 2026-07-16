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
from backend.tools.solver.layout_solver import LayoutResult  # noqa: E402
from backend.tools.taskcards import build_task_cards, task_card_markdown  # noqa: E402


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
    args = ap.parse_args()

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
    print(json.dumps({"cards": len(cards)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

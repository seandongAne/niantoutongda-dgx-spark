#!/usr/bin/env python
"""旁白转结构化条目 — GROUP 的主证据通道。

输入二选一:
  --transcript  文本稿,每行一件物品(ASR 产出或人工誊写;# 开头为注释)
  --items       已是 NarrationItem JSONL,只做校验透传
输出 narration.jsonl(NarrationItem 契约)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.hero_bundle import NarrationItem  # noqa: E402
from backend.tools.grouping.narration import parse_narration_line  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--transcript", type=Path)
    src.add_argument("--items", type=Path)
    ap.add_argument("--audio-ref", default="")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    items: list[NarrationItem] = []
    if args.transcript:
        lines = [
            line.strip()
            for line in args.transcript.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for idx, line in enumerate(lines, start=1):
            item = parse_narration_line(f"n{idx:02d}", line)
            if args.audio_ref:
                item = item.model_copy(update={"audio_ref": args.audio_ref})
            items.append(item)
    else:
        with args.items.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(NarrationItem.model_validate(json.loads(line)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    with_partners = sum(1 for i in items if i.group_partners)
    print(
        json.dumps(
            {
                "items": len(items),
                "with_partners": with_partners,
                "with_target": sum(1 for i in items if i.target_location),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

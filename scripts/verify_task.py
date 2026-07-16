#!/usr/bin/env python
"""验收复核阶段 — 任务卡 + 验收照片清单 → 消息族 + verdict 汇总。

FAILED/NEEDS_USER 是合法业务结局,不是管线错误(退出码仍为 0);
协议错误(覆盖缺口、id 不符)会抛异常停链,绝不静默降级。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.hero_bundle import AcceptanceManifest, TaskCard  # noqa: E402
from backend.tools.verification.acceptance import (  # noqa: E402
    card_status_after,
    verify_cards,
)
from backend.tools.trace import require_handoff, write_fragment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", required=True, type=Path)
    ap.add_argument("--photos", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trace-parent", type=Path)
    ap.add_argument("--trace-out", type=Path)
    args = ap.parse_args()

    cards = [
        TaskCard.model_validate(json.loads(line))
        for line in args.cards.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    acceptance = AcceptanceManifest.model_validate(
        json.loads(args.photos.read_text(encoding="utf-8"))
    )

    parent = (
        require_handoff(args.trace_parent, "TASKS_READY")
        if args.trace_parent
        else None
    )
    results = verify_cards(
        cards,
        acceptance,
        parent_message_id=parent.message_id if parent else None,
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    messages = []
    for result in results:
        messages.extend(
            [result.request, result.presence, result.compliance, result.verdict]
        )
        if result.adjudication:
            messages.append(result.adjudication)
    trace_out = args.trace_out or (out / "messages.jsonl")
    write_fragment(trace_out, messages)
    # 保留历史约定的 verify/messages.jsonl；若 trace-out 另有路径则同步一份。
    if trace_out != out / "messages.jsonl":
        write_fragment(out / "messages.jsonl", messages)

    summary = {
        r.card.card_id: {
            "verdict": r.verdict.verdict,
            "reason_codes": r.verdict.reason_codes,
            "photo_refs": r.request.photo_refs,
            "status_after": card_status_after(r).value,
            "adjudication": (
                {
                    "decision": r.adjudication.decision,
                    "note": r.adjudication.note,
                }
                if r.adjudication
                else None
            ),
        }
        for r in results
    }
    (out / "verdicts.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (out / "taskcards_verified.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            card = r.card.model_copy(update={"status": card_status_after(r)})
            f.write(card.model_dump_json() + "\n")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict.verdict] = counts.get(r.verdict.verdict, 0) + 1
    print(json.dumps({"cards": len(results), **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""合并或严格回放四 Agent trace；退出 0 才代表协议、hash 与闭环全通过。

单文件判分入口:
  .venv/bin/python scripts/replay_trace.py results/hero/dev-fixture/audit/events.jsonl --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.trace import load_trace, merge_fragments, validate_trace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", nargs="?", type=Path)
    ap.add_argument("--fragments", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--require-main-chain", action="store_true")
    ap.add_argument("--require-verification", action="store_true")
    ap.add_argument("--require-closed-choices", action="store_true")
    ap.add_argument("--require-adjudication", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    if bool(args.trace) == bool(args.fragments):
        ap.error("exactly one of TRACE or --fragments is required")
    if args.fragments:
        if args.out is None:
            ap.error("--fragments requires --out")
        messages = merge_fragments(args.fragments, args.out)
    else:
        messages = load_trace(args.trace)

    report = validate_trace(
        messages,
        require_main_chain=args.strict or args.require_main_chain,
        require_verification=args.strict or args.require_verification,
        require_closed_choices=args.strict or args.require_closed_choices,
        require_adjudication=args.strict or args.require_adjudication,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print("✅ TRACE REPLAY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

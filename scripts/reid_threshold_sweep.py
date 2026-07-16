#!/usr/bin/env python
"""S3 match/margin 阈值扫描:同一 ingest 上跑网格,产出交给 anchor_gt_eval 打分。

只在拿到数据所有者 GT 之后使用(P2 纪律);本脚本不读 GT、不宣布优胜,
只负责确定性地生成每个组合的完整 S3 产物。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.reid.matcher import IdentityConstraints, run_reid  # noqa: E402
from backend.tools.reid.model import ReIDConfig, Vocabulary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--config", required=True, help="基准配置(其余字段不动)")
    parser.add_argument("--matches", default="0.80,0.82,0.84,0.86")
    parser.add_argument("--margins", default="0.00,0.02,0.04")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = ReIDConfig.from_yaml(args.config)
    vocab = Vocabulary.from_json(args.vocab)
    matches = [float(x) for x in args.matches.split(",")]
    margins = [float(x) for x in args.margins.split(",")]
    out = Path(args.out)
    summary = []
    for match in matches:
        for margin in margins:
            version = f"{base.version}-sweep-m{match:.2f}-g{margin:.2f}"
            config = replace(
                base,
                version=version,
                thresholds=replace(base.thresholds, match=match, margin=margin),
            )
            config.validate()
            run = run_reid(
                ingest_root=args.ingest_root,
                config=config,
                vocab=vocab,
                constraints=IdentityConstraints(),
            )
            combo_dir = out / f"m{match:.2f}-g{margin:.2f}"
            run.write(combo_dir)
            summary.append(
                {
                    "match": match,
                    "margin": margin,
                    "automatic_link_count": run.metrics["automatic_link_count"],
                    "clarification_count": run.metrics["clarification_count"],
                    "matched_entity_count": run.metrics["matched_entity_count"],
                }
            )
            print(json.dumps(summary[-1]))
    (out / "sweep-summary.json").write_text(
        json.dumps({"base_config": base.version, "grid": summary},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

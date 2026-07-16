#!/usr/bin/env python
"""48 词候选判卷 — 逐类过滤 GT 后按冻结公式排序,输出对照表。

score = recall − λ·FP/帧 − μ·碎片率(单帧扫描无跨帧轨,碎片率恒定,
排序实际由 recall 与 FP/帧 驱动)。当前词表的表现同表列出作 baseline 对照;
GT 支持度(框数)一并输出——tumbler 仅 1 框、luggage 2 框,结论只作方向参考。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.detection_eval import (  # noqa: E402
    load_json_document,
    rank_prompt_candidates,
)


def filter_gt(gt: dict, canonical_id: str) -> dict:
    frames = []
    for frame in gt["frames"]:
        instances = [
            i for i in frame["instances"] if i["canonical_id"] == canonical_id
        ]
        frames.append({**frame, "instances": instances})
    return {"dataset_id": gt["dataset_id"], "frames": frames}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--scan-dir", required=True, type=Path)
    ap.add_argument("--lambda-fp", type=float, default=0.2)
    ap.add_argument("--mu-fragmentation", type=float, default=0.4)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    manifest = json.loads(
        (args.scan_dir / "scan-manifest.json").read_text(encoding="utf-8")
    )
    phrases = {k: v["phrase"] for k, v in manifest["candidates"].items()}

    by_category: dict[str, dict[str, dict]] = {}
    for path in sorted(args.scan_dir.glob("*__c*.json")):
        category, cid = path.stem.split("__")
        by_category.setdefault(category, {})[cid] = load_json_document(path)

    report: dict[str, object] = {
        "lambda_fp": args.lambda_fp,
        "mu_fragmentation": args.mu_fragmentation,
        "categories": {},
    }
    for category in sorted(by_category):
        gt_sub = filter_gt(gt, category)
        support = sum(
            1 for f in gt_sub["frames"] for i in f["instances"] if i.get("visible")
        )
        ranked = rank_prompt_candidates(
            gt_sub,
            by_category[category],
            lambda_fp=args.lambda_fp,
            mu_fragmentation=args.mu_fragmentation,
        )
        rows = []
        for r in ranked:
            ev = r.evaluation
            rows.append(
                {
                    "candidate": r.candidate_id,
                    "phrase": phrases.get(f"{category}/{r.candidate_id}", ""),
                    "score": round(r.score, 4),
                    "recall": round(ev.recall, 4),
                    "fp_per_frame": round(ev.false_positives_per_frame, 3),
                    "fragmentation": round(ev.fragmentation_rate, 4),
                }
            )
        report["categories"][category] = {"gt_boxes": support, "ranking": rows}
        best = rows[0] if rows else {}
        print(
            f"{category:<16} GT框 {support:2d} | 最优 {best.get('candidate')} "
            f"recall {best.get('recall')} fp/帧 {best.get('fp_per_frame')} "
            f"| {best.get('phrase')}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""AutoTune-v1 内循环代理指标(不读 GT——GT 判卷预算受契约限制 ≤2 次)。

对一个候选 reid 运行目录输出:
  - 实体数 / 跨视频实体数 / 澄清数(G2e 的代理)
  - 已接受链接数与 margin 分布(合并置信度的代理)
  - tutor 一致率:tutor 高置信同/异判定 vs 候选运行的同实体归属(检索质量的代理)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reid-dir", required=True, type=Path)
    ap.add_argument("--tutor-pairs", required=True, type=Path)
    ap.add_argument("--min-tutor-conf", type=float, default=0.70)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tid_entity: dict[str, str] = {}
    n_entities = n_cross = 0
    for line in (args.reid_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        n_entities += 1
        videos = {t.rsplit("_t", 1)[0] for t in row["tracklet_ids"]}
        if len(videos) > 1:
            n_cross += 1
        for t in row["tracklet_ids"]:
            tid_entity[t] = row["entity_id"]

    n_clar = sum(
        1 for line in (args.reid_dir / "clarifications.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    )

    margins, scores = [], []
    for line in (args.reid_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("assigned"):
            scores.append(row["score"])
            if isinstance(row.get("margin"), (int, float)):
                margins.append(row["margin"])

    agree = {"same_total": 0, "same_hit": 0, "diff_total": 0, "diff_hit": 0}
    for line in args.tutor_pairs.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        tutor = row.get("tutor") or {}
        conf = tutor.get("confidence")
        if not isinstance(conf, (int, float)) or conf < args.min_tutor_conf:
            continue
        a, b = row["tracklet_a"], row["tracklet_b"]
        if a not in tid_entity or b not in tid_entity:
            continue
        same_run = tid_entity[a] == tid_entity[b]
        if tutor.get("same") is True:
            agree["same_total"] += 1
            agree["same_hit"] += int(same_run)
        elif tutor.get("same") is False:
            agree["diff_total"] += 1
            agree["diff_hit"] += int(not same_run)

    def pct(hit: int, total: int) -> float | None:
        return round(hit / total, 4) if total else None

    def q(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(p * len(s)))], 4)

    report = {
        "reid_dir": str(args.reid_dir),
        "entities": n_entities,
        "cross_video_entities": n_cross,
        "clarifications": n_clar,
        "accepted_links": len(scores),
        "margin": {
            "mean": round(statistics.fmean(margins), 4) if margins else None,
            "p10": q(margins, 0.10),
        },
        "tutor_agreement": {
            "same_recall": pct(agree["same_hit"], agree["same_total"]),
            "diff_precision_guard": pct(agree["diff_hit"], agree["diff_total"]),
            "counts": agree,
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

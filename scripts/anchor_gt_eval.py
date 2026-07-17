#!/usr/bin/env python
"""17 锚点 G2 判卷器:用数据所有者确认的 anchor→tracklet 真值给 S3 运行打分。

指标口径(entities.template.json 冻结的 gate):
* complete_merge:该锚点存在一个实体,覆盖其每个 confirmed 视频至少一条确认轨,
  且该实体不含任何其他锚点的确认轨(混入即不完整,只算 false merge)。
* high_confidence_false_merge:同一实体含 >=2 个锚点的确认轨(多成员实体只能
  由自动链接产生,故一律算高置信)。
* recall_at_1:对每条确认轨(rep 空间)与每个"该锚点在对面视频有确认轨"的视频对,
  candidates.jsonl 里它在该视频的 top-1 伙伴是否属于同锚点。
* clarification 负担:澄清请求按"同锚点(有效提问)/跨锚点(危险提问)/涉锚点单边/
  不涉锚点"分桶。
* 四组硬负:任一实体同时含两侧确认轨(直接或经 stitch 传递)即 FAIL。

stitch 运行(v2)的 candidates/clarifications 在代表 id 空间,经 stitch-map 展开;
entities.jsonl 本就是原始 id 空间。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(
    run_dir: Path, review: dict, complete_min: int = 15, recall_min: float = 0.85
) -> dict:
    entities = _rows(run_dir / "entities.jsonl")
    candidates = _rows(run_dir / "candidates.jsonl")
    clarifications = _rows(run_dir / "clarifications.jsonl")
    stitch_path = run_dir / "stitch-map.json"
    groups = json.loads(stitch_path.read_text()).get("groups", {}) if stitch_path.exists() else {}
    member_to_rep = {m: rep for rep, members in groups.items() for m in members}

    def rep(tid: str) -> str:
        return member_to_rep.get(tid, tid)

    anchor_of: dict[str, str] = {}
    confirmed_by_anchor: dict[str, dict[str, list[str]]] = {}
    for ent in review["entities"]:
        aid = ent["anchor_id"]
        confirmed_by_anchor[aid] = ent["confirmed_tracklet_ids_by_video"]
        for ids in ent["confirmed_tracklet_ids_by_video"].values():
            for tid in ids:
                anchor_of[tid] = aid

    # rep 空间的锚点归属;stitch 把两个锚点的轨并进一组 = stitch 级误合并
    rep_anchors: dict[str, set[str]] = defaultdict(set)
    for tid, aid in anchor_of.items():
        rep_anchors[rep(tid)].add(aid)
    stitch_false_merges = sorted(
        (r, sorted(aids)) for r, aids in rep_anchors.items() if len(aids) > 1
    )

    # ---- 实体级:完整合并 / 误合并 ----
    entity_anchors: list[tuple[dict, set[str]]] = []
    for entity in entities:
        aids = {anchor_of[t] for t in entity["tracklet_ids"] if t in anchor_of}
        if aids:
            entity_anchors.append((entity, aids))
    false_merge_entities = [
        {
            "entity_id": e["entity_id"],
            "anchors": sorted(a),
            "confirmed_members": sorted(t for t in e["tracklet_ids"] if t in anchor_of),
        }
        for e, a in entity_anchors
        if len(a) > 1
    ]

    complete, incomplete = [], {}
    for aid, by_video in confirmed_by_anchor.items():
        videos = set(by_video)
        best_cover: set[str] = set()
        ok = False
        for entity, aids in entity_anchors:
            if aid not in aids:
                continue
            members = set(entity["tracklet_ids"])
            covered = {v for v, ids in by_video.items() if members & set(ids)}
            if covered >= videos and aids == {aid}:
                ok = True
                break
            if len(covered) > len(best_cover) and aids == {aid}:
                best_cover = covered
        if ok:
            complete.append(aid)
        else:
            incomplete[aid] = {
                "needed_videos": sorted(videos),
                "best_pure_entity_covers": sorted(best_cover),
            }

    # ---- Recall@1(candidates 是 rep 空间) ----
    video_of_rep: dict[str, str] = {}
    best_partner: dict[tuple[str, str], tuple[float, str]] = {}
    for row in candidates:
        a, b, score = row["tracklet_a"], row["tracklet_b"], row["score"]
        va, vb = row["video_pair"]
        video_of_rep[a] = va
        video_of_rep[b] = vb
        for src, dst, dst_video in ((a, b, vb), (b, a, va)):
            key = (src, dst_video)
            if key not in best_partner or score > best_partner[key][0] or (
                score == best_partner[key][0] and dst < best_partner[key][1]
            ):
                best_partner[key] = (score, dst)

    hits, opportunities = 0, 0
    per_anchor_r1: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    misses = []
    for aid, by_video in confirmed_by_anchor.items():
        anchor_reps_by_video: dict[str, set[str]] = defaultdict(set)
        for v, ids in by_video.items():
            for tid in ids:
                anchor_reps_by_video[v].add(rep(tid))
        for v_src, reps in anchor_reps_by_video.items():
            for r in reps:
                for v_dst, dst_reps in anchor_reps_by_video.items():
                    if v_dst == v_src:
                        continue
                    opportunities += 1
                    per_anchor_r1[aid][1] += 1
                    top = best_partner.get((r, v_dst))
                    if top and top[1] in dst_reps:
                        hits += 1
                        per_anchor_r1[aid][0] += 1
                    else:
                        misses.append(
                            {"anchor": aid, "from": r, "to_video": v_dst,
                             "top1": top[1] if top else None,
                             "top1_score": round(top[0], 4) if top else None}
                        )

    # ---- 澄清负担分桶 ----
    def expanded_anchors(rep_id: str) -> set[str]:
        return rep_anchors.get(rep_id, set())

    buckets = {"same_anchor": 0, "cross_anchor": 0, "anchor_vs_unknown": 0, "no_anchor": 0}
    cross_anchor_pairs = []
    for req in clarifications:
        a_anchors = expanded_anchors(req["candidate_a"])
        b_anchors = expanded_anchors(req["candidate_b"])
        if a_anchors and b_anchors:
            if a_anchors & b_anchors:
                buckets["same_anchor"] += 1
            else:
                buckets["cross_anchor"] += 1
                cross_anchor_pairs.append(
                    {"a": req["candidate_a"], "b": req["candidate_b"],
                     "anchors": sorted(a_anchors | b_anchors)}
                )
        elif a_anchors or b_anchors:
            buckets["anchor_vs_unknown"] += 1
        else:
            buckets["no_anchor"] += 1

    # ---- 四组硬负 ----
    hard_negative = {}
    for pair in review["hard_negative_pairs"]:
        x, y = pair["anchor_ids"]
        crossing = [
            fm for fm in false_merge_entities if x in fm["anchors"] and y in fm["anchors"]
        ]
        stitch_crossing = [
            s for s in stitch_false_merges if x in s[1] and y in s[1]
        ]
        hard_negative[pair["group_id"]] = {
            "anchors": [x, y],
            "category": pair["category_id"],
            "entity_crossings": crossing,
            "stitch_crossings": stitch_crossing,
            "verdict": "PASS" if not crossing and not stitch_crossing else "FAIL",
        }

    gate = {
        "complete_merge": {
            "value": len(complete), "min": complete_min,
            "pass": len(complete) >= complete_min,
        },
        "recall_at_1": {
            "value": round(hits / opportunities, 4) if opportunities else None,
            "hits": hits, "opportunities": opportunities, "min": recall_min,
            "pass": opportunities > 0 and hits / opportunities >= recall_min,
        },
        "high_confidence_false_merge": {
            "value": len(false_merge_entities), "max": 0,
            "pass": not false_merge_entities,
        },
        "hard_negative_groups": {
            "value": sum(v["verdict"] == "PASS" for v in hard_negative.values()),
            "of": len(hard_negative),
            "pass": all(v["verdict"] == "PASS" for v in hard_negative.values()),
        },
    }
    return {
        "run_dir": str(run_dir),
        "config_version": json.loads((run_dir / "metrics.json").read_text()).get("config_version"),
        "gate": gate,
        "complete_anchors": sorted(complete),
        "incomplete_anchors": incomplete,
        "false_merge_entities": false_merge_entities,
        "stitch_false_merges": stitch_false_merges,
        "recall_at_1_per_anchor": {
            aid: {"hits": v[0], "opportunities": v[1]} for aid, v in sorted(per_anchor_r1.items())
        },
        "recall_misses": misses,
        "clarification_buckets": buckets,
        "cross_anchor_clarifications": cross_anchor_pairs,
        "hard_negative": hard_negative,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True, help="anchor_review.v6.confirmed.json")
    parser.add_argument("--runs", nargs="+", required=True, help="S3 输出目录(可多个对照)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--complete-min", type=int, default=15,
                        help="G2c 完整合并门(dev_a=15;hero-s1 预注册=16)")
    parser.add_argument("--recall-min", type=float, default=0.85,
                        help="G2d R@1 门(dev_a=0.85;hero-s1 预注册=0.90)")
    args = parser.parse_args()

    review = json.loads(Path(args.review).read_text())
    if review.get("status") != "data_owner_confirmed":
        raise SystemExit("refusing to evaluate: review status is not data_owner_confirmed")
    reports = [evaluate(Path(run), review, args.complete_min, args.recall_min)
               for run in args.runs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": "1.0", "reports": reports},
                              ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    for report in reports:
        summary = {k: v for k, v in report["gate"].items()}
        print(report["config_version"], json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

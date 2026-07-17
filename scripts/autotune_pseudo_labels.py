#!/usr/bin/env python
"""AutoTune-v1 伪标签构造(纯本地,不调云)。

来源契约(验收门修订记录 2026-07-17):伪标签仅来自
  ① reid 运行自身的高置信证据:stitch 组(同视频缝合)+ score≥门限的跨视频链接;
  ② tutor 建议:autotune_tutor.py pairs 通道 confidence≥门限的同物/异物判定。
364 轨 GT 确认集**绝不**参与标签构造;本脚本读 GT 仅为输出"与 GT 集的
tracklet 重叠率"这一项修订条款要求的审计数字。

产物:SF1 标签 schema(entities[].anchor_id + confirmed_tracklet_ids_by_video,
anchor_id 为 pseudo_NNN 伪簇号)+ manifest(参数/来源计数/冲突/重叠率)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 取字典序小者为根,保证确定性
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reid-dir", required=True, type=Path)
    ap.add_argument("--ingest-root", required=True, type=Path)
    ap.add_argument("--tutor-pairs", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path, help="仅用于重叠率审计,不入标签")
    ap.add_argument("--min-link-score", type=float, default=0.93)
    ap.add_argument("--min-tutor-conf", type=float, default=0.70)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    tid_video: dict[str, str] = {}
    for path in sorted(args.ingest_root.glob("*/tracklets.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                tid_video[row["tracklet_id"]] = row["video_id"]

    uf = UnionFind()
    counts = {"stitch_edges": 0, "link_edges": 0, "tutor_same_edges": 0, "tutor_diff_constraints": 0}

    stitch = json.loads((args.reid_dir / "stitch-map.json").read_text(encoding="utf-8"))
    for canonical, members in (stitch.get("groups") or {}).items():
        for m in members:
            if m != canonical and m in tid_video and canonical in tid_video:
                uf.union(canonical, m)
                counts["stitch_edges"] += 1

    for line in (args.reid_dir / "accepted-links.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["score"] >= args.min_link_score:
            uf.union(row["tracklet_a"], row["tracklet_b"])
            counts["link_edges"] += 1

    cannot: list[tuple[str, str]] = []
    for line in args.tutor_pairs.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        tutor = row.get("tutor") or {}
        conf = tutor.get("confidence")
        if not isinstance(conf, (int, float)) or conf < args.min_tutor_conf:
            continue
        a, b = row["tracklet_a"], row["tracklet_b"]
        if a not in tid_video or b not in tid_video:
            continue
        if tutor.get("same") is True:
            uf.union(a, b)
            counts["tutor_same_edges"] += 1
        elif tutor.get("same") is False:
            cannot.append((a, b))
            counts["tutor_diff_constraints"] += 1

    clusters: dict[str, list[str]] = {}
    for tid in list(uf.parent):
        clusters.setdefault(uf.find(tid), []).append(tid)

    poisoned = set()
    for a, b in cannot:
        if uf.find(a) == uf.find(b):
            poisoned.add(uf.find(a))

    kept, dropped = [], {"conflict": 0, "single_video": 0, "too_small": 0, "thin_train": 0}
    for root, tids in sorted(clusters.items()):
        if root in poisoned:
            dropped["conflict"] += 1
            continue
        videos = sorted({tid_video[t] for t in tids})
        if len(videos) < 2:
            dropped["single_video"] += 1
            continue
        if len(tids) < 3:
            dropped["too_small"] += 1
            continue
        held = videos[-1]
        if sum(1 for t in tids if tid_video[t] != held) < 2:
            dropped["thin_train"] += 1
            continue
        kept.append(sorted(tids))

    kept.sort(key=lambda tids: tids[0])
    entities = []
    for i, tids in enumerate(kept):
        by_video: dict[str, list[str]] = {}
        for t in tids:
            by_video.setdefault(tid_video[t], []).append(t)
        entities.append(
            {
                "anchor_id": f"pseudo_{i:03d}",
                "confirmed_tracklet_ids_by_video": {v: sorted(ts) for v, ts in sorted(by_video.items())},
            }
        )

    labeled = {t for tids in kept for t in tids}
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    gt_tids = {
        t
        for e in gt["entities"]
        for ts in e.get("confirmed_tracklet_ids_by_video", {}).values()
        for t in ts
    }
    overlap = len(labeled & gt_tids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "_provenance": "AutoTune-v1 伪标签:模型高置信证据+tutor 建议;非人工真值,禁用于判卷",
                "dataset_version": "hero-s1-vocab1",
                "entities": entities,
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {
            "min_link_score": args.min_link_score,
            "min_tutor_conf": args.min_tutor_conf,
        },
        "sources": {
            "reid_dir": str(args.reid_dir),
            "tutor_pairs": {"path": str(args.tutor_pairs), "sha256": sha256_file(args.tutor_pairs)},
        },
        "edges": counts,
        "clusters": {"kept": len(kept), "dropped": dropped},
        "samples": len(labeled),
        "gt_overlap": {
            "note": "修订条款要求的审计数字;GT 未参与标签构造",
            "labeled_in_gt": overlap,
            "labeled_total": len(labeled),
            "rate": round(overlap / max(1, len(labeled)), 4),
        },
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["clusters"] | {"samples": len(labeled), "gt_overlap_rate": manifest["gt_overlap"]["rate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

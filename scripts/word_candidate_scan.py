#!/usr/bin/env python
"""48 词候选 GDINO 扫描 — 每短语独跑 hardval 40 帧,导出 detection_eval 预测。

单短语单批(不与其他词同批),隔离词间干扰;全帧视图、不开 tile,
48 个候选完全同工况。产物:<out>/<category>__c<idx>.json + scan-manifest.json。
在 ~/venv(视觉环境)运行,预计 10~20 分钟,走 nohup。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.pipeline.detect import GroundingDinoDetector  # noqa: E402

FRAME_RE = re.compile(r"^(?P<seq>v\d+)_kf_(?P<frame>\d+)\.jpg$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset-id", default="dev_a_hardval")
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    candidates: dict[str, list[str]] = json.loads(
        args.candidates.read_text(encoding="utf-8")
    )
    frames = []
    for path in sorted(args.frames_dir.glob("*.jpg")):
        m = FRAME_RE.match(path.name)
        if not m:
            raise ValueError(f"帧名不符合 v*_kf_*.jpg: {path.name}")
        frames.append((m.group("seq"), m.group("frame"), str(path)))
    if not frames:
        raise SystemExit("frames-dir 为空")

    detector = GroundingDinoDetector(args.model, box_threshold=args.box_threshold)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": detector.model_version,
        "box_threshold": args.box_threshold,
        "frames": len(frames),
        "candidates": {},
    }
    paths = [p for _, _, p in frames]
    t_all = time.time()
    for category in sorted(candidates):
        for idx, phrase in enumerate(candidates[category]):
            t0 = time.time()
            per_frame = detector.detect_many(paths, [phrase])
            rows = []
            for (seq, frame_id, _), dets in zip(frames, per_frame):
                rows.append(
                    {
                        "sequence_id": seq,
                        "frame_id": frame_id,
                        "predictions": [
                            {
                                "bbox": list(d.box),
                                "canonical_id": category,
                                "track_id": f"{category}_c{idx}_{seq}_{frame_id}_{k}",
                            }
                            for k, d in enumerate(dets)
                        ],
                    }
                )
            name = f"{category}__c{idx}.json"
            (args.out_dir / name).write_text(
                json.dumps(
                    {"dataset_id": args.dataset_id, "frames": rows},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            det_total = sum(len(d) for d in per_frame)
            manifest["candidates"][f"{category}/c{idx}"] = {
                "phrase": phrase,
                "detections": det_total,
                "seconds": round(time.time() - t0, 1),
            }
            print(f"{category} c{idx} {det_total:4d} dets "
                  f"{time.time() - t0:5.1f}s  {phrase}", flush=True)
    manifest["wall_seconds"] = round(time.time() - t_all, 1)
    (args.out_dir / "scan-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wall_seconds": manifest["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

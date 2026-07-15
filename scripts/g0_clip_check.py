#!/usr/bin/env python
"""G0 素材快检 — 单段视频过真实管线(关键帧→检测→轨迹),输出可用性报告。

用法(节点主环境):
  python scripts/g0_clip_check.py <video.mp4> [--prompts "lamp,cup,..."] [--workdir DIR]

判定口径:
  - 关键帧数、每帧平均检出数;
  - 成轨情况:长轨(≥4 帧)/短轨/碎轨比 —— 碎轨占比高 = 移动过快或光线问题;
  - 各轨迹的标签与时长,供人工对照"锚点是否被看见"。
这是拍摄质量检查,不是验收;不写审计事件,不产生契约产物。
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from pathlib import Path
import sys

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

DEFAULT_PROMPTS = (
    "lamp,cup,mug,book,box,pillow,bottle,plant,clock,phone,charger,"
    "bag,shoe,towel,picture frame,mirror,chair,desk,bed,laptop,keyboard,"
    "headphones,glasses,watch,remote control,tissue box,trash can,fan,speaker"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    from backend.pipeline.detect import GroundingDinoDetector
    from backend.pipeline.keyframes import sample_keyframes
    from backend.pipeline.track import Box, FrameDetection, GreedyIoUTracker

    video = Path(args.video)
    workdir = Path(args.workdir or f"/tmp/g0_check_{video.stem}")
    if workdir.exists():
        shutil.rmtree(workdir)
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    t0 = time.perf_counter()
    keyframes = sample_keyframes(video, workdir / "kf", target_fps=2.0)
    kf_s = time.perf_counter() - t0
    print(f"keyframes: {len(keyframes)} kept (sample+dedup, {kf_s:.1f}s)")
    if len(keyframes) < 10:
        print("⚠️ 关键帧过少:视频太短或画面几乎静止")

    detector = GroundingDinoDetector(str(Path.home() / "models" / "IDEA-Research__grounding-dino-base"))
    per_frame: list[list[FrameDetection]] = []
    total_dets = 0
    t0 = time.perf_counter()
    for kf in keyframes:
        dets = detector.detect(kf.path, prompts)
        total_dets += len(dets)
        per_frame.append(
            [
                FrameDetection(kf.frame_index, kf.timestamp_ms, Box(*d.box), d.label, d.score)
                for d in dets
            ]
        )
    det_s = time.perf_counter() - t0
    print(f"detections: {total_dets} total, {total_dets / max(len(keyframes),1):.1f}/frame "
          f"({det_s / max(len(keyframes),1):.2f}s/frame)")

    verdict = "REVIEW"
    for name, cfg in (
        ("baseline(iou=0.3,miss=2)", dict(iou_threshold=0.3, max_missed=2)),
        ("relaxed (iou=0.2,miss=4)", dict(iou_threshold=0.2, max_missed=4)),
    ):
        tracker = GreedyIoUTracker(min_track_len=1, **cfg)
        for dets in per_frame:
            tracker.update(dets)
        tracks = tracker.finalize()
        long_tracks = [t for t in tracks if len(t.detections) >= 4]
        short_tracks = [t for t in tracks if 2 <= len(t.detections) < 4]
        fragments = [t for t in tracks if len(t.detections) == 1]
        frag_ratio = len(fragments) / max(len(tracks), 1)
        print(f"\n== {name} ==")
        print(f"tracks: {len(long_tracks)} long(≥4f) / {len(short_tracks)} short(2-3f) / "
              f"{len(fragments)} fragments(1f), frag_ratio={frag_ratio:.0%}")
        print("long tracks:")
        for t in sorted(long_tracks, key=lambda t: -len(t.detections)):
            span_s = (t.detections[-1].timestamp_ms - t.detections[0].timestamp_ms) / 1000
            print(f"  {t.label:20s} {len(t.detections):3d} frames  span {span_s:.1f}s")
        frag_labels = Counter(t.label for t in fragments)
        print(f"fragment labels: {dict(frag_labels.most_common(12))}")
        dup_labels = {k: v for k, v in Counter(t.label for t in long_tracks).items() if v > 1}
        if dup_labels:
            print(f"同标签多长轨(相似对候选): {dup_labels}")
        if len(long_tracks) >= 5 and frag_ratio < 0.5:
            verdict = "PASS"

    print(f"\nG0_CLIP_{verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

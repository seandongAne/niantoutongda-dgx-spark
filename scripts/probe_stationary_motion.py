#!/usr/bin/env python
"""Measure 2fps global camera motion to calibrate real stationary holds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.pipeline.keyframes import Keyframe, select_tiled_keyframes


def _quantiles(values):
    return {
        str(q): round(float(np.quantile(values, q)), 6)
        for q in (0.1, 0.25, 0.5, 0.75, 0.9)
    }


def _runs(values, threshold, target_fps):
    runs = []
    current = 0
    for value in values:
        if value <= threshold:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return {
        "segment_count_ge_2s": sum(run / target_fps >= 2.0 for run in runs),
        "max_run_s": round(max(runs, default=0) / target_fps, 3),
        "transition_count": sum(run for run in runs),
    }


def analyze(path, target_fps):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(source_fps / target_fps))
    previous = None
    flows = []
    differences = []
    frame_index = -1
    sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % step:
            continue
        sampled += 1
        small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                small,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            flows.append(float(np.median(np.linalg.norm(flow, axis=2))))
            differences.append(float(np.abs(small.astype(np.float32) - previous).mean()))
        previous = small
    cap.release()
    motion_frames = [
        Keyframe(
            frame_index=index,
            timestamp_ms=round((index + 1) / target_fps * 1000),
            path=f"probe-{index}",
            motion_score=score,
        )
        for index, score in enumerate(flows)
    ]
    selected, selection_mode = select_tiled_keyframes(
        motion_frames,
        stationary_min_ms=2000,
        adaptive_quantile=0.10,
        adaptive_max_count=12,
        adaptive_min_gap_ms=2000,
    )
    return {
        "path": path,
        "sampled_frames": sampled,
        "target_fps": target_fps,
        "flow_quantiles": _quantiles(flows),
        "mad_quantiles": _quantiles(differences),
        "flow_thresholds": {
            str(value): _runs(flows, value, target_fps)
            for value in (0.15, 0.25, 0.4, 0.6, 0.8, 1.0)
        },
        "mad_thresholds": {
            str(value): _runs(differences, value, target_fps)
            for value in (2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
        },
        "adaptive_selection": {
            "mode": selection_mode,
            "count": len(selected),
            "timestamp_ms": [frame.timestamp_ms for frame in selected],
            "motion_scores": [round(float(frame.motion_score), 6) for frame in selected],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--target-fps", type=float, default=2.0)
    args = parser.parse_args()
    print(
        json.dumps(
            [analyze(path, args.target_fps) for path in args.videos],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

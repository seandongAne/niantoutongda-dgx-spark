"""S1 关键帧采样 — 确定性:均匀降采样 + 静止段去重。

bbox/坐标约定:像素坐标,由 manifest 声明(设计文档 §7.1)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Keyframe:
    frame_index: int  # 源视频帧号
    timestamp_ms: int
    path: str


def sample_keyframes(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    target_fps: float = 2.0,
    diff_threshold: float = 2.0,
    jpeg_quality: int = 92,
) -> list[Keyframe]:
    """均匀采样到 target_fps,再丢弃与上一保留帧几乎相同的帧(灰度 64x64 平均绝对差)。"""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / target_fps))

    kept: list[Keyframe] = []
    last_small: np.ndarray | None = None
    frame_index = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % step != 0:
            continue
        small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        if last_small is not None and float(np.abs(small - last_small).mean()) < diff_threshold:
            continue
        last_small = small
        timestamp_ms = int(frame_index / src_fps * 1000)
        path = out_dir / f"kf_{frame_index:06d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        kept.append(Keyframe(frame_index=frame_index, timestamp_ms=timestamp_ms, path=str(path)))
    cap.release()
    return kept

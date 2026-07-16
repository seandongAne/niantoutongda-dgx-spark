"""S1 关键帧采样 — 确定性:均匀降采样 + 静止段去重。

bbox/坐标约定:像素坐标,由 manifest 声明(设计文档 §7.1)。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Keyframe:
    frame_index: int  # 源视频帧号
    timestamp_ms: int
    path: str
    # 该采样视角之后与其近似静止的持续时间。S2.5 仅对 >=2s 的用户停留
    # 视角做瓦片检测，避免把多尺度成本摊到整段视频。
    stationary_ms: int = 0
    # 与上一个 2fps 采样帧之间的全局中位光流（64x64 像素尺度）。数值越低，
    # 越接近用户停留/对焦；用于真实手持素材没有绝对静止段时的有界降级。
    motion_score: float | None = None


def median_global_motion(previous: np.ndarray, current: np.ndarray) -> float:
    """Estimate global camera motion while ignoring locally moving objects."""

    flow = cv2.calcOpticalFlowFarneback(
        previous.astype(np.uint8),
        current.astype(np.uint8),
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    return float(np.median(np.linalg.norm(flow, axis=2)))


def select_tiled_keyframes(
    keyframes: list[Keyframe],
    *,
    stationary_min_ms: int,
    adaptive_quantile: float = 0.10,
    adaptive_max_count: int = 12,
    adaptive_min_gap_ms: int = 2000,
) -> tuple[list[Keyframe], str]:
    """Select strict stationary holds, or a bounded low-motion fallback.

    The fallback is explicit because real hand-held task-A clips contain no
    reliable two-second absolute holds.  It chooses the lowest-motion quantile,
    applies temporal NMS, and caps tile cost per video.
    """

    if stationary_min_ms < 0 or adaptive_min_gap_ms < 0:
        raise ValueError("stationary/adaptive time thresholds cannot be negative")
    if not 0.0 < adaptive_quantile <= 1.0:
        raise ValueError("adaptive_quantile must be in (0, 1]")
    if adaptive_max_count < 0:
        raise ValueError("adaptive_max_count cannot be negative")

    strict = [frame for frame in keyframes if frame.stationary_ms >= stationary_min_ms]
    if strict:
        return strict, "strict_stationary"
    candidates = [frame for frame in keyframes if frame.motion_score is not None]
    if not candidates or adaptive_max_count == 0:
        return [], "none"
    ranked = sorted(candidates, key=lambda frame: (frame.motion_score, frame.timestamp_ms))
    ranked = ranked[: max(1, ceil(len(ranked) * adaptive_quantile))]
    selected: list[Keyframe] = []
    for frame in ranked:
        if any(
            abs(frame.timestamp_ms - existing.timestamp_ms) < adaptive_min_gap_ms
            for existing in selected
        ):
            continue
        selected.append(frame)
        if len(selected) >= adaptive_max_count:
            break
    selected.sort(key=lambda frame: frame.timestamp_ms)
    return selected, "adaptive_low_motion_fallback"


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
    previous_sample_small: np.ndarray | None = None
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
        motion_score = (
            median_global_motion(previous_sample_small, small)
            if previous_sample_small is not None
            else None
        )
        previous_sample_small = small
        timestamp_ms = int(frame_index / src_fps * 1000)
        if last_small is not None and float(np.abs(small - last_small).mean()) < diff_threshold:
            # 不额外落重复帧，但把用户在这个视角停留了多久记到代表帧上。
            # 后续检测只需看这一帧即可放大小物，仍保持原有去重语义。
            previous = kept[-1]
            stationary_ms = max(previous.stationary_ms, timestamp_ms - previous.timestamp_ms)
            kept[-1] = Keyframe(
                frame_index=previous.frame_index,
                timestamp_ms=previous.timestamp_ms,
                path=previous.path,
                stationary_ms=stationary_ms,
                motion_score=(
                    min(previous.motion_score, motion_score)
                    if previous.motion_score is not None and motion_score is not None
                    else previous.motion_score or motion_score
                ),
            )
            continue
        last_small = small
        path = out_dir / f"kf_{frame_index:06d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        kept.append(
            Keyframe(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                path=str(path),
                motion_score=motion_score,
            )
        )
    cap.release()
    return kept

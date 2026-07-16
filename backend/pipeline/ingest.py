"""S1+S2 编排:视频 → 关键帧 → 检测 → 轨迹 → Observation/Tracklet + 证据裁剪 + 审计。

检测器/嵌入器按协议注入:本地测试用 fake,Spark 上接 detect.GroundingDinoDetector
与 embed.Dinov2Embedder。产物写入 workdir:
  keyframes/            关键帧 jpg
  evidence/             轨迹 Top-K 证据裁剪
  observations.jsonl    每条检测一行(契约 Observation)
  tracklets.jsonl       每条轨迹一行(契约 Tracklet)
  audit-events.jsonl    阶段性审计事件
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import cv2

from backend.pipeline.detect import RawDetection
from backend.pipeline.keyframes import Keyframe, sample_keyframes
from backend.pipeline.track import Box, FrameDetection, GreedyIoUTracker, Track
from backend.schemas.core import AuditEvent, Observation, Tracklet
from backend.tools.audit.store import append_event

PROTOTYPE_TOP_K = 3


class Detector(Protocol):
    model_version: str

    def detect(self, image_path: str, prompts: list[str]) -> list[RawDetection]: ...


class Embedder(Protocol):
    model_version: str

    def embed(self, image_path: str) -> list[float]: ...


@dataclass
class IngestResult:
    video_id: str
    keyframes: list[Keyframe]
    observations: list[Observation]
    tracklets: list[Tracklet]
    workdir: str
    tiled_keyframe_count: int = 0
    detection_elapsed_s: float = 0.0
    frame_batching_used: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _crop(frame_path: str, box: tuple[float, float, float, float], out_path: Path) -> None:
    img = cv2.imread(frame_path)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate box {box} for {frame_path}")
    cv2.imwrite(str(out_path), img[y1:y2, x1:x2])


def _hero_crop_score_image(
    image,
    box: tuple[float, float, float, float],
) -> float:
    """Area x sharpness x completeness score for the S5 representative crop.

    Sharpness is normalized Laplacian variance.  A box touching the outer 2%
    of the frame is treated as truncated; touching two or more edges is
    penalized again.  The score is only compared within one tracklet.
    """

    if image is None or image.size == 0:
        return 0.0
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(box[0])))
    y1 = max(0, min(height, int(box[1])))
    x2 = max(0, min(width, int(box[2])))
    y2 = max(0, min(height, int(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = sharpness_variance / (sharpness_variance + 100.0)
    area = ((x2 - x1) * (y2 - y1)) / float(width * height)

    margin_x, margin_y = max(2, round(width * 0.02)), max(2, round(height * 0.02))
    touched_edges = sum(
        (
            x1 <= margin_x,
            y1 <= margin_y,
            x2 >= width - margin_x,
            y2 >= height - margin_y,
        )
    )
    completeness = 1.0 if touched_edges == 0 else (0.45 if touched_edges == 1 else 0.20)
    return max(0.0, min(1.0, area * sharpness * completeness))


def hero_crop_score(
    frame_path: str | Path,
    box: tuple[float, float, float, float],
) -> float:
    """Public path-based helper kept separate for deterministic unit tests."""

    return _hero_crop_score_image(cv2.imread(str(frame_path)), box)


def ingest_video(
    video_id: str,
    video_path: str | Path,
    prompts: list[str],
    workdir: str | Path,
    detector: Detector,
    embedder: Embedder | None = None,
    *,
    config_version: str,
    target_fps: float = 2.0,
    # 2026-07-15 试机片 A/B:iou 0.2/miss 4 比 0.3/2 少 25% 碎片且缝合断轨,
    # 真实批次到货后随 S3 阈值一起冻结
    iou_threshold: float = 0.2,
    max_missed: int = 4,
    min_track_len: int = 2,
    stationary_min_ms: int = 2000,
    enable_stationary_tiles: bool = True,
) -> IngestResult:
    if stationary_min_ms < 0:
        raise ValueError("stationary_min_ms cannot be negative")
    workdir = Path(workdir)
    (workdir / "evidence").mkdir(parents=True, exist_ok=True)
    audit_path = workdir / "audit-events.jsonl"

    keyframes = sample_keyframes(video_path, workdir / "keyframes", target_fps=target_fps)
    append_event(
        audit_path,
        AuditEvent(
            event_id=f"{video_id}_kf",
            event_type="KeyframesSampled",
            actor="MEM",
            input_refs=[str(video_path)],
            output_refs=[kf.path for kf in keyframes],
            config_version=config_version,
            created_at=_utc_now(),
        ),
    )

    observations: list[Observation] = []
    tracker = GreedyIoUTracker(
        iou_threshold=iou_threshold, max_missed=max_missed, min_track_len=min_track_len
    )
    frame_path_by_index: dict[int, str] = {kf.frame_index: kf.path for kf in keyframes}
    image_paths = [kf.path for kf in keyframes]
    tiled_paths = {
        kf.path
        for kf in keyframes
        if enable_stationary_tiles and kf.stationary_ms >= stationary_min_ms
    }
    detect_many = getattr(detector, "detect_many", None)
    detection_started = time.perf_counter()
    if callable(detect_many):
        raw_by_frame = detect_many(image_paths, prompts, tiled_image_paths=tiled_paths)
        frame_batching_used = True
    else:
        raw_by_frame = [detector.detect(path, prompts) for path in image_paths]
        frame_batching_used = False
    detection_elapsed_s = time.perf_counter() - detection_started
    if len(raw_by_frame) != len(keyframes):
        raise ValueError(
            f"detector returned {len(raw_by_frame)} frame results for {len(keyframes)} keyframes"
        )

    for kf, raw in zip(keyframes, raw_by_frame):
        frame_image = cv2.imread(kf.path) if raw else None
        frame_dets: list[FrameDetection] = []
        for di, d in enumerate(raw):
            ob = Observation(
                observation_id=f"{video_id}_f{kf.frame_index:06d}_d{di:02d}",
                video_id=video_id,
                timestamp_ms=kf.timestamp_ms,
                bbox=d.box,
                crop_ref="",  # 只有进入轨迹 Top-K 的检测才落证据裁剪
                quality=d.score,
                model_version=detector.model_version,
            )
            observations.append(ob)
            frame_dets.append(
                FrameDetection(
                    frame_index=kf.frame_index,
                    timestamp_ms=kf.timestamp_ms,
                    box=Box(*d.box),
                    label=d.label,
                    score=d.score,
                    ref=ob.observation_id,
                    hero_score=_hero_crop_score_image(frame_image, d.box),
                )
            )
        tracker.update(frame_dets)
    append_event(
        audit_path,
        AuditEvent(
            event_id=f"{video_id}_det",
            event_type="DetectionCompleted",
            actor="MEM",
            input_refs=[kf.path for kf in keyframes],
            output_refs=[ob.observation_id for ob in observations],
            config_version=config_version,
            created_at=_utc_now(),
        ),
    )

    tracks: list[Track] = tracker.finalize()
    tracklets: list[Tracklet] = []
    obs_by_id = {ob.observation_id: ob for ob in observations}
    for t in tracks:
        top = sorted(
            t.detections,
            key=lambda d: (-d.hero_score, -d.score, d.frame_index),
        )[:PROTOTYPE_TOP_K]
        prototype_refs: list[str] = []
        for det in top:
            crop_path = workdir / "evidence" / f"{video_id}_t{t.track_id:03d}_f{det.frame_index:06d}.jpg"
            _crop(
                frame_path_by_index[det.frame_index],
                (det.box.x1, det.box.y1, det.box.x2, det.box.y2),
                crop_path,
            )
            prototype_refs.append(str(crop_path))
            obs_by_id[det.ref].crop_ref = str(crop_path)

        embedding_ref = None
        if embedder is not None and prototype_refs:
            vectors = [embedder.embed(p) for p in prototype_refs]
            mean = [sum(col) / len(col) for col in zip(*vectors)]
            norm = max(sum(v * v for v in mean) ** 0.5, 1e-12)
            emb_path = workdir / "evidence" / f"{video_id}_t{t.track_id:03d}_emb.json"
            emb_path.write_text(
                json.dumps({"model": embedder.model_version, "vector": [v / norm for v in mean]})
            )
            embedding_ref = str(emb_path)

        obs_ids = [det.ref for det in t.detections]
        tracklets.append(
            Tracklet(
                tracklet_id=f"{video_id}_t{t.track_id:03d}",
                video_id=video_id,
                observation_ids=obs_ids,
                prototype_refs=prototype_refs,
                embedding_ref=embedding_ref,
                attributes={
                    "label": t.label,
                    "hero_ref": prototype_refs[0] if prototype_refs else "",
                    "hero_score": f"{top[0].hero_score:.8f}" if top else "0.00000000",
                    "hero_scoring_version": "area-sharpness-completeness-v1",
                },
            )
        )
    append_event(
        audit_path,
        AuditEvent(
            event_id=f"{video_id}_trk",
            event_type="TrackletsFormed",
            actor="MEM",
            input_refs=[ob.observation_id for ob in observations],
            output_refs=[t.tracklet_id for t in tracklets],
            config_version=config_version,
            created_at=_utc_now(),
        ),
    )

    with open(workdir / "observations.jsonl", "w") as f:
        for ob in observations:
            f.write(ob.model_dump_json() + "\n")
    with open(workdir / "tracklets.jsonl", "w") as f:
        for t in tracklets:
            f.write(t.model_dump_json() + "\n")

    return IngestResult(
        video_id=video_id,
        keyframes=keyframes,
        observations=observations,
        tracklets=tracklets,
        workdir=str(workdir),
        tiled_keyframe_count=len(tiled_paths),
        detection_elapsed_s=detection_elapsed_s,
        frame_batching_used=frame_batching_used,
    )

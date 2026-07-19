"""静态场景的邻域拓扑证据。

该模块只从 ingest 的检测框与词表构建证据，不读取人工 anchor GT。核心假设是
同一批视频拍摄的是未移动的物品：目标外观可能模糊，但目标附近稳定出现的、
单实例类别可以组成一个局部空间指纹。

二维方向会随相机视角改变，因此这里只使用较稳健的共现、归一化距离和邻近
排序。所有分数都允许缺失；证据不足时调用方必须保持原 ReID 排序。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

from backend.pipeline.vocab import Vocabulary
from backend.schemas.core import Observation, Tracklet


BBox = tuple[float, float, float, float]
FrameKey = tuple[str, int]
CONTEXT_FORMAT_VERSION = "reid-neighborhood-context-v1"


@dataclass(frozen=True)
class GeometryObservation:
    timestamp_ms: int
    bbox: BBox
    quality: float


@dataclass(frozen=True)
class TrackGeometry:
    tracklet_id: str
    video_id: str
    category_id: str | None
    observations: tuple[GeometryObservation, ...]


@dataclass(frozen=True)
class FrameDetection:
    tracklet_id: str
    category_id: str
    bbox: BBox
    quality: float


@dataclass(frozen=True)
class NeighborRelation:
    anchor_category: str
    co_visibility: float
    normalized_distance: float
    normalized_rank: float
    common_frames: int


@dataclass(frozen=True)
class NeighborhoodSignature:
    tracklet_id: str
    video_id: str
    relations: tuple[NeighborRelation, ...]

    @property
    def by_anchor(self) -> dict[str, NeighborRelation]:
        return {relation.anchor_category: relation for relation in self.relations}


@dataclass(frozen=True)
class ContextEvidence:
    score: float
    shared_anchors: tuple[str, ...]
    overlap: float
    relation_agreement: float


def _read_jsonl(path: Path, model_type):
    if not path.exists():
        return []
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_track_geometries(
    ingest_root: str | Path,
    *,
    vocab: Vocabulary,
) -> dict[str, TrackGeometry]:
    """读取原始 tracklet 的逐帧框；未知类别保留但不会成为邻域锚点。"""

    root = Path(ingest_root)
    if not root.is_dir():
        raise FileNotFoundError(f"ingest root not found: {root}")
    geometries: dict[str, TrackGeometry] = {}
    for tracklet_path in sorted(root.glob("*/tracklets.jsonl")):
        observations = {
            observation.observation_id: observation
            for observation in _read_jsonl(
                tracklet_path.parent / "observations.jsonl", Observation
            )
        }
        for tracklet in _read_jsonl(tracklet_path, Tracklet):
            if tracklet.tracklet_id in geometries:
                raise ValueError(f"duplicate tracklet geometry: {tracklet.tracklet_id}")
            label = str(tracklet.attributes.get("label", ""))
            match = vocab.match(label)
            linked = [
                observations[observation_id]
                for observation_id in tracklet.observation_ids
                if observation_id in observations
            ]
            geometries[tracklet.tracklet_id] = TrackGeometry(
                tracklet_id=tracklet.tracklet_id,
                video_id=tracklet.video_id,
                category_id=match.category_id,
                observations=tuple(
                    GeometryObservation(
                        timestamp_ms=item.timestamp_ms,
                        bbox=item.bbox,
                        quality=item.quality,
                    )
                    for item in sorted(
                        linked, key=lambda item: (item.timestamp_ms, item.observation_id)
                    )
                ),
            )
    if not geometries:
        raise ValueError("ingest contains no tracklet geometries")
    return geometries


def load_stitch_groups(path: str | Path) -> dict[str, tuple[str, ...]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = raw.get("groups") or {}
    result: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for representative, members in sorted(groups.items()):
        normalized = tuple(sorted(str(member) for member in members))
        if not normalized or str(representative) not in normalized:
            raise ValueError(f"invalid stitch group for {representative}")
        overlap = seen & set(normalized)
        if overlap:
            raise ValueError(f"stitch groups overlap: {sorted(overlap)}")
        seen.update(normalized)
        result[str(representative)] = normalized
    return result


def collapse_stitched_geometries(
    geometries: Mapping[str, TrackGeometry],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, TrackGeometry]:
    """增加 stitch representative 几何；原始几何仍保留供代理判卷。"""

    result = dict(geometries)
    for representative, member_ids in sorted(groups.items()):
        members = [geometries[member_id] for member_id in member_ids]
        videos = {member.video_id for member in members}
        categories = {member.category_id for member in members}
        if len(videos) != 1:
            raise ValueError(f"stitch group spans videos: {representative}")
        if len(categories) != 1:
            raise ValueError(f"stitch group spans categories: {representative}")
        observations = sorted(
            (observation for member in members for observation in member.observations),
            key=lambda item: (item.timestamp_ms, -item.quality, item.bbox),
        )
        result[representative] = TrackGeometry(
            tracklet_id=representative,
            video_id=members[0].video_id,
            category_id=members[0].category_id,
            observations=tuple(observations),
        )
    return result


def bbox_iou(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _nms(detections: Sequence[FrameDetection], *, iou_threshold: float) -> list[FrameDetection]:
    kept: list[FrameDetection] = []
    for detection in sorted(
        detections, key=lambda item: (-item.quality, item.tracklet_id, item.bbox)
    ):
        if all(bbox_iou(detection.bbox, item.bbox) < iou_threshold for item in kept):
            kept.append(detection)
    return kept


def build_frame_index(
    geometries: Mapping[str, TrackGeometry],
) -> dict[FrameKey, tuple[FrameDetection, ...]]:
    """只用原始 tracklet 建帧索引，避免 stitch representative 重复计数。"""

    rows: dict[FrameKey, list[FrameDetection]] = defaultdict(list)
    for geometry in geometries.values():
        if not geometry.category_id:
            continue
        for observation in geometry.observations:
            rows[(geometry.video_id, observation.timestamp_ms)].append(
                FrameDetection(
                    tracklet_id=geometry.tracklet_id,
                    category_id=geometry.category_id,
                    bbox=observation.bbox,
                    quality=observation.quality,
                )
            )
    return {
        key: tuple(sorted(value, key=lambda item: (item.category_id, item.tracklet_id)))
        for key, value in sorted(rows.items())
    }


def select_anchor_categories(
    frame_index: Mapping[FrameKey, Sequence[FrameDetection]],
    *,
    videos: Iterable[str],
    min_visible_frames: int = 5,
    min_single_fraction: float = 0.80,
    nms_iou: float = 0.50,
) -> tuple[frozenset[str], dict[str, dict[str, float | int]]]:
    """选出每段视频都稳定呈单实例的类别，作为无 GT 环境锚点。"""

    if min_visible_frames < 1:
        raise ValueError("min_visible_frames must be positive")
    if not 0 <= min_single_fraction <= 1:
        raise ValueError("min_single_fraction must be in [0, 1]")
    video_ids = tuple(sorted(set(videos)))
    counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    categories: set[str] = set()
    for (video_id, _), detections in frame_index.items():
        by_category: dict[str, list[FrameDetection]] = defaultdict(list)
        for detection in detections:
            by_category[detection.category_id].append(detection)
        for category, values in by_category.items():
            categories.add(category)
            counts[(video_id, category)].append(
                len(_nms(values, iou_threshold=nms_iou))
            )

    diagnostics: dict[str, dict[str, float | int]] = {}
    eligible: set[str] = set()
    for category in sorted(categories):
        per_video_ok = True
        row: dict[str, float | int] = {}
        for video_id in video_ids:
            values = counts.get((video_id, category), [])
            visible = len(values)
            single_fraction = (
                sum(value == 1 for value in values) / visible if visible else 0.0
            )
            row[f"{video_id}_visible_frames"] = visible
            row[f"{video_id}_single_fraction"] = round(single_fraction, 6)
            per_video_ok &= (
                visible >= min_visible_frames
                and single_fraction >= min_single_fraction
            )
        diagnostics[category] = row
        if per_video_ok:
            eligible.add(category)
    return frozenset(eligible), diagnostics


def _center_distance(a: BBox, b: BBox, frame_size: tuple[int, int]) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    diagonal = math.hypot(*frame_size)
    return math.hypot(ax - bx, ay - by) / diagonal if diagonal > 0 else 0.0


def build_signatures(
    geometries: Mapping[str, TrackGeometry],
    frame_index: Mapping[FrameKey, Sequence[FrameDetection]],
    *,
    anchor_categories: frozenset[str],
    frame_sizes: Mapping[str, tuple[int, int]],
    max_neighbors: int = 5,
    min_common_frames: int = 2,
    max_target_overlap: float = 0.30,
    nms_iou: float = 0.50,
) -> dict[str, NeighborhoodSignature]:
    if max_neighbors < 1 or min_common_frames < 1:
        raise ValueError("max_neighbors/min_common_frames must be positive")
    signatures: dict[str, NeighborhoodSignature] = {}
    for tracklet_id, geometry in sorted(geometries.items()):
        if geometry.video_id not in frame_sizes:
            raise ValueError(f"missing frame size for {geometry.video_id}")
        # stitch 产物理论上不共帧；仍按时间保留最高质量框以 fail-stable。
        target_by_time: dict[int, GeometryObservation] = {}
        for observation in geometry.observations:
            previous = target_by_time.get(observation.timestamp_ms)
            if previous is None or observation.quality > previous.quality:
                target_by_time[observation.timestamp_ms] = observation
        samples: dict[str, list[float]] = defaultdict(list)
        for timestamp, target in sorted(target_by_time.items()):
            detections = frame_index.get((geometry.video_id, timestamp), ())
            by_category: dict[str, list[FrameDetection]] = defaultdict(list)
            for detection in detections:
                if (
                    detection.category_id in anchor_categories
                    and detection.category_id != geometry.category_id
                    and bbox_iou(target.bbox, detection.bbox) < max_target_overlap
                ):
                    by_category[detection.category_id].append(detection)
            for category, values in by_category.items():
                candidates = _nms(values, iou_threshold=nms_iou)
                if not candidates:
                    continue
                distance = min(
                    _center_distance(target.bbox, candidate.bbox, frame_sizes[geometry.video_id])
                    for candidate in candidates
                )
                samples[category].append(distance)

        relations = []
        observation_count = max(1, len(target_by_time))
        for category, distances in samples.items():
            if len(distances) < min_common_frames:
                continue
            relations.append(
                (
                    category,
                    len(distances) / observation_count,
                    median(distances),
                    len(distances),
                )
            )
        relations.sort(key=lambda item: (item[2], -item[1], item[0]))
        selected = relations[:max_neighbors]
        rank_denominator = max(1, len(selected) - 1)
        signatures[tracklet_id] = NeighborhoodSignature(
            tracklet_id=tracklet_id,
            video_id=geometry.video_id,
            relations=tuple(
                NeighborRelation(
                    anchor_category=category,
                    co_visibility=round(co_visibility, 8),
                    normalized_distance=round(distance, 8),
                    normalized_rank=round(index / rank_denominator, 8),
                    common_frames=common_frames,
                )
                for index, (category, co_visibility, distance, common_frames) in enumerate(selected)
            ),
        )
    return signatures


def compare_signatures(
    left: NeighborhoodSignature,
    right: NeighborhoodSignature,
    *,
    min_shared_anchors: int = 2,
    distance_tolerance: float = 0.25,
) -> ContextEvidence | None:
    """比较两个局部图；证据不足返回 None，绝不制造 0.5 伪观测。"""

    if left.video_id == right.video_id:
        return None
    if min_shared_anchors < 1 or distance_tolerance <= 0:
        raise ValueError("invalid context comparison thresholds")
    left_by_anchor, right_by_anchor = left.by_anchor, right.by_anchor
    shared = tuple(sorted(set(left_by_anchor) & set(right_by_anchor)))
    if len(shared) < min_shared_anchors:
        return None
    union = set(left_by_anchor) | set(right_by_anchor)
    overlap = len(shared) / len(union) if union else 0.0
    agreements = []
    for anchor in shared:
        a, b = left_by_anchor[anchor], right_by_anchor[anchor]
        distance_agreement = max(
            0.0,
            1.0 - abs(a.normalized_distance - b.normalized_distance) / distance_tolerance,
        )
        rank_agreement = max(0.0, 1.0 - abs(a.normalized_rank - b.normalized_rank))
        visibility_agreement = max(0.0, 1.0 - abs(a.co_visibility - b.co_visibility))
        agreements.append(
            0.50 * distance_agreement
            + 0.30 * rank_agreement
            + 0.20 * visibility_agreement
        )
    relation_agreement = sum(agreements) / len(agreements)
    score = 0.50 * overlap + 0.50 * relation_agreement
    return ContextEvidence(
        score=max(0.0, min(1.0, score)),
        shared_anchors=shared,
        overlap=overlap,
        relation_agreement=relation_agreement,
    )

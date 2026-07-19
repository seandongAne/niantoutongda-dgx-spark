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
LEGACY_CONTEXT_FORMAT_VERSION = "reid-neighborhood-context-v1"
CONTEXT_FORMAT_VERSION = "reid-neighborhood-context-v2"
SUPPORTED_CONTEXT_FORMAT_VERSIONS = frozenset(
    {LEGACY_CONTEXT_FORMAT_VERSION, CONTEXT_FORMAT_VERSION}
)


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
    local_scale_distance: float | None = None
    distance_mad: float = 0.0
    local_distance_mad: float | None = None
    local_scale_coverage: float = 0.0


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
    confidence: float
    support: float


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


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _center_distance(a: BBox, b: BBox, frame_size: tuple[int, int]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    diagonal = math.hypot(*frame_size)
    return math.hypot(ax - bx, ay - by) / diagonal if diagonal > 0 else 0.0


def _frame_anchor_scale(
    by_category: Mapping[str, Sequence[FrameDetection]],
    *,
    frame_size: tuple[int, int],
    nms_iou: float,
) -> float | None:
    """用同帧稳定锚点间距估计局部缩放，削弱相机远近变化。"""

    representatives = []
    for values in by_category.values():
        candidates = _nms(values, iou_threshold=nms_iou)
        if candidates:
            representatives.append(candidates[0])
    if len(representatives) < 2:
        return None
    diagonal = math.hypot(*frame_size)
    if diagonal <= 0:
        return None
    distances = []
    for index, left in enumerate(representatives):
        lx, ly = _bbox_center(left.bbox)
        for right in representatives[index + 1 :]:
            rx, ry = _bbox_center(right.bbox)
            distance = math.hypot(lx - rx, ly - ry) / diagonal
            if distance > 1e-9:
                distances.append(distance)
    return median(distances) if distances else None


def _median_absolute_deviation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return median([abs(value - center) for value in values])


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
        samples: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
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
            local_scale = _frame_anchor_scale(
                by_category,
                frame_size=frame_sizes[geometry.video_id],
                nms_iou=nms_iou,
            )
            for category, values in by_category.items():
                candidates = _nms(values, iou_threshold=nms_iou)
                if not candidates:
                    continue
                distance = min(
                    _center_distance(target.bbox, candidate.bbox, frame_sizes[geometry.video_id])
                    for candidate in candidates
                )
                # d / scale 对相机 zoom 更稳；x/(1+x) 把无界比值压到 [0,1)。
                local_distance = None
                if local_scale is not None and local_scale > 1e-9:
                    ratio = distance / local_scale
                    local_distance = ratio / (1.0 + ratio)
                samples[category].append((distance, local_distance))

        relations = []
        observation_count = max(1, len(target_by_time))
        for category, rows in samples.items():
            if len(rows) < min_common_frames:
                continue
            distances = [distance for distance, _ in rows]
            local_distances = [
                distance for _, distance in rows if distance is not None
            ]
            relations.append(
                (
                    category,
                    len(rows) / observation_count,
                    median(distances),
                    len(rows),
                    median(local_distances) if local_distances else None,
                    _median_absolute_deviation(distances),
                    (
                        _median_absolute_deviation(local_distances)
                        if local_distances
                        else None
                    ),
                    len(local_distances) / len(rows),
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
                    local_scale_distance=(
                        round(local_distance, 8)
                        if local_distance is not None
                        else None
                    ),
                    distance_mad=round(distance_mad, 8),
                    local_distance_mad=(
                        round(local_distance_mad, 8)
                        if local_distance_mad is not None
                        else None
                    ),
                    local_scale_coverage=round(local_scale_coverage, 8),
                )
                for index, (
                    category,
                    co_visibility,
                    distance,
                    common_frames,
                    local_distance,
                    distance_mad,
                    local_distance_mad,
                    local_scale_coverage,
                ) in enumerate(selected)
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
    supports = []
    for anchor in shared:
        a, b = left_by_anchor[anchor], right_by_anchor[anchor]
        distance_agreement = max(
            0.0,
            1.0 - abs(a.normalized_distance - b.normalized_distance) / distance_tolerance,
        )
        rank_agreement = max(0.0, 1.0 - abs(a.normalized_rank - b.normalized_rank))
        visibility_agreement = max(0.0, 1.0 - abs(a.co_visibility - b.co_visibility))
        weighted = [
            (0.55, distance_agreement),
            (0.25, rank_agreement),
            (0.20, visibility_agreement),
        ]
        if a.local_scale_distance is not None and b.local_scale_distance is not None:
            local_agreement = max(
                0.0,
                1.0
                - abs(a.local_scale_distance - b.local_scale_distance)
                / distance_tolerance,
            )
            local_coverage = math.sqrt(
                a.local_scale_coverage * b.local_scale_coverage
            )
            # 只在两侧都有局部尺度时让渡最多 25% 权重。
            local_weight = 0.25 * local_coverage
            weighted = [
                (weight * (1.0 - local_weight), value)
                for weight, value in weighted
            ]
            weighted.append((local_weight, local_agreement))
        agreement = sum(weight * value for weight, value in weighted) / sum(
            weight for weight, _ in weighted
        )

        stability_a = 1.0 / (1.0 + a.distance_mad / distance_tolerance)
        stability_b = 1.0 / (1.0 + b.distance_mad / distance_tolerance)
        if a.local_distance_mad is not None:
            stability_a *= 1.0 / (
                1.0 + a.local_distance_mad / distance_tolerance
            )
        if b.local_distance_mad is not None:
            stability_b *= 1.0 / (
                1.0 + b.local_distance_mad / distance_tolerance
            )
        stability = math.sqrt(stability_a * stability_b)
        visibility_support = math.sqrt(a.co_visibility * b.co_visibility)
        frame_support = min(1.0, min(a.common_frames, b.common_frames) / 3.0)
        support = (stability * visibility_support * frame_support) ** (1.0 / 3.0)
        agreements.append((agreement, support))
        supports.append(support)

    support_total = sum(supports)
    relation_agreement = (
        sum(agreement * support for agreement, support in agreements) / support_total
        if support_total > 0
        else 0.0
    )
    support = sum(supports) / len(supports)
    # 一个锚点只能提供弱证据；达到三个共同锚点后不再额外放大置信度。
    confidence = min(1.0, len(shared) / 3.0) * support
    score = 0.50 * overlap + 0.50 * relation_agreement
    return ContextEvidence(
        score=max(0.0, min(1.0, score)),
        shared_anchors=shared,
        overlap=overlap,
        relation_agreement=relation_agreement,
        confidence=max(0.0, min(1.0, confidence)),
        support=max(0.0, min(1.0, support)),
    )

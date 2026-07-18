#!/usr/bin/env python
"""Build automatic spatial-anchor hypotheses with the Spark-local Nemotron VLM.

The command consumes only automatic ingest artifacts and the configured anchor
vocabulary.  It never accepts a hand-authored region manifest, visual review,
candidate override, or manual track mapping.  For every automatic furniture
track it classifies up to three independent target-only crops, aggregates real
view votes and emits the strict assignment contract consumed by
``space_task.py``.  A separate contact sheet remains audit evidence only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import sys
import threading
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.schemas.core import Tracklet  # noqa: E402
from backend.tools.spatial import (  # noqa: E402
    AutomaticAnchorCandidate,
    PowerState,
    SpatialObservation,
    load_observations_jsonl,
)

CLASSIFIER_SCHEMA_VERSION = "1.0"
CLASSIFIER_VERSION = "space-anchor-nemotron-v8"
VISUAL_INSTANCE_VERSION = "spatiotemporal-semantic-v3"
THIN_SHELF_CALIBRATION_VERSION = "thin-shelf-geometry-v1"
MAIN_MAX_TOKENS = 700
MAX_CLASSIFICATION_VIEWS = 3
MIN_VALID_CLASSIFICATION_VIEWS = 2
MIN_TARGET_VIEW_PIXELS = 20_000
LONG_GAP_SEMANTIC_MAX_MS = 120_000
LONG_GAP_SEMANTIC_MIN_COSINE = 0.60
LONG_GAP_SEMANTIC_MIN_CONFIDENCE = 0.85
ANCHOR_CANDIDATES_FILENAME = "anchor_candidates.json"
METRICS_FILENAME = "metrics.json"
HASHES_FILENAME = "hashes.json"
DEFAULT_MODEL = (
    "/models/nv-community__NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"
)

ANCHOR_DISPLAY_ZH = {
    "study_desk": "学习桌面",
    "vanity": "花布面桌台面",
    "wall_shelf": "墙面置物架",
    "chest_of_drawers": "斗柜台面",
    "display_cabinet": "展示柜层板",
}

DEFAULT_ANCHOR_DESCRIPTIONS = {
    "study_desk": "a writing or study desk work surface",
    "vanity": "a vanity or narrow console-table top used as the requested vanity surface",
    "wall_shelf": "a wall-mounted floating shelf",
    "chest_of_drawers": "the usable top or body of a chest of drawers or dresser",
    "display_cabinet": "a glass-door display cabinet and its usable shelves",
}

_FRAME_RE = re.compile(r"(?:^|_)f(?P<index>\d+)(?:_|\.|$)")
_print_lock = threading.Lock()


@dataclass(frozen=True)
class TrackEvidence:
    track_id: str
    observations: tuple[SpatialObservation, ...]
    prototype_refs: tuple[str, ...]
    hero_ref: str | None
    visual_instance_id: str


@dataclass(frozen=True)
class TrackEmbedding:
    model: str
    vector: tuple[float, ...]


def _canonical_anchor(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold().strip()).strip("_")


def _expected_anchors(values: Sequence[str]) -> list[str]:
    result = [
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    ]
    canonical = [_canonical_anchor(item) for item in result]
    if not result:
        raise ValueError("at least one --expected-anchor is required")
    if any(not item for item in canonical):
        raise ValueError("expected anchors must contain letters or numbers")
    if len(canonical) != len(set(canonical)):
        raise ValueError("expected anchors contain duplicates")
    return [value for _, value in sorted(zip(canonical, result, strict=True))]


def _json_bytes(value: object, *, indent: int | None = None) -> bytes:
    separators = None if indent is not None else (",", ":")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=separators,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJ / path


def _load_tracklets(path: Path) -> dict[str, Tracklet]:
    result: dict[str, Tracklet] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                tracklet = Tracklet.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if tracklet.tracklet_id in result:
                raise ValueError(f"duplicate tracklet_id: {tracklet.tracklet_id}")
            result[tracklet.tracklet_id] = tracklet
    return result


def _load_track_embeddings(
    tracklets: Mapping[str, Tracklet],
) -> dict[str, TrackEmbedding]:
    """Load only strictly valid, finite, normalized automatic DINO vectors."""

    result: dict[str, TrackEmbedding] = {}
    expected_dimension: int | None = None
    for track_id in sorted(tracklets):
        ref = str(tracklets[track_id].embedding_ref or "").strip()
        if not ref:
            continue
        path = _resolve_path(ref)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            model = str(payload["model"]).strip()
            vector = tuple(float(value) for value in payload["vector"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not model or not vector or not all(math.isfinite(value) for value in vector):
            continue
        if expected_dimension is None:
            expected_dimension = len(vector)
        if len(vector) != expected_dimension:
            continue
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            continue
        result[track_id] = TrackEmbedding(model=model, vector=vector)
    return result


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _bbox_iom(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    smaller_area = min(
        (left[2] - left[0]) * (left[3] - left[1]),
        (right[2] - right[0]) * (right[3] - right[1]),
    )
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _bbox_center_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    scale = max(
        1.0,
        math.hypot(left[2] - left[0], left[3] - left[1]),
        math.hypot(right[2] - right[0], right[3] - right[1]),
    )
    return math.hypot(
        left_center[0] - right_center[0], left_center[1] - right_center[1]
    ) / scale


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _embedding_cosine(
    left: TrackEmbedding | None, right: TrackEmbedding | None
) -> float | None:
    if (
        left is None
        or right is None
        or left.model != right.model
        or len(left.vector) != len(right.vector)
    ):
        return None
    return math.fsum(a * b for a, b in zip(left.vector, right.vector, strict=True))


def _center(observation: SpatialObservation) -> tuple[float, float]:
    assert observation.bbox is not None
    x1, y1, x2, y2 = observation.bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _predict_center(
    observations: Sequence[SpatialObservation], target_ms: int
) -> tuple[float, float]:
    points = [(item.timestamp_ms, *_center(item)) for item in observations]
    if len(points) == 1 or len({item[0] for item in points}) == 1:
        return points[-1][1], points[-1][2]
    mean_time = math.fsum(item[0] for item in points) / len(points)
    denominator = math.fsum((item[0] - mean_time) ** 2 for item in points)

    def predict(index: int) -> float:
        mean_value = math.fsum(item[index] for item in points) / len(points)
        slope = math.fsum(
            (item[0] - mean_time) * (item[index] - mean_value) for item in points
        ) / denominator
        return mean_value + slope * (target_ms - mean_time)

    return predict(1), predict(2)


def _motion_error(
    history: Sequence[SpatialObservation], target: SpatialObservation
) -> float:
    predicted = _predict_center(history, target.timestamp_ms)
    actual = _center(target)
    assert target.bbox is not None
    diagonal = max(
        1.0,
        math.hypot(
            target.bbox[2] - target.bbox[0], target.bbox[3] - target.bbox[1]
        ),
    )
    return math.hypot(predicted[0] - actual[0], predicted[1] - actual[1]) / diagonal


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        first, second = sorted((root_left, root_right))
        self.parent[second] = first


def automatic_visual_instance_ids(
    grouped: Mapping[str, Sequence[SpatialObservation]],
    *,
    embeddings: Mapping[str, TrackEmbedding] | None = None,
    min_shared_frames: int = 2,
    min_median_iou: float = 0.80,
    max_short_gap_ms: int = 1_500,
) -> dict[str, str]:
    """Group parallel, nested and short-gap fragments of one physical object.

    Existing high-IoU grouping remains available without embeddings.  More
    permissive nested/continuation edges fail closed unless the automatic DINO
    vectors and conservative temporal geometry agree.
    """

    if min_shared_frames < 1:
        raise ValueError("min_shared_frames must be positive")
    if not 0.0 <= min_median_iou <= 1.0:
        raise ValueError("min_median_iou must be in [0, 1]")
    if max_short_gap_ms < 0:
        raise ValueError("max_short_gap_ms must be non-negative")
    embeddings = embeddings or {}
    track_ids = sorted(grouped)
    union = _UnionFind(track_ids)
    frame_boxes: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    track_rows: dict[str, list[SpatialObservation]] = {}
    track_labels: dict[str, str] = {}
    track_videos: dict[str, str] = {}
    tracks_by_frame: dict[str, list[str]] = defaultdict(list)
    for track_id in track_ids:
        rows = sorted(
            (item for item in grouped[track_id] if item.bbox is not None),
            key=lambda item: (item.timestamp_ms, item.frame_ref),
        )
        track_rows[track_id] = rows
        labels = {_canonical_anchor(item.anchor_label) for item in rows}
        videos = {item.video_id for item in rows}
        track_labels[track_id] = next(iter(labels)) if len(labels) == 1 else ""
        track_videos[track_id] = next(iter(videos)) if len(videos) == 1 else ""
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for observation in rows:
            assert observation.bbox is not None
            boxes[observation.frame_ref] = observation.bbox
        frame_boxes[track_id] = boxes
        for frame_ref in boxes:
            tracks_by_frame[frame_ref].append(track_id)

    overlaps: dict[
        tuple[str, str], list[tuple[float, float, float]]
    ] = defaultdict(list)
    for frame_ref in sorted(tracks_by_frame):
        frame_tracks = sorted(set(tracks_by_frame[frame_ref]))
        for left_index, left in enumerate(frame_tracks):
            for right in frame_tracks[left_index + 1 :]:
                left_box = frame_boxes[left][frame_ref]
                right_box = frame_boxes[right][frame_ref]
                overlaps[(left, right)].append(
                    (
                        _bbox_iou(left_box, right_box),
                        _bbox_iom(left_box, right_box),
                        _bbox_center_distance(left_box, right_box),
                    )
                )

    cannot_link: set[tuple[str, str]] = set()
    edges: list[tuple[int, float, str, str]] = []
    for (left, right), values in sorted(overlaps.items()):
        ious = [item[0] for item in values]
        ioms = [item[1] for item in values]
        distances = [item[2] for item in values]
        if (
            len(values) >= 2
            and sum(value < 0.20 for value in ioms) / len(values) >= 0.80
            and _median(distances) > 0.25
        ):
            cannot_link.add((left, right))
            continue
        high_ious = [value for value in ious if value >= min_median_iou]
        if len(high_ious) >= min_shared_frames:
            edges.append((0, -_median(high_ious), left, right))
            continue
        cosine = _embedding_cosine(embeddings.get(left), embeddings.get(right))
        if (
            len(values) >= 3
            and _median(ioms) >= 0.95
            and sum(value >= 0.85 for value in ioms) / len(values) >= 0.80
            and _median(distances) <= 0.03
            and sum(value <= 0.05 for value in distances) / len(values) >= 0.80
            and cosine is not None
            and cosine >= 0.70
        ):
            edges.append((1, -cosine, left, right))

    for left_index, left in enumerate(track_ids):
        for right in track_ids[left_index + 1 :]:
            if (left, right) in overlaps:
                continue
            if (
                not track_rows[left]
                or not track_rows[right]
                or not track_videos[left]
                or track_videos[left] != track_videos[right]
                or not track_labels[left]
                or track_labels[left] != track_labels[right]
            ):
                continue
            left_first, left_last = (
                track_rows[left][0].timestamp_ms,
                track_rows[left][-1].timestamp_ms,
            )
            right_first, right_last = (
                track_rows[right][0].timestamp_ms,
                track_rows[right][-1].timestamp_ms,
            )
            if left_last < right_first:
                earlier, later = left, right
                gap = right_first - left_last
            elif right_last < left_first:
                earlier, later = right, left
                gap = left_first - right_last
            else:
                continue
            if gap > max_short_gap_ms:
                continue
            cosine = _embedding_cosine(embeddings.get(earlier), embeddings.get(later))
            if cosine is None or cosine < 0.35:
                continue
            early_rows, late_rows = track_rows[earlier], track_rows[later]
            if len(early_rows) < 3 or len(late_rows) < 3:
                continue
            early_box = early_rows[-1].bbox
            late_box = late_rows[0].bbox
            assert early_box is not None and late_box is not None
            early_width, early_height = (
                early_box[2] - early_box[0],
                early_box[3] - early_box[1],
            )
            late_width, late_height = (
                late_box[2] - late_box[0],
                late_box[3] - late_box[1],
            )
            early_aspect, late_aspect = early_width / early_height, late_width / late_height
            aspect_factor = max(early_aspect, late_aspect) / min(
                early_aspect, late_aspect
            )
            early_area, late_area = early_width * early_height, late_width * late_height
            area_factor = max(early_area, late_area) / min(early_area, late_area)
            forward_error = _motion_error(early_rows[-3:], late_rows[0])
            backward_error = _motion_error(late_rows[:3], early_rows[-1])
            if (
                aspect_factor <= 1.25
                and area_factor <= 1.50
                and forward_error <= 0.15
                and backward_error <= 0.30
            ):
                edges.append((2, -cosine, earlier, later))

    def can_union(left: str, right: str) -> bool:
        left_root, right_root = union.find(left), union.find(right)
        if left_root == right_root:
            return False
        left_members = {item for item in track_ids if union.find(item) == left_root}
        right_members = {item for item in track_ids if union.find(item) == right_root}
        return not any(
            tuple(sorted((a, b))) in cannot_link
            for a in left_members
            for b in right_members
        )

    for _, _, left, right in sorted(edges):
        if can_union(left, right):
            union.union(left, right)

    members: dict[str, list[str]] = defaultdict(list)
    for track_id in track_ids:
        members[union.find(track_id)].append(track_id)
    instance_by_track: dict[str, str] = {}
    for items in sorted((sorted(value) for value in members.values()), key=lambda x: x[0]):
        digest = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]
        instance_id = f"auto_visual_{digest}"
        for track_id in items:
            instance_by_track[track_id] = instance_id
    return instance_by_track


def semantic_visual_instance_ids(
    grouped: Mapping[str, Sequence[SpatialObservation]],
    candidates: Sequence[AutomaticAnchorCandidate],
    initial_ids: Mapping[str, str],
    *,
    embeddings: Mapping[str, TrackEmbedding] | None = None,
    max_gap_ms: int = 10_000,
) -> dict[str, str]:
    """Consolidate part/whole fragments only after independent semantic votes agree."""

    embeddings = embeddings or {}
    track_ids = sorted(grouped)
    if set(initial_ids) != set(track_ids):
        raise ValueError("initial visual instance IDs must cover every track")
    union = _UnionFind(track_ids)
    by_initial: dict[str, list[str]] = defaultdict(list)
    for track_id in track_ids:
        by_initial[initial_ids[track_id]].append(track_id)
    for members in by_initial.values():
        for track_id in members[1:]:
            union.union(members[0], track_id)

    dominant: dict[str, tuple[str, str, str]] = {}
    strong_consensus: set[str] = set()
    for candidate in candidates:
        if len(candidate.source_track_ids) != 1:
            continue
        track_id = candidate.source_track_ids[0]
        hypotheses = [
            item
            for item in candidate.anchor_hypotheses
            if item.label_vote_count >= 2
            and item.mean_confidence >= 0.70
            and item.support_type is not None
            and item.capacity_class is not None
        ]
        if not hypotheses:
            continue
        winner = min(
            hypotheses,
            key=lambda item: (
                -item.label_vote_count,
                -item.mean_confidence,
                -item.max_confidence,
                _canonical_anchor(item.anchor),
            ),
        )
        dominant[track_id] = (
            _canonical_anchor(winner.anchor),
            winner.support_type.value,
            winner.capacity_class.value,
        )
        if (
            winner.label_vote_count == candidate.semantic_observation_count
            and winner.mean_confidence >= LONG_GAP_SEMANTIC_MIN_CONFIDENCE
        ):
            strong_consensus.add(track_id)

    frame_boxes: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    rows: dict[str, list[SpatialObservation]] = {}
    videos: dict[str, str] = {}
    for track_id in track_ids:
        track_rows = sorted(
            (item for item in grouped[track_id] if item.bbox is not None),
            key=lambda item: (item.timestamp_ms, item.frame_ref),
        )
        rows[track_id] = track_rows
        video_values = {item.video_id for item in track_rows}
        videos[track_id] = next(iter(video_values)) if len(video_values) == 1 else ""
        frame_boxes[track_id] = {
            item.frame_ref: item.bbox for item in track_rows if item.bbox is not None
        }

    cannot_link: set[tuple[str, str]] = set()
    semantic_edges: list[tuple[int, float, str, str]] = []
    for left_index, left in enumerate(track_ids):
        if left not in dominant:
            continue
        for right in track_ids[left_index + 1 :]:
            if (
                right not in dominant
                or dominant[left] != dominant[right]
                or not videos[left]
                or videos[left] != videos[right]
            ):
                continue
            shared = sorted(set(frame_boxes[left]) & set(frame_boxes[right]))
            if shared:
                ioms = [
                    _bbox_iom(frame_boxes[left][frame], frame_boxes[right][frame])
                    for frame in shared
                ]
                distances = [
                    _bbox_center_distance(
                        frame_boxes[left][frame], frame_boxes[right][frame]
                    )
                    for frame in shared
                ]
                if (
                    len(shared) >= 2
                    and sum(value < 0.20 for value in ioms) / len(ioms) >= 0.80
                    and _median(distances) > 0.25
                ):
                    cannot_link.add((left, right))
                    continue
                if (
                    len(shared) >= 2
                    and _median(ioms) >= 0.85
                    and sum(value >= 0.80 for value in ioms) / len(ioms) >= 0.80
                ):
                    semantic_edges.append((0, -_median(ioms), left, right))
                continue
            if not rows[left] or not rows[right]:
                continue
            left_start, left_end = rows[left][0].timestamp_ms, rows[left][-1].timestamp_ms
            right_start, right_end = (
                rows[right][0].timestamp_ms,
                rows[right][-1].timestamp_ms,
            )
            if left_end < right_start:
                gap = right_start - left_end
            elif right_end < left_start:
                gap = left_start - right_end
            else:
                continue
            cosine = _embedding_cosine(embeddings.get(left), embeddings.get(right))
            if cosine is not None and (
                (gap <= max_gap_ms and cosine >= 0.50)
                or (
                    gap <= LONG_GAP_SEMANTIC_MAX_MS
                    and left in strong_consensus
                    and right in strong_consensus
                    and cosine >= LONG_GAP_SEMANTIC_MIN_COSINE
                )
            ):
                semantic_edges.append((1, -cosine, left, right))

    def can_union(left: str, right: str) -> bool:
        left_root, right_root = union.find(left), union.find(right)
        if left_root == right_root:
            return False
        left_members = {item for item in track_ids if union.find(item) == left_root}
        right_members = {item for item in track_ids if union.find(item) == right_root}
        return not any(
            tuple(sorted((a, b))) in cannot_link
            for a in left_members
            for b in right_members
        )

    for _, _, left, right in sorted(semantic_edges):
        if can_union(left, right):
            union.union(left, right)

    members: dict[str, list[str]] = defaultdict(list)
    for track_id in track_ids:
        members[union.find(track_id)].append(track_id)
    result: dict[str, str] = {}
    for items in sorted((sorted(value) for value in members.values()), key=lambda x: x[0]):
        digest = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]
        for track_id in items:
            result[track_id] = f"auto_visual_{digest}"
    return result


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    source = image.convert("RGB").copy()
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (118, 118, 118))
    canvas.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    return canvas


def _expanded_crop(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    padding_ratio: float = 0.12,
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
    bounds = (
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(image.width, math.ceil(x2 + pad_x)),
        min(image.height, math.ceil(y2 + pad_y)),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("bbox does not intersect the evidence frame")
    return image.crop(bounds)


def _context_crop_with_target(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    *,
    context_ratio: float = 0.45,
) -> Image.Image:
    """Crop enough local context for thin surfaces and retain target grounding."""

    x1, y1, x2, y2 = bbox
    span = max(x2 - x1, y2 - y1)
    padding = max(8.0, span * context_ratio)
    left = max(0, math.floor(x1 - padding))
    top = max(0, math.floor(y1 - padding))
    right = min(image.width, math.ceil(x2 + padding))
    bottom = min(image.height, math.ceil(y2 + padding))
    if right <= left or bottom <= top:
        raise ValueError("bbox does not intersect the evidence frame")
    crop = image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        (x1 - left, y1 - top, x2 - left, y2 - top),
        outline=(255, 0, 0),
        width=max(4, crop.width // 120),
    )
    return crop


def _frame_index(value: str) -> int | None:
    match = _FRAME_RE.search(Path(value).name)
    return int(match.group("index")) if match else None


def _representative_observation(
    observations: Sequence[SpatialObservation], hero_ref: str | None
) -> SpatialObservation:
    hero_index = _frame_index(hero_ref or "")
    if hero_index is not None:
        matching = [
            observation
            for observation in observations
            if _frame_index(observation.frame_ref) == hero_index
        ]
        if matching:
            return max(
                matching,
                key=lambda item: (item.model_confidence, item.timestamp_ms, item.frame_ref),
            )
    return max(
        observations,
        key=lambda item: (
            item.model_confidence,
            ((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
            if item.bbox is not None
            else 0.0,
            -item.timestamp_ms,
        ),
    )


def build_contact_sheet(evidence: TrackEvidence) -> tuple[bytes, list[str]]:
    """Render scene context and up to three automatic crop views into one image."""

    representative = _representative_observation(evidence.observations, evidence.hero_ref)
    if representative.bbox is None:
        raise ValueError(f"{evidence.track_id}: representative bbox missing")
    frame_path = _resolve_path(representative.frame_ref)
    if not frame_path.exists():
        raise FileNotFoundError(f"{evidence.track_id}: frame missing: {frame_path}")

    with Image.open(frame_path) as opened:
        frame = opened.convert("RGB")
    scene = frame.copy()
    draw = ImageDraw.Draw(scene)
    draw.rectangle(representative.bbox, outline=(255, 0, 0), width=max(4, scene.width // 240))
    panels: list[Image.Image] = [scene, _expanded_crop(frame, representative.bbox)]
    sources = [str(frame_path)]
    for ref in evidence.prototype_refs:
        path = _resolve_path(ref)
        if not path.exists() or path == frame_path:
            continue
        with Image.open(path) as opened:
            panels.append(opened.convert("RGB"))
        sources.append(str(path))
        if len(panels) == 4:
            break
    while len(panels) < 4:
        panels.append(panels[-1].copy())

    canvas = Image.new("RGB", (1024, 1024), (92, 92, 92))
    labels = ("SCENE - RED TARGET", "TARGET ZOOM", "SECOND VIEW", "THIRD VIEW")
    for index, panel in enumerate(panels[:4]):
        tile = _letterbox(panel, (512, 512))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.rectangle((0, 0, 511, 26), fill=(0, 0, 0))
        tile_draw.text((8, 7), labels[index], fill=(255, 255, 255))
        canvas.paste(tile, ((index % 2) * 512, (index // 2) * 512))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue(), sources


def build_classification_views(
    evidence: TrackEvidence,
) -> list[tuple[bytes, str, int]]:
    """Return up to three independent, target-only automatic crop views.

    Prototype crops are preferred because they contain the tracked proposal
    rather than a full scene with unrelated furniture.  A locally grounded
    representative crop is used only when ingest supplied fewer than three
    prototype frames.
    """

    views: list[tuple[bytes, str, int]] = []
    seen_refs: set[str] = set()
    for ref in evidence.prototype_refs:
        path = _resolve_path(ref)
        normalized = str(path)
        if normalized in seen_refs or not path.exists():
            continue
        with Image.open(path) as opened:
            target_pixels = opened.width * opened.height
            target = _letterbox(opened.convert("RGB"), (768, 768))
        buffer = io.BytesIO()
        target.save(buffer, format="JPEG", quality=92, optimize=True)
        views.append((buffer.getvalue(), normalized, target_pixels))
        seen_refs.add(normalized)
        if len(views) == MAX_CLASSIFICATION_VIEWS:
            return views

    representative = _representative_observation(evidence.observations, evidence.hero_ref)
    if representative.bbox is None:
        return views
    frame_path = _resolve_path(representative.frame_ref)
    normalized = str(frame_path)
    if normalized not in seen_refs and frame_path.exists():
        with Image.open(frame_path) as opened:
            grounded = _context_crop_with_target(
                opened.convert("RGB"), representative.bbox
            )
        x1, y1, x2, y2 = representative.bbox
        target_pixels = max(0, round(x2 - x1)) * max(0, round(y2 - y1))
        grounded = _letterbox(grounded, (768, 768))
        buffer = io.BytesIO()
        grounded.save(buffer, format="JPEG", quality=92, optimize=True)
        views.append((buffer.getvalue(), normalized, target_pixels))
    return views[:MAX_CLASSIFICATION_VIEWS]


def _json_schema(anchors: Sequence[str]) -> dict[str, Any]:
    score_properties = {
        anchor: {"type": "integer", "minimum": 0, "maximum": 100}
        for anchor in anchors
    }
    score_properties["other"] = {"type": "integer", "minimum": 0, "maximum": 100}
    return {
        "type": "object",
        "properties": {
            "anchor_scores": {
                "type": "object",
                "properties": score_properties,
                "required": [*anchors, "other"],
                "additionalProperties": False,
            },
            "best_anchor": {"type": "string", "enum": [*anchors, "other"]},
            "display_name_zh": {"type": "string", "maxLength": 24},
            "support_type": {
                "type": "string",
                "enum": ["surface", "shelf", "floor", "unknown"],
            },
            "support_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "capacity_class": {
                "type": "string",
                "enum": ["small", "medium", "large", "unknown"],
            },
            "capacity_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": [
            "anchor_scores",
            "best_anchor",
            "display_name_zh",
            "support_type",
            "support_confidence",
            "capacity_class",
            "capacity_confidence",
        ],
        "additionalProperties": False,
    }


def _prompt(
    anchors: Sequence[str], anchor_descriptions: Mapping[str, str] | None = None
) -> str:
    descriptions = {
        **DEFAULT_ANCHOR_DESCRIPTIONS,
        **(anchor_descriptions or {}),
    }
    categories = ", ".join(
        f'"{anchor}" ({descriptions.get(_canonical_anchor(anchor), anchor.replace("_", " "))})'
        for anchor in anchors
    )
    return (
        "The image is one automatically cropped target view from a new-home video. "
        "The tracked target occupies the center of the image; small surrounding pixels "
        "are context only. Classify this target alone. Never classify furniture that is "
        "merely visible behind, below, or beside the central crop. Do not infer from crop "
        "framing or assume that the target belongs to a requested class. Treat every "
        "quoted anchor identifier as an opaque stable output token: its parenthesized "
        "description is the only semantic definition. "
        f"Allowed anchor classes are: {categories}; use other when none fits. "
        "Return integer anchor_scores from 0 to 100 as calibrated confidence for every "
        "class (scores need not sum to 100) and best_anchor. Give 90 or above only when "
        "the central crop directly shows the description's defining visual cues; use "
        "0-20 when those cues are absent or only occur outside the target. support_type "
        "means the usable "
        "placement relation: surface for a tabletop/top, shelf for a shelf/compartment, "
        "floor only for a floor zone, unknown if not visible. capacity_class is visual "
        "relative usable capacity, not exact measurement: small = one narrow shelf or one "
        "cabinet compartment; medium = a normal tabletop/dresser top holding several items; "
        "large = a room-scale broad surface or multiple full shelves. Give independent "
        "support/capacity confidence and a short Chinese display name. Output JSON only."
    )


HARD_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "support_type": {
            "type": "string",
            "enum": ["surface", "shelf", "floor", "unknown"],
        },
        "support_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "capacity_class": {
            "type": "string",
            "enum": ["small", "medium", "large", "unknown"],
        },
        "capacity_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": [
        "support_type",
        "support_confidence",
        "capacity_class",
        "capacity_confidence",
    ],
    "additionalProperties": False,
}

HARD_FIELD_PROMPT = (
    "Re-inspect only the central automatically cropped target. Return EXACTLY one flat JSON "
    "object with all four keys: support_type (surface/shelf/floor/unknown), "
    "support_confidence (0-100), capacity_class (small/medium/large/unknown), and "
    "capacity_confidence (0-100). Confidence must be your independent visual "
    "confidence in that field. Do not omit confidence and do not nest values."
)


class Client:
    def __init__(
        self,
        endpoint: str,
        model: str,
        anchors: Sequence[str],
        guided: bool,
        anchor_descriptions: Mapping[str, str] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.anchors = list(anchors)
        self.guided = guided
        self.schema = _json_schema(self.anchors)
        self.anchor_descriptions = dict(anchor_descriptions or {})
        self.prompt = _prompt(self.anchors, self.anchor_descriptions)
        self.usage = {"calls": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.lock = threading.Lock()

    def _chat(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        max_tokens: int,
    ) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                }
            ],
        }
        if self.guided:
            payload["guided_json"] = schema
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        usage = result.get("usage", {})
        with self.lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            self.usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
        return str(result["choices"][0]["message"]["content"])

    def chat(self, image_bytes: bytes) -> str:
        return self._chat(
            image_bytes,
            prompt=self.prompt,
            schema=self.schema,
            # Nemotron sometimes expands the five scores into five nested
            # objects.  The observed shape needs roughly 400 tokens; 220 cut
            # valid JSON before the closing brace and looked like a parser
            # failure.  Short compliant JSON still stops normally.
            max_tokens=MAIN_MAX_TOKENS,
        )

    def chat_hard_fields(self, image_bytes: bytes) -> str:
        return self._chat(
            image_bytes,
            prompt=HARD_FIELD_PROMPT,
            schema=HARD_FIELD_SCHEMA,
            max_tokens=100,
        )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _parse_anchor_fields(
    payload: Mapping[str, Any], anchors: Sequence[str]
) -> dict[str, Any] | None:
    expected_scores = {*anchors, "other"}
    scores = payload.get("anchor_scores")
    per_anchor_nodes: dict[str, Mapping[str, Any]] = {}
    if not isinstance(scores, dict):
        # Nemotron occasionally flattens the score object despite guided JSON.
        flat_scores = {
            anchor: payload[anchor]
            for anchor in [*anchors, "other"]
            if anchor in payload and not isinstance(payload[anchor], dict)
        }
        if flat_scores:
            scores = flat_scores
        else:
            # Another observed shape makes each anchor a small object.  Only
            # the model-provided score is lifted; support/capacity still need
            # their own confidence gate below or an independent repair call.
            per_anchor_nodes = {
                anchor: payload[anchor]
                for anchor in anchors
                if isinstance(payload.get(anchor), dict)
            }
            if set(per_anchor_nodes) == set(anchors) and all(
                "anchor_score" in node for node in per_anchor_nodes.values()
            ):
                scores = {
                    anchor: node["anchor_score"]
                    for anchor, node in per_anchor_nodes.items()
                }
                other_node = payload.get("other")
                if isinstance(other_node, dict) and "anchor_score" in other_node:
                    scores["other"] = other_node["anchor_score"]
            else:
                scores = None
    if not isinstance(scores, dict):
        return None
    # The model also consistently omits only the catch-all ``other`` score in
    # one response shape.  All requested semantic scores remain mandatory;
    # treating the omitted catch-all as zero does not invent a target label.
    if set(scores) == set(anchors):
        scores = {**scores, "other": 0}
    if set(scores) != expected_scores:
        return None
    try:
        normalized_scores = {
            anchor: max(0, min(100, int(scores[anchor]))) for anchor in sorted(scores)
        }
        best_anchor_raw = payload.get("best_anchor") or payload.get("target_object")
        if best_anchor_raw is None:
            best_score = max(normalized_scores.values())
            winners = [
                anchor
                for anchor, score in normalized_scores.items()
                if score == best_score
            ]
            if len(winners) != 1:
                return None
            best_anchor_raw = winners[0]
        best_anchor = str(best_anchor_raw)
    except (KeyError, TypeError, ValueError):
        return None
    if best_anchor not in expected_scores:
        return None
    best_node = per_anchor_nodes.get(best_anchor, {})
    return {
        "anchor_scores": normalized_scores,
        "best_anchor": best_anchor,
        "display_name_zh": str(
            payload.get("display_name_zh")
            or payload.get("display_name")
            or payload.get("chinese_display_name")
            or best_node.get("display_name_zh")
            or best_node.get("display_name")
            or best_node.get("chinese_display_name")
            or ANCHOR_DISPLAY_ZH.get(best_anchor)
            or best_anchor.replace("_", " ")
            or "自动识别区域"
        ).strip()[:24]
        or "自动识别区域",
    }


def _parse_hard_fields(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        support_raw = payload["support_type"]
        capacity_raw = payload["capacity_class"]

        def nested_name(value: Any) -> Any:
            if not isinstance(value, dict):
                return value
            return (
                value.get("name")
                or value.get("description")
                or value.get("value")
                or value.get("type")
            )

        support_type = str(
            nested_name(support_raw)
        )
        capacity_class = str(
            nested_name(capacity_raw)
        )
        support_confidence = max(
            0,
            min(
                100,
                int(
                    support_raw.get("confidence")
                    if isinstance(support_raw, dict)
                    else payload["support_confidence"]
                ),
            ),
        )
        capacity_confidence = max(
            0,
            min(
                100,
                int(
                    capacity_raw.get("confidence")
                    if isinstance(capacity_raw, dict)
                    else payload["capacity_confidence"]
                ),
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if support_type not in {"surface", "shelf", "floor", "unknown"}:
        return None
    if capacity_class not in {"small", "medium", "large", "unknown"}:
        return None
    return {
        "support_type": support_type,
        "support_confidence": support_confidence,
        "capacity_class": capacity_class,
        "capacity_confidence": capacity_confidence,
    }


def parse_anchor_prediction(text: str, anchors: Sequence[str]) -> dict[str, Any] | None:
    payload = _parse_json_object(text)
    return _parse_anchor_fields(payload, anchors) if payload is not None else None


def parse_hard_field_prediction(text: str) -> dict[str, Any] | None:
    payload = _parse_json_object(text)
    return _parse_hard_fields(payload) if payload is not None else None


def parse_prediction(text: str, anchors: Sequence[str]) -> dict[str, Any] | None:
    payload = _parse_json_object(text)
    if payload is None:
        return None
    anchor_fields = _parse_anchor_fields(payload, anchors)
    hard_fields = _parse_hard_fields(payload)
    if hard_fields is None and anchor_fields is not None:
        best_node = payload.get(anchor_fields["best_anchor"])
        if isinstance(best_node, dict):
            hard_fields = _parse_hard_fields(best_node)
    if anchor_fields is None or hard_fields is None:
        return None
    return {**anchor_fields, **hard_fields}


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[str(record["key"])] = dict(record["prediction"])
    return cache


def _aggregate_power(
    observations: Sequence[SpatialObservation], min_confidence: float = 0.70
) -> tuple[PowerState, float | None, list[str]]:
    valid: list[tuple[float, list[str]]] = []
    for observation in observations:
        confidence = observation.power_confidence
        refs = sorted({ref.strip() for ref in observation.power_evidence_refs if ref.strip()})
        if (
            observation.power_state is PowerState.NEAR
            and confidence is not None
            and confidence >= min_confidence
            and refs
        ):
            valid.append((confidence, refs))
    if not valid:
        return PowerState.UNKNOWN, None, []
    return (
        PowerState.NEAR,
        round(max(item[0] for item in valid), 6),
        sorted({ref for _, refs in valid for ref in refs}),
    )


def aggregate_view_predictions(
    predictions: Sequence[Mapping[str, Any]], anchors: Sequence[str]
) -> dict[str, Any]:
    """Aggregate independent target views into real votes and calibrated scores."""

    if len(predictions) < MIN_VALID_CLASSIFICATION_VIEWS:
        raise ValueError(
            f"at least {MIN_VALID_CLASSIFICATION_VIEWS} valid views are required"
        )
    expected_scores = {*anchors, "other"}
    if any(set(item["anchor_scores"]) != expected_scores for item in predictions):
        raise ValueError("view predictions do not share the expected anchor schema")

    view_count = len(predictions)
    vote_counts = Counter(str(item["best_anchor"]) for item in predictions)
    mean_scores = {
        anchor: int(
            round(
                math.fsum(float(item["anchor_scores"][anchor]) for item in predictions)
                / view_count
            )
        )
        for anchor in sorted(expected_scores)
    }
    max_scores = {
        anchor: max(int(item["anchor_scores"][anchor]) for item in predictions)
        for anchor in sorted(expected_scores)
    }
    best_anchor = min(
        expected_scores,
        key=lambda anchor: (
            -vote_counts[anchor],
            -mean_scores[anchor],
            -max_scores[anchor],
            anchor,
        ),
    )

    def consensus(value_key: str, confidence_key: str) -> tuple[str, int]:
        values = [str(item[value_key]) for item in predictions]
        counts = Counter(value for value in values if value != "unknown")
        if not counts:
            return "unknown", 0
        winner = min(
            counts,
            key=lambda value: (
                -counts[value],
                -math.fsum(
                    float(item[confidence_key])
                    for item in predictions
                    if str(item[value_key]) == value
                ),
                value,
            ),
        )
        if counts[winner] <= view_count / 2:
            return "unknown", 0
        confidences = [
            int(item[confidence_key])
            for item in predictions
            if str(item[value_key]) == winner
        ]
        return winner, int(round(math.fsum(confidences) / len(confidences)))

    support_type, support_confidence = consensus(
        "support_type", "support_confidence"
    )
    capacity_class, capacity_confidence = consensus(
        "capacity_class", "capacity_confidence"
    )
    matching_names = [
        (
            int(item["anchor_scores"][best_anchor]),
            str(item["display_name_zh"]),
        )
        for item in predictions
        if str(item["best_anchor"]) == best_anchor
    ]
    display_name = (
        min(matching_names, key=lambda item: (-item[0], item[1]))[1]
        if matching_names
        else ANCHOR_DISPLAY_ZH.get(best_anchor, "自动识别区域")
    )
    return {
        "anchor_scores": mean_scores,
        "anchor_max_scores": max_scores,
        "anchor_vote_counts": {
            anchor: vote_counts[anchor] for anchor in sorted(expected_scores)
        },
        "best_anchor": best_anchor,
        "display_name_zh": display_name,
        "support_type": support_type,
        "support_confidence": support_confidence,
        "capacity_class": capacity_class,
        "capacity_confidence": capacity_confidence,
        "view_count": view_count,
    }


def calibrate_prediction_geometry(
    evidence: TrackEvidence, prediction: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply narrow, deterministic geometry calibration to visual hard fields."""

    projected = dict(prediction)
    calibrations: list[dict[str, Any]] = []
    if prediction["support_type"] != "shelf" or prediction["capacity_class"] != "medium":
        return projected, calibrations
    aspect_ratios = sorted(
        (item.bbox[2] - item.bbox[0]) / (item.bbox[3] - item.bbox[1])
        for item in evidence.observations
        if item.bbox is not None and item.bbox[3] > item.bbox[1]
    )
    if not aspect_ratios:
        return projected, calibrations
    median_aspect = _median(aspect_ratios)
    # A single, extremely thin horizontal support is the classifier's own
    # documented ``small = one narrow shelf`` case.  This rule is independent
    # of anchor identity and never upgrades unknown/low-confidence evidence.
    if median_aspect >= 4.0 and int(prediction["capacity_confidence"]) >= 70:
        projected["capacity_class"] = "small"
        projected["capacity_confidence"] = min(
            90, int(prediction["capacity_confidence"])
        )
        calibrations.append(
            {
                "version": THIN_SHELF_CALIBRATION_VERSION,
                "field": "capacity_class",
                "from": "medium",
                "to": "small",
                "median_bbox_aspect_ratio": round(median_aspect, 8),
            }
        )
    return projected, calibrations


def quarantine_low_information_prediction(
    prediction: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep a diagnostic candidate while making every semantic edge ineligible."""

    quarantined = dict(prediction)
    score_keys = sorted(prediction["anchor_scores"])
    quarantined["anchor_scores"] = {
        key: 100 if key == "other" else 0 for key in score_keys
    }
    quarantined["anchor_max_scores"] = {
        key: 100 if key == "other" else 0 for key in score_keys
    }
    quarantined["anchor_vote_counts"] = {
        key: int(prediction["view_count"]) if key == "other" else 0
        for key in score_keys
    }
    quarantined.update(
        {
            "best_anchor": "other",
            "display_name_zh": "低信息自动候选",
            "support_type": "unknown",
            "support_confidence": 0,
            "capacity_class": "unknown",
            "capacity_confidence": 0,
        }
    )
    return quarantined


def _candidate_from_prediction(
    evidence: TrackEvidence,
    prediction: Mapping[str, Any],
    *,
    contact_ref: str,
    contact_sha256: str,
    model: str,
    view_evidence: Sequence[tuple[str, str]] = (),
    extra_model_versions: Sequence[str] = (),
) -> AutomaticAnchorCandidate:
    observation_count = len(evidence.observations)
    semantic_observation_count = int(
        prediction.get("view_count") or observation_count
    )
    support = None if prediction["support_type"] == "unknown" else prediction["support_type"]
    capacity = (
        None if prediction["capacity_class"] == "unknown" else prediction["capacity_class"]
    )
    power_state, power_confidence, power_refs = _aggregate_power(evidence.observations)
    hypotheses = []
    for anchor in sorted(prediction["anchor_scores"]):
        if anchor == "other":
            continue
        score = float(prediction["anchor_scores"][anchor]) / 100.0
        vote_counts = prediction.get("anchor_vote_counts")
        max_scores = prediction.get("anchor_max_scores")
        vote_count = (
            int(vote_counts.get(anchor, 0))
            if isinstance(vote_counts, Mapping)
            else max(
                1,
                min(
                    semantic_observation_count,
                    int(round(score * semantic_observation_count)),
                ),
            )
        )
        max_score = (
            float(max_scores.get(anchor, 0)) / 100.0
            if isinstance(max_scores, Mapping)
            else score
        )
        hypotheses.append(
            {
                "anchor": anchor,
                "label_vote_count": vote_count,
                "mean_confidence": score,
                "max_confidence": max_score,
                "proposal_display_name_zh": prediction["display_name_zh"],
                "support_type": support,
                "support_confidence": float(prediction["support_confidence"]) / 100.0,
                "capacity_class": capacity,
                "capacity_confidence": float(prediction["capacity_confidence"]) / 100.0,
            }
        )
    detector_versions = {
        item.model_version.strip() for item in evidence.observations if item.model_version.strip()
    }
    original_refs = {
        ref.strip()
        for item in evidence.observations
        for ref in item.evidence_refs
        if ref.strip()
    }
    return AutomaticAnchorCandidate.model_validate(
        {
            "candidate_id": f"auto_anchor_{evidence.track_id}",
            "visual_instance_id": evidence.visual_instance_id,
            "observation_count": observation_count,
            "semantic_observation_count": semantic_observation_count,
            "display_name_zh": prediction["display_name_zh"],
            "power_state": power_state.value,
            "power_confidence": power_confidence,
            "power_evidence_refs": power_refs,
            "evidence_refs": sorted(
                {
                    *original_refs,
                    *(
                        {
                            f"auto_vlm_view:{ref}@sha256:{digest}"
                            for ref, digest in view_evidence
                        }
                        if view_evidence
                        else {
                            f"auto_vlm_contact:{contact_ref}@sha256:{contact_sha256}"
                        }
                    ),
                }
            ),
            "source_track_ids": [evidence.track_id],
            "model_versions": sorted(
                {
                    *detector_versions,
                    *extra_model_versions,
                    f"{CLASSIFIER_VERSION}@{model}",
                }
            ),
            "anchor_hypotheses": hypotheses,
        }
    )


def _classify_image(
    client: Client,
    image_bytes: bytes,
    *,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock,
    track_id: str,
    view_index: int,
) -> tuple[dict[str, Any] | None, bool, bool, list[str], str]:
    image_sha = _sha256_bytes(image_bytes)
    inference_contract_sha = _sha256_bytes(
        _json_bytes(
            {
                "prompt": client.prompt,
                "schema": client.schema,
                "hard_field_prompt": HARD_FIELD_PROMPT,
                "hard_field_schema": HARD_FIELD_SCHEMA,
                "guided": client.guided,
            }
        ).rstrip(b"\n")
    )
    cache_key = (
        f"{image_sha}:{inference_contract_sha}:{CLASSIFIER_VERSION}:{client.model}"
    )
    with cache_lock:
        prediction = cache.get(cache_key)
    cache_hit = prediction is not None
    raw_responses: list[str] = []
    hard_fields_unresolved = bool(
        prediction is not None
        and prediction.get("support_type") == "unknown"
        and prediction.get("support_confidence") == 0
        and prediction.get("capacity_class") == "unknown"
        and prediction.get("capacity_confidence") == 0
    )
    if prediction is None:
        raw_text: str | None = None
        for attempt in range(2):
            try:
                raw_text = client.chat(image_bytes)
                raw_responses.append(raw_text)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                with client.lock:
                    client.usage["errors"] += 1
                with _print_lock:
                    print(
                        f"[warn] {track_id} view {view_index} attempt "
                        f"{attempt + 1}: {exc}",
                        file=sys.stderr,
                    )
                raw_text = None
            if raw_text is not None:
                break
        prediction = (
            parse_prediction(raw_text, client.anchors) if raw_text is not None else None
        )
        if prediction is None and raw_text is not None:
            anchor_fields = parse_anchor_prediction(raw_text, client.anchors)
            if anchor_fields is not None:
                for attempt in range(2):
                    try:
                        hard_text = client.chat_hard_fields(image_bytes)
                        raw_responses.append(hard_text)
                        hard_fields = parse_hard_field_prediction(hard_text)
                    except (
                        urllib.error.URLError,
                        urllib.error.HTTPError,
                        TimeoutError,
                    ) as exc:
                        with client.lock:
                            client.usage["errors"] += 1
                        with _print_lock:
                            print(
                                f"[warn] {track_id} view {view_index} hard-fields "
                                f"attempt {attempt + 1}: {exc}",
                                file=sys.stderr,
                            )
                        hard_fields = None
                    if hard_fields is not None:
                        prediction = {**anchor_fields, **hard_fields}
                        break
                if prediction is None:
                    prediction = {
                        **anchor_fields,
                        "support_type": "unknown",
                        "support_confidence": 0,
                        "capacity_class": "unknown",
                        "capacity_confidence": 0,
                    }
                    hard_fields_unresolved = True
        if prediction is not None:
            with cache_lock:
                cache[cache_key] = prediction
                with cache_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {"key": cache_key, "prediction": prediction},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
    return prediction, cache_hit, hard_fields_unresolved, raw_responses, image_sha


def classify_one(
    client: Client,
    evidence: TrackEvidence,
    *,
    out_dir: Path,
    evidence_dir: Path,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_lock: threading.Lock,
) -> tuple[AutomaticAnchorCandidate | None, dict[str, Any]]:
    contact_bytes, contact_sources = build_contact_sheet(evidence)
    contact_sha = _sha256_bytes(contact_bytes)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    contact_path = evidence_dir / f"{evidence.track_id}.jpg"
    contact_path.write_bytes(contact_bytes)
    view_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    informative_predictions: list[dict[str, Any]] = []
    view_evidence: list[tuple[str, str]] = []
    all_raw_responses: list[str] = []
    for view_index, (image_bytes, source_ref, target_pixel_area) in enumerate(
        build_classification_views(evidence), 1
    ):
        view_path = evidence_dir / f"{evidence.track_id}.view-{view_index}.jpg"
        view_path.write_bytes(image_bytes)
        prediction, cache_hit, unresolved, raw_responses, image_sha = _classify_image(
            client,
            image_bytes,
            cache=cache,
            cache_path=cache_path,
            cache_lock=cache_lock,
            track_id=evidence.track_id,
            view_index=view_index,
        )
        relative_view_ref = (
            view_path.relative_to(PROJ).as_posix()
            if view_path.is_relative_to(PROJ)
            else view_path.as_posix()
        )
        view_rows.append(
            {
                "view_index": view_index,
                "source_ref": source_ref,
                "target_pixel_area": target_pixel_area,
                "information_eligible": target_pixel_area >= MIN_TARGET_VIEW_PIXELS,
                "evidence_ref": relative_view_ref,
                "sha256": image_sha,
                "cache_hit": cache_hit,
                "status": (
                    "HARD_FIELDS_UNRESOLVED"
                    if unresolved
                    else "OK" if prediction is not None else "CLASSIFICATION_FAILED"
                ),
                "prediction": prediction,
            }
        )
        if raw_responses:
            all_raw_responses.extend(raw_responses)
        if prediction is not None:
            predictions.append(prediction)
            if target_pixel_area >= MIN_TARGET_VIEW_PIXELS:
                informative_predictions.append(prediction)
            view_evidence.append((relative_view_ref, image_sha))

    raw_aggregate_prediction = (
        aggregate_view_predictions(predictions, client.anchors)
        if len(predictions) >= MIN_VALID_CLASSIFICATION_VIEWS
        else None
    )
    informative_view_count = len(informative_predictions)
    prediction = (
        aggregate_view_predictions(informative_predictions, client.anchors)
        if informative_view_count >= MIN_VALID_CLASSIFICATION_VIEWS
        else raw_aggregate_prediction
    )
    low_information = bool(
        raw_aggregate_prediction is not None
        and informative_view_count < MIN_VALID_CLASSIFICATION_VIEWS
    )
    if low_information and prediction is not None:
        prediction = quarantine_low_information_prediction(prediction)
    hard_fields_unresolved = bool(
        prediction is not None
        and (
            prediction["support_type"] == "unknown"
            or prediction["capacity_class"] == "unknown"
        )
    )
    diagnostic = {
        "track_id": evidence.track_id,
        "visual_instance_id": evidence.visual_instance_id,
        "contact_sha256": contact_sha,
        "contact_ref": (
            contact_path.relative_to(PROJ).as_posix()
            if contact_path.is_relative_to(PROJ)
            else contact_path.as_posix()
        ),
        "source_refs": contact_sources,
        "requested_view_count": len(view_rows),
        "valid_view_count": len(predictions),
        "informative_view_count": informative_view_count,
        "minimum_target_view_pixels": MIN_TARGET_VIEW_PIXELS,
        "cache_hit": bool(view_rows) and all(item["cache_hit"] for item in view_rows),
        "views": view_rows,
        "status": (
            "LOW_INFORMATION_VIEWS"
            if low_information
            else "HARD_FIELDS_UNRESOLVED"
            if hard_fields_unresolved
            else "OK" if prediction is not None else "CLASSIFICATION_FAILED"
        ),
        "prediction": prediction,
    }
    if low_information:
        diagnostic["raw_aggregate_prediction"] = raw_aggregate_prediction
    if (prediction is None or hard_fields_unresolved) and all_raw_responses:
        diagnostic["response_excerpts"] = [
            item[-2000:] for item in all_raw_responses
        ]
    if prediction is None:
        return None, diagnostic
    projected_prediction, calibrations = calibrate_prediction_geometry(
        evidence, prediction
    )
    if calibrations:
        diagnostic["projection_calibrations"] = calibrations
    candidate = _candidate_from_prediction(
        evidence,
        projected_prediction,
        contact_ref=diagnostic["contact_ref"],
        contact_sha256=contact_sha,
        model=client.model,
        view_evidence=view_evidence,
        extra_model_versions=(
            [THIN_SHELF_CALIBRATION_VERSION] if calibrations else []
        ),
    )
    return candidate, diagnostic


def run_classifier(
    *,
    observations_path: Path,
    ingest_dir: Path,
    out_dir: Path,
    evidence_dir: Path | None = None,
    expected_anchors: Sequence[str],
    anchor_descriptions: Mapping[str, str] | None = None,
    endpoint: str,
    model: str,
    concurrency: int = 8,
    guided: bool = True,
    max_failed_rate: float = 0.02,
    limit: int = 0,
) -> tuple[list[AutomaticAnchorCandidate], dict[str, Any], dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    observations = load_observations_jsonl(observations_path)
    grouped: dict[str, list[SpatialObservation]] = defaultdict(list)
    for observation in observations:
        if not observation.region_track_id:
            raise ValueError("all classifier observations must have automatic region_track_id")
        grouped[observation.region_track_id].append(observation)
    tracklets = _load_tracklets(ingest_dir / "tracklets.jsonl")
    missing = sorted(set(grouped) - set(tracklets))
    if missing:
        raise ValueError(f"automatic tracks missing from ingest: {missing[:5]}")
    track_embeddings = _load_track_embeddings(tracklets)
    instance_ids = automatic_visual_instance_ids(
        grouped, embeddings=track_embeddings
    )
    evidence_rows: list[TrackEvidence] = []
    for track_id in sorted(grouped):
        tracklet = tracklets[track_id]
        evidence_rows.append(
            TrackEvidence(
                track_id=track_id,
                observations=tuple(
                    sorted(grouped[track_id], key=lambda item: (item.timestamp_ms, item.frame_ref))
                ),
                prototype_refs=tuple(tracklet.prototype_refs),
                hero_ref=str(tracklet.attributes.get("hero_ref") or "").strip() or None,
                visual_instance_id=instance_ids[track_id],
            )
        )
    if limit:
        evidence_rows = evidence_rows[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = evidence_dir or out_dir / "evidence"
    cache_path = out_dir / "anchor_predictions.cache.jsonl"
    cache = _load_cache(cache_path)
    cache_lock = threading.Lock()
    client = Client(
        endpoint,
        model,
        expected_anchors,
        guided,
        anchor_descriptions=anchor_descriptions,
    )
    candidates: list[AutomaticAnchorCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                classify_one,
                client,
                evidence,
                out_dir=out_dir,
                evidence_dir=evidence_dir,
                cache=cache,
                cache_path=cache_path,
                cache_lock=cache_lock,
            ): evidence.track_id
            for evidence in evidence_rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            track_id = futures[future]
            try:
                candidate, diagnostic = future.result()
            except Exception as exc:  # keep full automatic-batch evidence, fail at the gate below
                candidate = None
                diagnostic = {
                    "track_id": track_id,
                    "status": "CLASSIFICATION_EXCEPTION",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            diagnostics.append(diagnostic)
            if candidate is not None:
                candidates.append(candidate)
            if completed % 10 == 0 or completed == len(futures):
                with _print_lock:
                    print(f"[space-anchor] {completed}/{len(futures)}", flush=True)

    pre_semantic_instance_count = len(set(instance_ids.values()))
    instance_ids = semantic_visual_instance_ids(
        grouped,
        candidates,
        instance_ids,
        embeddings=track_embeddings,
    )
    candidates = [
        candidate.model_copy(
            update={
                "visual_instance_id": instance_ids[candidate.source_track_ids[0]],
                "model_versions": sorted(
                    {*candidate.model_versions, VISUAL_INSTANCE_VERSION}
                ),
            }
        )
        for candidate in candidates
    ]
    for diagnostic in diagnostics:
        track_id = str(diagnostic.get("track_id") or "")
        if track_id in instance_ids:
            diagnostic["visual_instance_id"] = instance_ids[track_id]
    candidates.sort(key=lambda item: item.candidate_id)
    diagnostics.sort(key=lambda item: item["track_id"])
    failed = len(evidence_rows) - len(candidates)
    failed_rate = failed / len(evidence_rows) if evidence_rows else 1.0
    hard_field_unresolved = sum(
        item["status"] == "HARD_FIELDS_UNRESOLVED" for item in diagnostics
    )
    low_information_tracks = sum(
        item["status"] == "LOW_INFORMATION_VIEWS" for item in diagnostics
    )
    requested_view_count = sum(
        int(item.get("requested_view_count", 0)) for item in diagnostics
    )
    valid_view_count = sum(int(item.get("valid_view_count", 0)) for item in diagnostics)
    visual_instance_members: dict[str, list[str]] = defaultdict(list)
    for track_id, instance_id in sorted(instance_ids.items()):
        visual_instance_members[instance_id].append(track_id)
    if failed_rate > max_failed_rate:
        failure_path = out_dir / "failure_diagnostics.json"
        failure_path.write_bytes(
            _json_bytes(
                {
                    "schema_version": CLASSIFIER_SCHEMA_VERSION,
                    "classifier_version": CLASSIFIER_VERSION,
                    "input_track_count": len(evidence_rows),
                    "classified_track_count": len(candidates),
                    "failed_track_count": failed,
                    "failed_rate": round(failed_rate, 8),
                    "max_failed_rate": max_failed_rate,
                    "usage": dict(client.usage),
                    "diagnostics": diagnostics,
                },
                indent=2,
            )
        )
        raise RuntimeError(
            f"automatic anchor classification failure rate {failed_rate:.4f} "
            f"exceeds {max_failed_rate:.4f}; diagnostics: {failure_path}"
        )

    candidate_payload = [item.model_dump(mode="json") for item in candidates]
    candidates_bytes = _json_bytes(candidate_payload, indent=2)
    metrics: dict[str, Any] = {
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "expected_anchor_labels": list(expected_anchors),
        "anchor_descriptions": dict(sorted((anchor_descriptions or {}).items())),
        "input_observation_count": len(observations),
        "input_track_count": len(grouped),
        "classified_track_count": len(candidates),
        "failed_track_count": failed,
        "failed_rate": round(failed_rate, 8),
        "hard_field_unresolved_track_count": hard_field_unresolved,
        "hard_field_unresolved_rate": round(
            hard_field_unresolved / len(evidence_rows) if evidence_rows else 1.0,
            8,
        ),
        "low_information_track_count": low_information_tracks,
        "minimum_target_view_pixels": MIN_TARGET_VIEW_PIXELS,
        "classification_vote_source": "INDEPENDENT_TARGET_CROPS",
        "minimum_valid_views_per_track": MIN_VALID_CLASSIFICATION_VIEWS,
        "maximum_views_per_track": MAX_CLASSIFICATION_VIEWS,
        "requested_classification_view_count": requested_view_count,
        "valid_classification_view_count": valid_view_count,
        "visual_instance_algorithm": VISUAL_INSTANCE_VERSION,
        "pre_semantic_visual_instance_count": pre_semantic_instance_count,
        "semantic_instance_merge_count": max(
            0, pre_semantic_instance_count - len(set(instance_ids.values()))
        ),
        "automatic_visual_instance_count": len(set(instance_ids.values())),
        "automatic_visual_instances": [
            {
                "visual_instance_id": instance_id,
                "source_track_ids": sorted(members),
            }
            for instance_id, members in sorted(visual_instance_members.items())
        ],
        "valid_track_embedding_count": len(track_embeddings),
        "model": model,
        "endpoint": endpoint,
        "guided_json": guided,
        "concurrency": concurrency,
        "usage": dict(client.usage),
        "diagnostics": diagnostics,
    }
    metrics_bytes = _json_bytes(metrics, indent=2)
    hashes: dict[str, Any] = {
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "algorithm": "sha256",
        "inputs": {
            "auto_observations.jsonl": _sha256_file(observations_path),
            "tracklets.jsonl": _sha256_file(ingest_dir / "tracklets.jsonl"),
            "expected_anchors": _sha256_bytes(_json_bytes(list(expected_anchors)).rstrip(b"\n")),
            "anchor_descriptions": _sha256_bytes(
                _json_bytes(dict(sorted((anchor_descriptions or {}).items()))).rstrip(b"\n")
            ),
            "classification_views": _sha256_bytes(
                _json_bytes(
                    [
                        {
                            "track_id": item["track_id"],
                            "views": [
                                {
                                    "source_ref": view["source_ref"],
                                    "sha256": view["sha256"],
                                }
                                for view in item.get("views", [])
                            ],
                        }
                        for item in diagnostics
                    ]
                ).rstrip(b"\n")
            ),
        },
        "outputs": {
            ANCHOR_CANDIDATES_FILENAME: _sha256_bytes(candidates_bytes),
            METRICS_FILENAME: _sha256_bytes(metrics_bytes),
        },
    }
    hashes["normalized_hash"] = _sha256_bytes(_json_bytes(hashes).rstrip(b"\n"))
    (out_dir / ANCHOR_CANDIDATES_FILENAME).write_bytes(candidates_bytes)
    (out_dir / METRICS_FILENAME).write_bytes(metrics_bytes)
    (out_dir / HASHES_FILENAME).write_bytes(_json_bytes(hashes, indent=2))
    return candidates, metrics, hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--ingest-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="contact sheets; set to local-data so pull_results only transfers JSON",
    )
    parser.add_argument("--expected-anchor", action="append", default=[])
    parser.add_argument(
        "--anchor-description",
        action="append",
        default=[],
        metavar="ANCHOR=TEXT",
        help="production target-vocabulary definition; never a region/track ID",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-failed-rate", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-guided", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    anchors = _expected_anchors(args.expected_anchor)
    anchor_keys = {_canonical_anchor(item) for item in anchors}
    anchor_descriptions: dict[str, str] = {}
    for raw in args.anchor_description:
        key, separator, description = raw.partition("=")
        canonical = _canonical_anchor(key)
        description = description.strip()
        if not separator or canonical not in anchor_keys or not description:
            raise ValueError(
                "--anchor-description must be ANCHOR=TEXT for an expected anchor"
            )
        if canonical in anchor_descriptions:
            raise ValueError(f"duplicate --anchor-description: {canonical}")
        anchor_descriptions[canonical] = description
    candidates, metrics, hashes = run_classifier(
        observations_path=args.observations,
        ingest_dir=args.ingest_dir,
        out_dir=args.out_dir,
        evidence_dir=args.evidence_dir,
        expected_anchors=anchors,
        anchor_descriptions=anchor_descriptions,
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        guided=not args.no_guided,
        max_failed_rate=args.max_failed_rate,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "classified_tracks": len(candidates),
                "visual_instances": metrics["automatic_visual_instance_count"],
                "failed_tracks": metrics["failed_track_count"],
                "normalized_hash": hashes["normalized_hash"],
                "output": str(args.out_dir / ANCHOR_CANDIDATES_FILENAME),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

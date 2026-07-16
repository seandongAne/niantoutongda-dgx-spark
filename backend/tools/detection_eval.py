"""S2.5 困难验证集的确定性检测/轨迹评测。

输入是两个 JSON object（也可先由 :func:`load_json_document` 从文件读取）：

Ground truth::

    {
      "dataset_id": "dev_a_hardval_v1",
      "frames": [{
        "sequence_id": "video_1",          # 可选；用于跨视频隔离 instance/track
        "frame_id": "000123",
        "instances": [{
          "instance_id": "anchor_06",      # 真值实例 ID；跨帧保持不变
          "canonical_id": "night_light",   # 检测概念，不得塞实例答案
          "bbox": [x1, y1, x2, y2],
          "visible": true                   # 省略时默认为 true
        }]
      }]
    }

Predictions::

    {
      "dataset_id": "dev_a_hardval_v1",
      "frames": [{
        "sequence_id": "video_1",
        "frame_id": "000123",
        "predictions": [{
          "track_id": "track_7",
          "canonical_id": "night_light",
          "bbox": [x1, y1, x2, y2]
        }]
      }]
    }

预测文件可以省略零检出帧，但不能含 GT 之外的帧。三个指标口径固定：

* 可见 GT 框按 canonical_id 相同且 IoU >= 0.5 做逐帧一对一匹配后的召回；
* 至少命中过一次的 GT instance 中，被多个不同 track_id 命中的比例；
* 一对一匹配后未配预测框数 / GT 帧数。

不读取图片、不生成标注，也不依赖模型；同一语义输入的结果与数组原始顺序无关。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IOU_THRESHOLD = 0.5
SCHEMA_VERSION = "1.0"
_FORBIDDEN_DATASET_MARKERS = ("task_b", "dev_b")


class EvaluationInputError(ValueError):
    """输入不完整、矛盾或触碰任务 B 隔离红线。"""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True, slots=True)
class GroundTruthInstance:
    sequence_id: str
    frame_id: str
    instance_id: str
    canonical_id: str
    bbox: BoundingBox
    visible: bool

    @property
    def identity(self) -> tuple[str, str]:
        return self.sequence_id, self.instance_id

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self.instance_id, self.canonical_id, *self.bbox.as_list())


@dataclass(frozen=True, slots=True)
class Prediction:
    sequence_id: str
    frame_id: str
    track_id: str
    canonical_id: str
    bbox: BoundingBox

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self.track_id, self.canonical_id, *self.bbox.as_list())


@dataclass(frozen=True, slots=True)
class Match:
    sequence_id: str
    frame_id: str
    instance_id: str
    track_id: str
    canonical_id: str
    iou: float

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "frame_id": self.frame_id,
            "instance_id": self.instance_id,
            "track_id": self.track_id,
            "canonical_id": self.canonical_id,
            "iou": self.iou,
        }
        if self.sequence_id:
            result["sequence_id"] = self.sequence_id
        return result


@dataclass(frozen=True, slots=True)
class FrameEvaluation:
    sequence_id: str
    frame_id: str
    visible_gt_boxes: int
    prediction_boxes: int
    matches: tuple[Match, ...]
    unmatched_prediction_track_ids: tuple[str, ...]

    @property
    def unmatched_predictions(self) -> int:
        return len(self.unmatched_prediction_track_ids)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "frame_id": self.frame_id,
            "visible_gt_boxes": self.visible_gt_boxes,
            "prediction_boxes": self.prediction_boxes,
            "matched_boxes": len(self.matches),
            "unmatched_predictions": self.unmatched_predictions,
            "unmatched_prediction_track_ids": list(self.unmatched_prediction_track_ids),
            "matches": [match.to_dict() for match in self.matches],
        }
        if self.sequence_id:
            result["sequence_id"] = self.sequence_id
        return result


@dataclass(frozen=True, slots=True)
class DetectionEvaluation:
    dataset_id: str
    frame_count: int
    visible_gt_boxes: int
    matched_visible_gt_boxes: int
    matched_gt_instances: int
    fragmented_gt_instances: int
    prediction_boxes: int
    unmatched_predictions: int
    frames: tuple[FrameEvaluation, ...]

    @property
    def visible_instance_recall(self) -> float:
        return self.matched_visible_gt_boxes / self.visible_gt_boxes

    @property
    def fragmentation_rate(self) -> float:
        if self.matched_gt_instances == 0:
            return 0.0
        return self.fragmented_gt_instances / self.matched_gt_instances

    @property
    def false_positives_per_frame(self) -> float:
        return self.unmatched_predictions / self.frame_count

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "visible_instance_recall": {
                "value": self.visible_instance_recall,
                "matched_visible_gt_boxes": self.matched_visible_gt_boxes,
                "visible_gt_boxes": self.visible_gt_boxes,
            },
            "fragmentation_rate": {
                "value": self.fragmentation_rate,
                "fragmented_gt_instances": self.fragmented_gt_instances,
                "matched_gt_instances": self.matched_gt_instances,
            },
            "false_positives_per_frame": {
                "value": self.false_positives_per_frame,
                "unmatched_predictions": self.unmatched_predictions,
                "frames": self.frame_count,
            },
        }

    def to_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "iou_threshold": IOU_THRESHOLD,
            "metrics": self.metrics_dict(),
            "counts": {
                "frames": self.frame_count,
                "visible_gt_boxes": self.visible_gt_boxes,
                "matched_visible_gt_boxes": self.matched_visible_gt_boxes,
                "matched_gt_instances": self.matched_gt_instances,
                "fragmented_gt_instances": self.fragmented_gt_instances,
                "prediction_boxes": self.prediction_boxes,
                "unmatched_predictions": self.unmatched_predictions,
            },
        }
        if include_frames:
            result["frames"] = [frame.to_dict() for frame in self.frames]
        return result


@dataclass(frozen=True, slots=True)
class PromptCandidateScore:
    candidate_id: str
    score: float
    evaluation: DetectionEvaluation

    def to_dict(self, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "candidate_id": self.candidate_id,
            "score": self.score,
            "metrics": self.evaluation.metrics_dict(),
        }


@dataclass(frozen=True, slots=True)
class _GroundTruthFrame:
    sequence_id: str
    frame_id: str
    instances: tuple[GroundTruthInstance, ...]


@dataclass(frozen=True, slots=True)
class _PredictionFrame:
    sequence_id: str
    frame_id: str
    predictions: tuple[Prediction, ...]


def intersection_over_union(a: BoundingBox, b: BoundingBox) -> float:
    """返回两个合法框的 IoU。"""

    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def load_json_document(path: str | Path) -> Mapping[str, Any]:
    """读取单个 JSON object；JSONL 不属于本评测契约。"""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise EvaluationInputError(f"{path}: top-level JSON must be an object")
    return value


def evaluate_detection(
    ground_truth: Mapping[str, Any], predictions: Mapping[str, Any]
) -> DetectionEvaluation:
    """按固定口径评测一份预测。"""

    gt_dataset_id = _dataset_id(ground_truth, "ground truth")
    pred_dataset_id = _dataset_id(predictions, "predictions")
    if gt_dataset_id != pred_dataset_id:
        raise EvaluationInputError(
            f"dataset_id mismatch: ground truth={gt_dataset_id!r}, predictions={pred_dataset_id!r}"
        )

    gt_frames = _parse_ground_truth_frames(ground_truth)
    pred_frames = _parse_prediction_frames(predictions)
    gt_by_key = {(frame.sequence_id, frame.frame_id): frame for frame in gt_frames}
    pred_by_key = {(frame.sequence_id, frame.frame_id): frame for frame in pred_frames}
    extra_prediction_frames = sorted(set(pred_by_key) - set(gt_by_key))
    if extra_prediction_frames:
        rendered = ", ".join(_render_frame_key(key) for key in extra_prediction_frames)
        raise EvaluationInputError(f"predictions contain frames absent from ground truth: {rendered}")

    visible_gt_boxes = 0
    matched_visible_gt_boxes = 0
    prediction_boxes = 0
    unmatched_predictions = 0
    tracks_by_gt_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    frame_results: list[FrameEvaluation] = []

    for key in sorted(gt_by_key):
        gt_frame = gt_by_key[key]
        pred_frame = pred_by_key.get(key)
        frame_predictions = pred_frame.predictions if pred_frame else ()
        visible_instances = tuple(instance for instance in gt_frame.instances if instance.visible)
        matches, unmatched = _match_frame(visible_instances, frame_predictions)

        for match in matches:
            tracks_by_gt_identity[(match.sequence_id, match.instance_id)].add(match.track_id)

        visible_gt_boxes += len(visible_instances)
        matched_visible_gt_boxes += len(matches)
        prediction_boxes += len(frame_predictions)
        unmatched_predictions += len(unmatched)
        frame_results.append(
            FrameEvaluation(
                sequence_id=gt_frame.sequence_id,
                frame_id=gt_frame.frame_id,
                visible_gt_boxes=len(visible_instances),
                prediction_boxes=len(frame_predictions),
                matches=matches,
                unmatched_prediction_track_ids=tuple(
                    prediction.track_id for prediction in unmatched
                ),
            )
        )

    if not frame_results:
        raise EvaluationInputError("ground truth must contain at least one frame")
    if visible_gt_boxes == 0:
        raise EvaluationInputError("ground truth must contain at least one visible instance")

    matched_gt_instances = len(tracks_by_gt_identity)
    fragmented_gt_instances = sum(
        1 for track_ids in tracks_by_gt_identity.values() if len(track_ids) > 1
    )
    return DetectionEvaluation(
        dataset_id=gt_dataset_id,
        frame_count=len(frame_results),
        visible_gt_boxes=visible_gt_boxes,
        matched_visible_gt_boxes=matched_visible_gt_boxes,
        matched_gt_instances=matched_gt_instances,
        fragmented_gt_instances=fragmented_gt_instances,
        prediction_boxes=prediction_boxes,
        unmatched_predictions=unmatched_predictions,
        frames=tuple(frame_results),
    )


def evaluate_detection_files(
    ground_truth_path: str | Path, predictions_path: str | Path
) -> DetectionEvaluation:
    return evaluate_detection(
        load_json_document(ground_truth_path), load_json_document(predictions_path)
    )


def score_prompt(
    evaluation: DetectionEvaluation, *, lambda_fp: float, mu_fragmentation: float
) -> float:
    """score = recall - lambda_fp * FP/frame - mu_fragmentation * fragmentation。"""

    lambda_fp = _nonnegative_finite(lambda_fp, "lambda_fp")
    mu_fragmentation = _nonnegative_finite(mu_fragmentation, "mu_fragmentation")
    return (
        evaluation.visible_instance_recall
        - lambda_fp * evaluation.false_positives_per_frame
        - mu_fragmentation * evaluation.fragmentation_rate
    )


def rank_prompt_candidates(
    ground_truth: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    lambda_fp: float,
    mu_fragmentation: float,
) -> tuple[PromptCandidateScore, ...]:
    """评测并确定性排序候选；同分时 candidate_id 字典序升序。"""

    if not candidates:
        raise EvaluationInputError("at least one prompt candidate is required")
    lambda_fp = _nonnegative_finite(lambda_fp, "lambda_fp")
    mu_fragmentation = _nonnegative_finite(mu_fragmentation, "mu_fragmentation")

    scored: list[PromptCandidateScore] = []
    for candidate_id, candidate_predictions in sorted(candidates.items()):
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise EvaluationInputError("candidate_id must be a non-empty string")
        if not isinstance(candidate_predictions, Mapping):
            raise EvaluationInputError(f"candidate {candidate_id!r}: predictions must be an object")
        evaluation = evaluate_detection(ground_truth, candidate_predictions)
        scored.append(
            PromptCandidateScore(
                candidate_id=candidate_id,
                score=score_prompt(
                    evaluation,
                    lambda_fp=lambda_fp,
                    mu_fragmentation=mu_fragmentation,
                ),
                evaluation=evaluation,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.candidate_id))
    return tuple(scored)


def _match_frame(
    ground_truth: Sequence[GroundTruthInstance], predictions: Sequence[Prediction]
) -> tuple[tuple[Match, ...], tuple[Prediction, ...]]:
    """最大基数二分匹配；候选边以 IoU 降序和稳定 ID 次序遍历。"""

    ordered_gt = tuple(sorted(ground_truth, key=lambda item: item.sort_key))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.sort_key))
    adjacency: dict[int, tuple[tuple[int, float], ...]] = {}
    for gt_index, gt in enumerate(ordered_gt):
        candidates: list[tuple[int, float]] = []
        for pred_index, prediction in enumerate(ordered_predictions):
            if gt.canonical_id != prediction.canonical_id:
                continue
            overlap = intersection_over_union(gt.bbox, prediction.bbox)
            if overlap >= IOU_THRESHOLD:
                candidates.append((pred_index, overlap))
        candidates.sort(
            key=lambda item: (-item[1], ordered_predictions[item[0]].sort_key)
        )
        adjacency[gt_index] = tuple(candidates)

    prediction_owner: dict[int, int] = {}
    prediction_by_gt: dict[int, int] = {}

    def augment(gt_index: int, seen_predictions: set[int]) -> bool:
        for pred_index, _overlap in adjacency[gt_index]:
            if pred_index in seen_predictions:
                continue
            seen_predictions.add(pred_index)
            previous_owner = prediction_owner.get(pred_index)
            if previous_owner is not None and not augment(previous_owner, seen_predictions):
                continue
            prediction_owner[pred_index] = gt_index
            prediction_by_gt[gt_index] = pred_index
            if (
                previous_owner is not None
                and prediction_by_gt.get(previous_owner) == pred_index
            ):
                del prediction_by_gt[previous_owner]
            return True
        return False

    for gt_index in range(len(ordered_gt)):
        augment(gt_index, set())

    matches: list[Match] = []
    for gt_index, pred_index in sorted(prediction_by_gt.items()):
        gt = ordered_gt[gt_index]
        prediction = ordered_predictions[pred_index]
        matches.append(
            Match(
                sequence_id=gt.sequence_id,
                frame_id=gt.frame_id,
                instance_id=gt.instance_id,
                track_id=prediction.track_id,
                canonical_id=gt.canonical_id,
                iou=intersection_over_union(gt.bbox, prediction.bbox),
            )
        )
    matches.sort(key=lambda item: (item.instance_id, item.track_id, -item.iou))
    unmatched = tuple(
        prediction
        for pred_index, prediction in enumerate(ordered_predictions)
        if pred_index not in prediction_owner
    )
    return tuple(matches), unmatched


def _dataset_id(document: Mapping[str, Any], label: str) -> str:
    dataset_id = document.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise EvaluationInputError(f"{label}: dataset_id must be a non-empty string")
    lowered = dataset_id.casefold()
    marker = next((value for value in _FORBIDDEN_DATASET_MARKERS if value in lowered), None)
    if marker is not None:
        raise EvaluationInputError(
            f"{label}: forbidden task B dataset_id marker {marker!r} in {dataset_id!r}"
        )
    return dataset_id


def _parse_ground_truth_frames(document: Mapping[str, Any]) -> tuple[_GroundTruthFrame, ...]:
    raw_frames = _list_field(document, "frames", "ground truth")
    seen_frames: set[tuple[str, str]] = set()
    canonical_by_identity: dict[tuple[str, str], str] = {}
    frames: list[_GroundTruthFrame] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        path = f"ground truth.frames[{frame_index}]"
        frame = _mapping(raw_frame, path)
        sequence_id, frame_id = _frame_identity(frame, path)
        frame_key = sequence_id, frame_id
        if frame_key in seen_frames:
            raise EvaluationInputError(f"{path}: duplicate frame {_render_frame_key(frame_key)}")
        seen_frames.add(frame_key)

        raw_instances = _list_field(frame, "instances", path)
        seen_instances: set[str] = set()
        instances: list[GroundTruthInstance] = []
        for instance_index, raw_instance in enumerate(raw_instances):
            instance_path = f"{path}.instances[{instance_index}]"
            item = _mapping(raw_instance, instance_path)
            instance_id = _identifier(item.get("instance_id"), f"{instance_path}.instance_id")
            canonical_id = _identifier(item.get("canonical_id"), f"{instance_path}.canonical_id")
            if instance_id in seen_instances:
                raise EvaluationInputError(
                    f"{instance_path}: duplicate instance_id {instance_id!r} in one frame"
                )
            seen_instances.add(instance_id)
            identity = sequence_id, instance_id
            prior_canonical = canonical_by_identity.setdefault(identity, canonical_id)
            if prior_canonical != canonical_id:
                raise EvaluationInputError(
                    f"{instance_path}: instance {instance_id!r} changes canonical_id "
                    f"from {prior_canonical!r} to {canonical_id!r}"
                )
            visible = item.get("visible", True)
            if not isinstance(visible, bool):
                raise EvaluationInputError(f"{instance_path}.visible must be boolean")
            instances.append(
                GroundTruthInstance(
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    instance_id=instance_id,
                    canonical_id=canonical_id,
                    bbox=_bbox(item.get("bbox"), f"{instance_path}.bbox"),
                    visible=visible,
                )
            )
        frames.append(
            _GroundTruthFrame(
                sequence_id=sequence_id,
                frame_id=frame_id,
                instances=tuple(sorted(instances, key=lambda item: item.sort_key)),
            )
        )
    frames.sort(key=lambda item: (item.sequence_id, item.frame_id))
    return tuple(frames)


def _parse_prediction_frames(document: Mapping[str, Any]) -> tuple[_PredictionFrame, ...]:
    raw_frames = _list_field(document, "frames", "predictions")
    seen_frames: set[tuple[str, str]] = set()
    frames: list[_PredictionFrame] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        path = f"predictions.frames[{frame_index}]"
        frame = _mapping(raw_frame, path)
        sequence_id, frame_id = _frame_identity(frame, path)
        frame_key = sequence_id, frame_id
        if frame_key in seen_frames:
            raise EvaluationInputError(f"{path}: duplicate frame {_render_frame_key(frame_key)}")
        seen_frames.add(frame_key)

        raw_predictions = _list_field(frame, "predictions", path)
        seen_track_ids: set[str] = set()
        parsed_predictions: list[Prediction] = []
        for prediction_index, raw_prediction in enumerate(raw_predictions):
            prediction_path = f"{path}.predictions[{prediction_index}]"
            item = _mapping(raw_prediction, prediction_path)
            track_id = _identifier(item.get("track_id"), f"{prediction_path}.track_id")
            canonical_id = _identifier(
                item.get("canonical_id"), f"{prediction_path}.canonical_id"
            )
            if track_id in seen_track_ids:
                raise EvaluationInputError(
                    f"{prediction_path}: duplicate track_id {track_id!r} in one frame"
                )
            seen_track_ids.add(track_id)
            parsed_predictions.append(
                Prediction(
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    track_id=track_id,
                    canonical_id=canonical_id,
                    bbox=_bbox(item.get("bbox"), f"{prediction_path}.bbox"),
                )
            )
        frames.append(
            _PredictionFrame(
                sequence_id=sequence_id,
                frame_id=frame_id,
                predictions=tuple(
                    sorted(parsed_predictions, key=lambda item: item.sort_key)
                ),
            )
        )
    frames.sort(key=lambda item: (item.sequence_id, item.frame_id))
    return tuple(frames)


def _frame_identity(frame: Mapping[str, Any], path: str) -> tuple[str, str]:
    sequence_id = frame.get("sequence_id", "")
    if not isinstance(sequence_id, str):
        raise EvaluationInputError(f"{path}.sequence_id must be a string")
    frame_id = _identifier(frame.get("frame_id"), f"{path}.frame_id")
    return sequence_id, frame_id


def _bbox(value: Any, path: str) -> BoundingBox:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise EvaluationInputError(f"{path} must be [x1, y1, x2, y2]")
    coordinates: list[float] = []
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise EvaluationInputError(f"{path}[{index}] must be numeric")
        number = float(coordinate)
        if not math.isfinite(number):
            raise EvaluationInputError(f"{path}[{index}] must be finite")
        coordinates.append(number)
    box = BoundingBox(*coordinates)
    if box.x2 <= box.x1 or box.y2 <= box.y1:
        raise EvaluationInputError(f"{path} must satisfy x2 > x1 and y2 > y1")
    return box


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationInputError(f"{path} must be an object")
    return value


def _list_field(document: Mapping[str, Any], key: str, path: str) -> Sequence[Any]:
    value = document.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvaluationInputError(f"{path}.{key} must be an array")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationInputError(f"{path} must be a non-empty string")
    return value


def _nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationInputError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise EvaluationInputError(f"{name} must be finite and >= 0")
    return result


def _render_frame_key(key: tuple[str, str]) -> str:
    sequence_id, frame_id = key
    return f"{sequence_id}/{frame_id}" if sequence_id else frame_id

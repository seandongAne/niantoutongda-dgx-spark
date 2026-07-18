#!/usr/bin/env python
"""Compare two audited Grounding DINO top-k captures at decision level.

This is a diagnostic comparator, not a replacement for the frozen SF1
acceptance scorer.  It validates capture provenance, compares raw query-ordered
tensors, measures proposal selection overlap, then aligns decoder queries by
their originating encoder proposal id.  Post-processed detections are matched
exactly and deterministically within each text label by maximum total IoU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


CAPTURE_NPZ_NAME = "stage-boundaries.npz"
CAPTURE_SUMMARY_NAME = "summary.json"
INPUT_NAMES = (
    "pixel_values",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "pixel_mask",
)
MODEL_ARTIFACT_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".safetensors",
    ".txt",
    ".vocab",
}
REQUIRED_CAPTURE_ARRAYS = (
    "topk_indices",
    "topk_selected_class_logits",
    "topk_selected_coord_logits",
    "topk_selected_boxes",
    "final_logits",
    "final_pred_boxes",
)
ALIGNED_ARRAYS = (
    "topk_selected_class_logits",
    "topk_selected_coord_logits",
    "topk_selected_boxes",
    "final_logits",
    "final_pred_boxes",
)
STRICT_IOU_MIN = 0.999
STRICT_SCORE_DELTA_MAX = 1e-3
STRICT_BOX_DELTA_PX_MAX = 0.5
DIAGNOSTIC_IOU_MIN = 0.99
DIAGNOSTIC_SCORE_DELTA_MAX = 1e-2
TIE_EPSILON = 1e-15


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(project: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_float(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.size),
        "sha256": _sha256_array(value),
    }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _model_artifact_hashes(model_dir: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_ARTIFACT_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"no model artifacts found under {model_dir}")
    records: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(model_dir).as_posix()
        digest = _sha256_file(path)
        size = path.stat().st_size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        records.append({"path": relative, "bytes": size, "sha256": digest})
    return {"combined_sha256": aggregate.hexdigest(), "files": records}


def _validate_recorded_array(
    name: str, value: np.ndarray, descriptors: Mapping[str, Any], capture_label: str
) -> None:
    recorded = _require_mapping(
        descriptors.get(name), f"{capture_label}.artifacts.npz_arrays.{name}"
    )
    actual = _array_descriptor(value)
    for key in ("shape", "dtype", "numel", "sha256"):
        if recorded.get(key) != actual[key]:
            raise RuntimeError(
                f"{capture_label} array descriptor mismatch for {name}.{key}: "
                f"recorded={recorded.get(key)!r}, actual={actual[key]!r}"
            )


def _validate_capture_shapes(arrays: Mapping[str, np.ndarray], label: str) -> None:
    indices = arrays["topk_indices"]
    if indices.ndim != 2 or indices.dtype.kind not in "iu":
        raise RuntimeError(
            f"{label} topk_indices must be a rank-2 integer array; "
            f"got shape={indices.shape}, dtype={indices.dtype}"
        )
    batch_size, query_count = indices.shape
    if batch_size < 1 or query_count < 1:
        raise RuntimeError(f"{label} topk_indices cannot have an empty dimension")
    for batch_index, row in enumerate(indices):
        proposal_ids = [int(item) for item in row]
        if any(item < 0 for item in proposal_ids):
            raise RuntimeError(f"{label} batch {batch_index} has a negative proposal id")
        if len(set(proposal_ids)) != query_count:
            raise RuntimeError(
                f"{label} batch {batch_index} topk proposal ids are not unique"
            )

    for name in ALIGNED_ARRAYS:
        value = arrays[name]
        if value.ndim < 3 or value.shape[:2] != (batch_size, query_count):
            raise RuntimeError(
                f"{label} {name} must start with [batch, query]="
                f"[{batch_size}, {query_count}]; got {value.shape}"
            )
    if arrays["final_pred_boxes"].shape[-1] != 4:
        raise RuntimeError(f"{label} final_pred_boxes must have a last dimension of 4")
    if arrays["topk_selected_coord_logits"].shape[-1] != 4:
        raise RuntimeError(
            f"{label} topk_selected_coord_logits must have a last dimension of 4"
        )
    if arrays["topk_selected_boxes"].shape[-1] != 4:
        raise RuntimeError(
            f"{label} topk_selected_boxes must have a last dimension of 4"
        )


def _load_capture(path: Path, label: str) -> dict[str, Any]:
    capture_dir = path.resolve()
    if not capture_dir.is_dir():
        raise NotADirectoryError(capture_dir)
    summary_path = capture_dir / CAPTURE_SUMMARY_NAME
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")), f"{label} summary"
    )
    if summary.get("probe") != "gdino_topk_stage_probe":
        raise RuntimeError(
            f"{label} is not an audited gdino_topk_stage_probe capture: "
            f"{summary.get('probe')!r}"
        )
    artifacts = _require_mapping(summary.get("artifacts"), f"{label}.artifacts")
    recorded_name = _require_nonempty_string(
        artifacts.get("npz"), f"{label}.artifacts.npz"
    )
    if Path(recorded_name).name != recorded_name or recorded_name != CAPTURE_NPZ_NAME:
        raise RuntimeError(
            f"{label} must reference the direct {CAPTURE_NPZ_NAME} artifact; "
            f"got {recorded_name!r}"
        )
    npz_path = capture_dir / recorded_name
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    actual_npz_sha256 = _sha256_file(npz_path)
    recorded_npz_sha256 = _require_nonempty_string(
        artifacts.get("npz_sha256"), f"{label}.artifacts.npz_sha256"
    )
    if recorded_npz_sha256 != actual_npz_sha256:
        raise RuntimeError(
            f"{label} NPZ hash mismatch: recorded={recorded_npz_sha256}, "
            f"actual={actual_npz_sha256}"
        )

    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = sorted(set(REQUIRED_CAPTURE_ARRAYS) - set(loaded.files))
        if missing:
            raise KeyError(f"{label} capture is missing arrays: {missing}")
        arrays = {
            name: np.array(loaded[name], copy=True) for name in REQUIRED_CAPTURE_ARRAYS
        }
    descriptors = _require_mapping(
        artifacts.get("npz_arrays"), f"{label}.artifacts.npz_arrays"
    )
    for name, value in arrays.items():
        _validate_recorded_array(name, value, descriptors, label)
    _validate_capture_shapes(arrays, label)
    return {
        "directory": capture_dir,
        "summary_path": summary_path,
        "summary_sha256": _sha256_file(summary_path),
        "summary": summary,
        "npz_path": npz_path,
        "npz_sha256": actual_npz_sha256,
        "arrays": arrays,
    }


def _recorded_input_sha256(capture: Mapping[str, Any], label: str) -> str:
    inputs = _require_mapping(capture["summary"].get("inputs"), f"{label}.inputs")
    return _require_nonempty_string(
        inputs.get("file_sha256"), f"{label}.inputs.file_sha256"
    )


def _recorded_manifest_sha256(capture: Mapping[str, Any], label: str) -> str:
    inputs = _require_mapping(capture["summary"].get("inputs"), f"{label}.inputs")
    manifest = _require_mapping(
        inputs.get("baseline_manifest"), f"{label}.inputs.baseline_manifest"
    )
    return _require_nonempty_string(
        manifest.get("sha256"), f"{label}.inputs.baseline_manifest.sha256"
    )


def _recorded_model_sha256(capture: Mapping[str, Any], label: str) -> str:
    model = _require_mapping(capture["summary"].get("model"), f"{label}.model")
    hashes = _require_mapping(
        model.get("artifact_hashes"), f"{label}.model.artifact_hashes"
    )
    return _require_nonempty_string(
        hashes.get("combined_sha256"),
        f"{label}.model.artifact_hashes.combined_sha256",
    )


def _finite_array_diff(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"cannot compare arrays with different shapes: {reference.shape} != "
            f"{candidate.shape}"
        )
    reference_nan = np.isnan(reference)
    candidate_nan = np.isnan(candidate)
    reference_posinf = np.isposinf(reference)
    candidate_posinf = np.isposinf(candidate)
    reference_neginf = np.isneginf(reference)
    candidate_neginf = np.isneginf(candidate)
    jointly_finite = np.isfinite(reference) & np.isfinite(candidate)
    delta = np.abs(
        reference[jointly_finite].astype(np.float64)
        - candidate[jointly_finite].astype(np.float64)
    )
    return {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "dtype_equal": reference.dtype == candidate.dtype,
        "bit_exact": bool(np.array_equal(reference, candidate)),
        "reference_nonfinite": {
            "nan": int(reference_nan.sum()),
            "posinf": int(reference_posinf.sum()),
            "neginf": int(reference_neginf.sum()),
        },
        "candidate_nonfinite": {
            "nan": int(candidate_nan.sum()),
            "posinf": int(candidate_posinf.sum()),
            "neginf": int(candidate_neginf.sum()),
        },
        "nonfinite_pattern_equal": bool(
            np.array_equal(reference_nan, candidate_nan)
            and np.array_equal(reference_posinf, candidate_posinf)
            and np.array_equal(reference_neginf, candidate_neginf)
        ),
        "jointly_finite_count": int(jointly_finite.sum()),
        "finite_abs_delta": {
            "max": _json_float(delta.max()) if delta.size else None,
            "mean": _json_float(delta.mean()) if delta.size else None,
            "p50": _json_float(np.percentile(delta, 50)) if delta.size else None,
            "p95": _json_float(np.percentile(delta, 95)) if delta.size else None,
            "p99": _json_float(np.percentile(delta, 99)) if delta.size else None,
            "rmse": _json_float(np.sqrt(np.mean(np.square(delta))))
            if delta.size
            else None,
        },
    }


def _topk_overlap(
    reference_indices: np.ndarray, candidate_indices: np.ndarray
) -> dict[str, Any]:
    if reference_indices.shape != candidate_indices.shape:
        raise RuntimeError(
            f"topk shape mismatch: {reference_indices.shape} != "
            f"{candidate_indices.shape}"
        )
    rows: list[dict[str, Any]] = []
    for batch_index, (reference_row, candidate_row) in enumerate(
        zip(reference_indices, candidate_indices, strict=True)
    ):
        reference_ids = [int(item) for item in reference_row]
        candidate_ids = [int(item) for item in candidate_row]
        reference_set = set(reference_ids)
        candidate_set = set(candidate_ids)
        common = reference_set & candidate_set
        union = reference_set | candidate_set
        mismatch_positions = [
            index
            for index, (left, right) in enumerate(
                zip(reference_ids, candidate_ids, strict=True)
            )
            if left != right
        ]
        rows.append(
            {
                "batch_index": batch_index,
                "k": len(reference_ids),
                "ordered_equal": reference_ids == candidate_ids,
                "same_rank_count": len(reference_ids) - len(mismatch_positions),
                "same_rank_fraction": (
                    (len(reference_ids) - len(mismatch_positions)) / len(reference_ids)
                ),
                "mismatch_position_count": len(mismatch_positions),
                "mismatch_positions": mismatch_positions,
                "set_equal": reference_set == candidate_set,
                "set_overlap_count": len(common),
                "set_overlap_fraction_of_k": len(common) / len(reference_set),
                "set_jaccard": len(common) / len(union) if union else 1.0,
                "reference_only_proposal_ids": sorted(reference_set - candidate_set),
                "candidate_only_proposal_ids": sorted(candidate_set - reference_set),
            }
        )
    return {
        "ordered_equal_all_batches": all(row["ordered_equal"] for row in rows),
        "sets_equal_all_batches": all(row["set_equal"] for row in rows),
        "per_batch": rows,
    }


def _proposal_id_aligned_diffs(
    reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    reference_indices = reference["topk_indices"]
    candidate_indices = candidate["topk_indices"]
    rows: list[dict[str, Any]] = []
    for batch_index in range(reference_indices.shape[0]):
        reference_rank = {
            int(proposal_id): rank
            for rank, proposal_id in enumerate(reference_indices[batch_index])
        }
        candidate_rank = {
            int(proposal_id): rank
            for rank, proposal_id in enumerate(candidate_indices[batch_index])
        }
        common_ids = sorted(reference_rank.keys() & candidate_rank.keys())
        reference_positions = np.asarray(
            [reference_rank[item] for item in common_ids], dtype=np.int64
        )
        candidate_positions = np.asarray(
            [candidate_rank[item] for item in common_ids], dtype=np.int64
        )
        rank_delta = np.abs(reference_positions - candidate_positions)
        tensor_diffs = {
            name: _finite_array_diff(
                reference[name][batch_index, reference_positions],
                candidate[name][batch_index, candidate_positions],
            )
            for name in ALIGNED_ARRAYS
        }
        rows.append(
            {
                "batch_index": batch_index,
                "alignment_key": "encoder_proposal_id",
                "common_count": len(common_ids),
                "common_proposal_ids": common_ids,
                "reference_rank_for_common_ids": reference_positions.tolist(),
                "candidate_rank_for_common_ids": candidate_positions.tolist(),
                "rank_displacement": {
                    "same_rank_count": int((rank_delta == 0).sum()),
                    "max_abs": int(rank_delta.max()) if rank_delta.size else None,
                    "mean_abs": _json_float(rank_delta.mean())
                    if rank_delta.size
                    else None,
                },
                "tensor_diffs": tensor_diffs,
            }
        )
    return {"per_batch": rows}


def _postprocess(
    processor,
    torch_module,
    arrays: Mapping[str, np.ndarray],
    input_ids: np.ndarray,
    target_sizes: Sequence[Sequence[int]],
    threshold: float,
    text_threshold: float,
) -> list[list[dict[str, Any]]]:
    outputs = SimpleNamespace(
        logits=torch_module.from_numpy(arrays["final_logits"]),
        pred_boxes=torch_module.from_numpy(arrays["final_pred_boxes"]),
    )
    processed = processor.post_process_grounded_object_detection(
        outputs,
        torch_module.from_numpy(input_ids),
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )
    decisions: list[list[dict[str, Any]]] = []
    for batch_index, result in enumerate(processed):
        labels = result.get("text_labels", result.get("labels"))
        if labels is None:
            raise RuntimeError(
                f"processor result {batch_index} has neither text_labels nor labels"
            )
        if not (len(labels) == len(result["scores"]) == len(result["boxes"])):
            raise RuntimeError(f"processor result {batch_index} has inconsistent lengths")
        batch: list[dict[str, Any]] = []
        for detection_index, (label, score, box) in enumerate(
            zip(labels, result["scores"], result["boxes"], strict=True)
        ):
            score_value = float(score)
            box_values = [float(item) for item in box]
            if not math.isfinite(score_value) or not all(
                math.isfinite(item) for item in box_values
            ):
                raise RuntimeError(
                    f"processor result {batch_index}/{detection_index} is non-finite"
                )
            batch.append(
                {
                    "index": detection_index,
                    "label": str(label),
                    "score": score_value,
                    "box_xyxy_px": box_values,
                }
            )
        decisions.append(batch)
    return decisions


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 1.0 if list(left) == list(right) else 0.0
    return intersection / union


def _prefer_assignment(
    candidate_score: float,
    candidate_pairs: tuple[tuple[int, int], ...],
    current: tuple[float, tuple[tuple[int, int], ...]] | None,
) -> bool:
    if current is None:
        return True
    current_score, current_pairs = current
    if candidate_score > current_score + TIE_EPSILON:
        return True
    if abs(candidate_score - current_score) <= TIE_EPSILON:
        return candidate_pairs < current_pairs
    return False


def _exact_max_iou_assignment(
    reference_indices: Sequence[int],
    candidate_indices: Sequence[int],
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    remaining_transition_budget: int,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Match the smaller side completely using an exact bitmask dynamic program."""

    if not reference_indices or not candidate_indices:
        return [], {"estimated_transition_upper_bound": 0, "actual_transitions": 0}
    reference_indices = tuple(sorted(reference_indices))
    candidate_indices = tuple(sorted(candidate_indices))
    if len(reference_indices) <= len(candidate_indices):
        small_side = "reference"
        small = reference_indices
        large = candidate_indices
    else:
        small_side = "candidate"
        small = candidate_indices
        large = reference_indices

    small_count = len(small)
    estimated = len(large) * (1 << small_count) * (small_count + 1)
    if estimated > remaining_transition_budget:
        raise RuntimeError(
            "exact label-constrained matching exceeds the remaining DP budget: "
            f"estimated_upper_bound={estimated}, "
            f"remaining_budget={remaining_transition_budget}, "
            f"reference_count={len(reference_indices)}, "
            f"candidate_count={len(candidate_indices)}"
        )

    states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {0: (0.0, ())}
    actual_transitions = 0
    for large_index in large:
        next_states = dict(states)
        actual_transitions += len(states)
        for mask, (score, pairs) in states.items():
            for small_position, small_index in enumerate(small):
                if mask & (1 << small_position):
                    continue
                if small_side == "reference":
                    reference_index, candidate_index = small_index, large_index
                else:
                    reference_index, candidate_index = large_index, small_index
                iou = _box_iou(
                    reference[reference_index]["box_xyxy_px"],
                    candidate[candidate_index]["box_xyxy_px"],
                )
                next_mask = mask | (1 << small_position)
                next_pairs = tuple(sorted((*pairs, (reference_index, candidate_index))))
                next_score = score + iou
                current = next_states.get(next_mask)
                if _prefer_assignment(next_score, next_pairs, current):
                    next_states[next_mask] = (next_score, next_pairs)
                actual_transitions += 1
        states = next_states
    full_mask = (1 << small_count) - 1
    if full_mask not in states:
        raise RuntimeError("exact matching DP did not produce a complete smaller-side match")
    return list(states[full_mask][1]), {
        "estimated_transition_upper_bound": estimated,
        "actual_transitions": actual_transitions,
    }


def _label_counts(decisions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item["label"]) for item in decisions).items()))


def _match_image_decisions(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    transition_budget: int,
) -> dict[str, Any]:
    reference_by_label: dict[str, list[int]] = {}
    candidate_by_label: dict[str, list[int]] = {}
    for index, item in enumerate(reference):
        reference_by_label.setdefault(str(item["label"]), []).append(index)
    for index, item in enumerate(candidate):
        candidate_by_label.setdefault(str(item["label"]), []).append(index)

    all_pairs: list[tuple[int, int]] = []
    label_reports: dict[str, Any] = {}
    estimated_total = 0
    actual_total = 0
    for label in sorted(reference_by_label.keys() | candidate_by_label.keys()):
        reference_indices = reference_by_label.get(label, [])
        candidate_indices = candidate_by_label.get(label, [])
        pairs, budget = _exact_max_iou_assignment(
            reference_indices,
            candidate_indices,
            reference,
            candidate,
            remaining_transition_budget=transition_budget - estimated_total,
        )
        estimated_total += budget["estimated_transition_upper_bound"]
        actual_total += budget["actual_transitions"]
        all_pairs.extend(pairs)
        label_reports[label] = {
            "reference_count": len(reference_indices),
            "candidate_count": len(candidate_indices),
            "matched_count": len(pairs),
            **budget,
        }

    all_pairs.sort()
    matched_reference = {left for left, _ in all_pairs}
    matched_candidate = {right for _, right in all_pairs}
    pair_reports: list[dict[str, Any]] = []
    for reference_index, candidate_index in all_pairs:
        reference_item = reference[reference_index]
        candidate_item = candidate[candidate_index]
        if reference_item["label"] != candidate_item["label"]:
            raise AssertionError("internal label-constrained matching invariant failed")
        iou = _box_iou(
            reference_item["box_xyxy_px"], candidate_item["box_xyxy_px"]
        )
        score_delta = candidate_item["score"] - reference_item["score"]
        box_delta = [
            right - left
            for left, right in zip(
                reference_item["box_xyxy_px"],
                candidate_item["box_xyxy_px"],
                strict=True,
            )
        ]
        pair_reports.append(
            {
                "reference_index": reference_index,
                "candidate_index": candidate_index,
                "label": reference_item["label"],
                "iou": iou,
                "reference_score": reference_item["score"],
                "candidate_score": candidate_item["score"],
                "score_delta_candidate_minus_reference": score_delta,
                "score_abs_delta": abs(score_delta),
                "reference_box_xyxy_px": reference_item["box_xyxy_px"],
                "candidate_box_xyxy_px": candidate_item["box_xyxy_px"],
                "box_delta_candidate_minus_reference_px": box_delta,
                "box_max_abs_delta_px": max(abs(item) for item in box_delta),
            }
        )

    unmatched_reference = sorted(set(range(len(reference))) - matched_reference)
    unmatched_candidate = sorted(set(range(len(candidate))) - matched_candidate)
    complete = not unmatched_reference and not unmatched_candidate
    strict_pair_pass = all(
        item["iou"] >= STRICT_IOU_MIN
        and item["score_abs_delta"] <= STRICT_SCORE_DELTA_MAX
        and item["box_max_abs_delta_px"] <= STRICT_BOX_DELTA_PX_MAX
        for item in pair_reports
    )
    diagnostic_pair_pass = all(
        item["iou"] >= DIAGNOSTIC_IOU_MIN
        and item["score_abs_delta"] <= DIAGNOSTIC_SCORE_DELTA_MAX
        for item in pair_reports
    )
    ious = [item["iou"] for item in pair_reports]
    score_deltas = [item["score_abs_delta"] for item in pair_reports]
    box_deltas = [item["box_max_abs_delta_px"] for item in pair_reports]
    return {
        "counts": {
            "reference_total": len(reference),
            "candidate_total": len(candidate),
            "matched": len(pair_reports),
            "reference_by_label": _label_counts(reference),
            "candidate_by_label": _label_counts(candidate),
        },
        "matching": {
            "method": "exact_label_partitioned_maximum_total_iou_bitmask_dp",
            "tie_break": "lexicographically_smallest_(reference_index,candidate_index)_pairs",
            "objective_total_iou": sum(ious),
            "transition_budget": transition_budget,
            "estimated_transition_upper_bound": estimated_total,
            "actual_transitions": actual_total,
            "per_label": label_reports,
        },
        "matched_pairs": pair_reports,
        "unmatched_reference_indices": unmatched_reference,
        "unmatched_candidate_indices": unmatched_candidate,
        "delta_summary": {
            "min_iou": min(ious) if ious else None,
            "mean_iou": _json_float(sum(ious) / len(ious)) if ious else None,
            "max_score_abs_delta": max(score_deltas) if score_deltas else None,
            "max_box_abs_delta_px": max(box_deltas) if box_deltas else None,
        },
        "gates": {
            "strict": {
                "pass": complete and strict_pair_pass,
                "complete_one_to_one_label_match": complete,
                "criteria": {
                    "iou_min_inclusive": STRICT_IOU_MIN,
                    "score_abs_delta_max_inclusive": STRICT_SCORE_DELTA_MAX,
                    "box_abs_delta_px_max_inclusive": STRICT_BOX_DELTA_PX_MAX,
                },
            },
            "diagnostic": {
                "pass": complete and diagnostic_pair_pass,
                "complete_one_to_one_label_match": complete,
                "criteria": {
                    "iou_min_inclusive": DIAGNOSTIC_IOU_MIN,
                    "score_abs_delta_max_inclusive": DIAGNOSTIC_SCORE_DELTA_MAX,
                },
            },
        },
    }


def _validate_target_sizes(value: Any, batch_size: int) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != batch_size:
        raise RuntimeError(
            f"baseline target_sizes must contain {batch_size} rows; got {value!r}"
        )
    target_sizes: list[list[int]] = []
    for index, row in enumerate(value):
        if (
            not isinstance(row, list)
            or len(row) != 2
            or any(not isinstance(item, int) or item <= 0 for item in row)
        ):
            raise RuntimeError(f"invalid target_sizes[{index}]: {row!r}")
        target_sizes.append([int(row[0]), int(row[1])])
    return target_sizes


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f".{path.name}.",
            dir=path.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        # A hard link is an atomic exclusive create: unlike os.replace(), it
        # cannot overwrite a file that appeared after the initial check.
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        path.chmod(0o644)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compare two Grounding DINO stage captures. Results are "
            "diagnostic only and do not replace frozen acceptance."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True, help="Frozen sample_inputs.npz")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--reference-capture", required=True)
    parser.add_argument("--candidate-capture", required=True)
    parser.add_argument("--output", required=True, help="New JSON output path")
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--max-match-transitions",
        type=int,
        default=5_000_000,
        help="Fail closed before exact matching exceeds this conservative DP budget.",
    )
    parser.add_argument("--code-commit")
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be in [0, 1]")
    if not 0.0 <= args.text_threshold <= 1.0:
        parser.error("text-threshold must be in [0, 1]")
    if args.max_match_transitions < 1:
        parser.error("max-match-transitions must be positive")

    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    model_dir = Path(args.model_dir).resolve()
    inputs_path = Path(args.inputs).resolve()
    manifest_path = Path(args.baseline_manifest).resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    reference_capture = _load_capture(Path(args.reference_capture), "reference")
    candidate_capture = _load_capture(Path(args.candidate_capture), "candidate")
    reference_arrays = reference_capture["arrays"]
    candidate_arrays = candidate_capture["arrays"]
    for name in REQUIRED_CAPTURE_ARRAYS:
        if reference_arrays[name].shape != candidate_arrays[name].shape:
            raise RuntimeError(
                f"capture shape mismatch for {name}: {reference_arrays[name].shape} != "
                f"{candidate_arrays[name].shape}"
            )

    actual_input_sha256 = _sha256_file(inputs_path)
    reference_input_sha256 = _recorded_input_sha256(reference_capture, "reference")
    candidate_input_sha256 = _recorded_input_sha256(candidate_capture, "candidate")
    if not (
        actual_input_sha256
        == reference_input_sha256
        == candidate_input_sha256
    ):
        raise RuntimeError(
            "frozen input hash mismatch across explicit input/reference/candidate: "
            f"{actual_input_sha256}, {reference_input_sha256}, "
            f"{candidate_input_sha256}"
        )

    actual_manifest_sha256 = _sha256_file(manifest_path)
    reference_manifest_sha256 = _recorded_manifest_sha256(
        reference_capture, "reference"
    )
    candidate_manifest_sha256 = _recorded_manifest_sha256(
        candidate_capture, "candidate"
    )
    if not (
        actual_manifest_sha256
        == reference_manifest_sha256
        == candidate_manifest_sha256
    ):
        raise RuntimeError(
            "baseline manifest hash mismatch across explicit manifest/reference/candidate: "
            f"{actual_manifest_sha256}, {reference_manifest_sha256}, "
            f"{candidate_manifest_sha256}"
        )

    reference_model_sha256 = _recorded_model_sha256(reference_capture, "reference")
    candidate_model_sha256 = _recorded_model_sha256(candidate_capture, "candidate")
    if reference_model_sha256 != candidate_model_sha256:
        raise RuntimeError(
            "reference and candidate model hashes differ: "
            f"{reference_model_sha256} != {candidate_model_sha256}"
        )
    model_artifact_hashes = _model_artifact_hashes(model_dir)
    if model_artifact_hashes["combined_sha256"] != reference_model_sha256:
        raise RuntimeError(
            "explicit model-dir does not match capture model artifacts: "
            f"{model_artifact_hashes['combined_sha256']} != {reference_model_sha256}"
        )
    reference_model_config = _require_mapping(
        reference_capture["summary"].get("model"), "reference.model"
    ).get("config")
    candidate_model_config = _require_mapping(
        candidate_capture["summary"].get("model"), "candidate.model"
    ).get("config")
    if reference_model_config != candidate_model_config:
        raise RuntimeError("reference and candidate captured model configs differ")

    with np.load(inputs_path, allow_pickle=False) as frozen:
        missing_inputs = sorted(set(INPUT_NAMES) - set(frozen.files))
        if missing_inputs:
            raise KeyError(f"frozen inputs are missing arrays: {missing_inputs}")
        input_ids = np.array(frozen["input_ids"], copy=True)
        frozen_input_manifest = {
            name: _array_descriptor(np.asarray(frozen[name])) for name in INPUT_NAMES
        }
    for label, capture in (
        ("reference", reference_capture),
        ("candidate", candidate_capture),
    ):
        recorded_arrays = _require_mapping(
            _require_mapping(capture["summary"].get("inputs"), f"{label}.inputs").get(
                "arrays"
            ),
            f"{label}.inputs.arrays",
        )
        for name in INPUT_NAMES:
            if recorded_arrays.get(name) != frozen_input_manifest[name]:
                raise RuntimeError(
                    f"{label} frozen input descriptor mismatch for {name}"
                )

    batch_size = int(reference_arrays["final_logits"].shape[0])
    if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
        raise RuntimeError(
            f"input_ids batch does not match capture batch: {input_ids.shape} vs "
            f"{batch_size}"
        )
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), "baseline manifest"
    )
    target_sizes = _validate_target_sizes(manifest.get("target_sizes"), batch_size)
    if manifest.get("batch_size") != batch_size:
        raise RuntimeError(
            f"baseline batch_size={manifest.get('batch_size')!r} does not match "
            f"capture batch_size={batch_size}"
        )

    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    reference_decisions = _postprocess(
        processor,
        torch,
        reference_arrays,
        input_ids,
        target_sizes,
        args.threshold,
        args.text_threshold,
    )
    candidate_decisions = _postprocess(
        processor,
        torch,
        candidate_arrays,
        input_ids,
        target_sizes,
        args.threshold,
        args.text_threshold,
    )
    if len(reference_decisions) != batch_size or len(candidate_decisions) != batch_size:
        raise RuntimeError("processor output batch size does not match the captures")

    per_batch_decisions: list[dict[str, Any]] = []
    remaining_match_transitions = args.max_match_transitions
    for batch_index, (reference_batch, candidate_batch) in enumerate(
        zip(reference_decisions, candidate_decisions, strict=True)
    ):
        comparison = _match_image_decisions(
            reference_batch,
            candidate_batch,
            transition_budget=remaining_match_transitions,
        )
        remaining_match_transitions -= comparison["matching"][
            "estimated_transition_upper_bound"
        ]
        per_batch_decisions.append(
            {
                "batch_index": batch_index,
                "target_size_hw": target_sizes[batch_index],
                "reference_detections": reference_batch,
                "candidate_detections": candidate_batch,
                **comparison,
            }
        )

    strict_pass = all(row["gates"]["strict"]["pass"] for row in per_batch_decisions)
    diagnostic_pass = all(
        row["gates"]["diagnostic"]["pass"] for row in per_batch_decisions
    )
    project = Path(__file__).resolve().parent.parent
    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "gdino_capture_decision_compare",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict_scope": "diagnostic_only_does_not_replace_frozen_acceptance",
        "acceptance_claim": False,
        "provenance_validation": {
            "same_frozen_input_sha256": True,
            "same_baseline_manifest_sha256": True,
            "same_model_artifact_sha256": True,
            "explicit_inputs": {
                "path": str(inputs_path),
                "sha256": actual_input_sha256,
                "arrays": frozen_input_manifest,
            },
            "baseline_manifest": {
                "path": str(manifest_path),
                "sha256": actual_manifest_sha256,
            },
            "model": {
                "directory": str(model_dir),
                "combined_sha256": model_artifact_hashes["combined_sha256"],
                "artifact_file_count": len(model_artifact_hashes["files"]),
            },
            "reference_capture": {
                "directory": str(reference_capture["directory"]),
                "summary_sha256": reference_capture["summary_sha256"],
                "npz_sha256": reference_capture["npz_sha256"],
            },
            "candidate_capture": {
                "directory": str(candidate_capture["directory"]),
                "summary_sha256": candidate_capture["summary_sha256"],
                "npz_sha256": candidate_capture["npz_sha256"],
            },
        },
        "postprocess": {
            "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
            "threshold": args.threshold,
            "text_threshold": args.text_threshold,
            "target_sizes_hw": target_sizes,
        },
        "raw_query_ordered_tensor_diffs": {
            "semantics": "rank i is compared directly to rank i without proposal realignment",
            "final_logits": _finite_array_diff(
                reference_arrays["final_logits"], candidate_arrays["final_logits"]
            ),
            "final_pred_boxes": _finite_array_diff(
                reference_arrays["final_pred_boxes"],
                candidate_arrays["final_pred_boxes"],
            ),
        },
        "topk_selection": _topk_overlap(
            reference_arrays["topk_indices"], candidate_arrays["topk_indices"]
        ),
        "proposal_id_aligned_query_diffs": _proposal_id_aligned_diffs(
            reference_arrays, candidate_arrays
        ),
        "decision_matching": {
            "constraint": "exact_text_label_equality",
            "objective": "maximum_total_iou_with_complete_smaller_side_matching",
            "global_transition_budget": {
                "configured": args.max_match_transitions,
                "conservative_estimate_consumed": (
                    args.max_match_transitions - remaining_match_transitions
                ),
                "remaining": remaining_match_transitions,
            },
            "per_batch": per_batch_decisions,
        },
        "diagnostic_gates": {
            "strict": {
                "pass": strict_pass,
                "criteria": {
                    "complete_one_to_one_label_match": True,
                    "iou_min_inclusive": STRICT_IOU_MIN,
                    "score_abs_delta_max_inclusive": STRICT_SCORE_DELTA_MAX,
                    "box_abs_delta_px_max_inclusive": STRICT_BOX_DELTA_PX_MAX,
                },
            },
            "diagnostic": {
                "pass": diagnostic_pass,
                "criteria": {
                    "complete_one_to_one_label_match": True,
                    "iou_min_inclusive": DIAGNOSTIC_IOU_MIN,
                    "score_abs_delta_max_inclusive": DIAGNOSTIC_SCORE_DELTA_MAX,
                },
            },
            "acceptance_boundary": (
                "These gates are diagnostic evidence only. They do not tune, replace, "
                "or confer a PASS on the independent frozen SF1 acceptance scorer."
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
        },
        "code": {
            "commit": args.code_commit or _git_commit(project),
            "script": str(script_path),
            "script_sha256": _sha256_file(script_path),
        },
    }
    _write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_path),
                "diagnostic_strict_pass": strict_pass,
                "diagnostic_relaxed_pass": diagnostic_pass,
                "acceptance_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

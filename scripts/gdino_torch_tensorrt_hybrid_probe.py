#!/usr/bin/env python
"""Fail-closed Grounding DINO Torch-TensorRT hybrid execution probe.

The probe deliberately uses frozen tensors and an explicit FP32 reference.  A
legacy ``logits``/``pred_boxes`` archive and the ``final_logits``/
``final_pred_boxes`` aliases from an audited no-TF32 top-k capture are both
supported.  It verifies this container's eager FP32 result first, then an
export-compatible eager rewrite, ``torch.export``, and finally a Torch-TensorRT
hybrid graph.  FP16 compilation is attempted only after the hybrid FP32 graph
passes the selected correctness gate.

This script is an evaluation tool, not a runtime switch.  A successful engine
build is insufficient: the dry-run partition report must prove that at least
one operator is placed in a TensorRT engine, and the compiled graph must expose
TensorRT engine evidence.  Any all-PyTorch fallback, unparseable coverage,
unsafe output, or selected-gate mismatch produces a NO_GO verdict and a
non-zero exit status.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import platform
import re
import resource
import statistics
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


INPUT_NAMES = (
    "pixel_values",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "pixel_mask",
)
OUTPUT_NAMES = ("logits", "pred_boxes")
BASELINE_OUTPUT_KEY_ALIASES = {
    "logits": ("logits", "final_logits"),
    "pred_boxes": ("pred_boxes", "final_pred_boxes"),
}
DECISION_SET_GATE_NAME = "strict_permutation_invariant_label_partitioned_set"
DEFAULT_MATCH_TRANSITION_BUDGET = 5_000_000


class ProbeAbort(RuntimeError):
    """A controlled, evidence-bearing fail-closed stop."""

    def __init__(self, verdict: str, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.verdict = verdict
        self.evidence = evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(value)
            temporary = Path(handle.name)
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _load_baseline_reference(
    baseline_outputs_path: Path,
    *,
    inputs_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Load legacy outputs or prove an adjacent no-TF32 stage capture."""

    key_mapping: dict[str, str] = {}
    with np.load(baseline_outputs_path, allow_pickle=False) as baseline_file:
        for canonical_name, aliases in BASELINE_OUTPUT_KEY_ALIASES.items():
            selected = next((name for name in aliases if name in baseline_file), None)
            if selected is None:
                raise KeyError(
                    f"baseline output {canonical_name} is missing; accepted keys={aliases}"
                )
            key_mapping[canonical_name] = selected
        selected_keys = set(key_mapping.values())
        legacy_keys = set(OUTPUT_NAMES)
        capture_keys = {"final_logits", "final_pred_boxes"}
        if selected_keys not in (legacy_keys, capture_keys):
            raise RuntimeError(
                f"mixed legacy/capture baseline key mapping is forbidden: {key_mapping}"
            )
        reference = {
            canonical_name: np.array(baseline_file[selected_name], copy=True)
            for canonical_name, selected_name in key_mapping.items()
        }

    provenance: dict[str, Any] = {
        "format": "legacy_logits_pred_boxes"
        if set(key_mapping.values()) == set(OUTPUT_NAMES)
        else "audited_topk_stage_boundaries",
        "key_mapping": key_mapping,
        "verified_no_tf32_capture": False,
    }
    if provenance["format"] == "legacy_logits_pred_boxes":
        provenance["precision_boundary"] = (
            "Legacy archives do not prove their TF32 policy; raw mode preserves the "
            "previous behavior, but decision-set mode requires an audited capture."
        )
        return reference, key_mapping, provenance

    summary_path = baseline_outputs_path.with_name("summary.json")
    if not summary_path.is_file():
        raise RuntimeError(
            "stage-boundaries aliases require an adjacent audited summary.json"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("probe") != "gdino_topk_stage_probe":
        raise RuntimeError(
            "stage-boundaries aliases require probe='gdino_topk_stage_probe'"
        )
    environment = summary.get("environment", {})
    if environment.get("matmul_allow_tf32") is not False or environment.get(
        "cudnn_allow_tf32"
    ) is not False:
        raise RuntimeError(
            "stage-boundaries reference is not explicitly audited with both TF32 "
            "policies disabled"
        )
    actual_npz_sha256 = _sha256(baseline_outputs_path)
    if summary.get("artifacts", {}).get("npz_sha256") != actual_npz_sha256:
        raise RuntimeError("stage-boundaries.npz hash does not match summary.json")
    actual_inputs_sha256 = _sha256(inputs_path)
    if summary.get("inputs", {}).get("file_sha256") != actual_inputs_sha256:
        raise RuntimeError("stage capture frozen-input hash mismatch")
    actual_manifest_sha256 = _sha256(manifest_path)
    if (
        summary.get("inputs", {}).get("baseline_manifest", {}).get("sha256")
        != actual_manifest_sha256
    ):
        raise RuntimeError("stage capture baseline-manifest hash mismatch")
    recorded_arrays = summary.get("artifacts", {}).get("npz_arrays", {})
    for canonical_name, selected_name in key_mapping.items():
        descriptor = recorded_arrays.get(selected_name)
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"stage capture descriptor missing for {selected_name}")
        value = reference[canonical_name]
        if (
            descriptor.get("shape") != list(value.shape)
            or descriptor.get("dtype") != str(value.dtype)
            or descriptor.get("numel") != int(value.size)
            or descriptor.get("sha256") != _sha256_array(value)
        ):
            raise RuntimeError(f"stage capture descriptor mismatch for {selected_name}")
    provenance.update(
        {
            "verified_no_tf32_capture": True,
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "capture_probe": summary.get("probe"),
            "capture_precision": summary.get("precision"),
            "precision_policy": {
                "matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            },
            "validated_hashes": {
                "stage_boundaries_npz_sha256": actual_npz_sha256,
                "frozen_inputs_sha256": actual_inputs_sha256,
                "baseline_manifest_sha256": actual_manifest_sha256,
            },
        }
    )
    return reference, key_mapping, provenance


def _git_commit(project: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit_file = project / "COMMIT"
        return commit_file.read_text().strip() if commit_file.exists() else "unknown"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _arrays(outputs: Any) -> dict[str, np.ndarray]:
    """Normalize HF, tuple, or mapping outputs without hiding output dtypes."""
    if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
        tensors = (outputs.logits, outputs.pred_boxes)
    elif isinstance(outputs, dict):
        tensors = (outputs["logits"], outputs["pred_boxes"])
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        tensors = (outputs[0], outputs[1])
    else:
        raise TypeError(f"unexpected model output type: {type(outputs).__name__}")
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in zip(OUTPUT_NAMES, tensors, strict=True):
        if not getattr(tensor, "is_floating_point", lambda: False)():
            raise TypeError(f"model output {name} is not floating point: {tensor.dtype}")
        arrays[name] = tensor.detach().cpu().numpy()
    return arrays


def _raw_diff(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    supported_dtypes = {np.dtype(np.float16), np.dtype(np.float32)}
    for name in OUTPUT_NAMES:
        left = np.asarray(reference[name])
        right = np.asarray(candidate[name])
        reference_dtype_supported = left.dtype in supported_dtypes
        candidate_dtype_supported = right.dtype in supported_dtypes
        if left.shape != right.shape:
            report[name] = {
                "shape_equal": False,
                "reference_shape": list(left.shape),
                "candidate_shape": list(right.shape),
                "reference_dtype": str(left.dtype),
                "candidate_dtype": str(right.dtype),
                "dtype_equal": bool(left.dtype == right.dtype),
                "reference_dtype_supported": reference_dtype_supported,
                "candidate_dtype_supported": candidate_dtype_supported,
                "comparison_safety_pass": False,
                "strict_equivalent": False,
            }
            continue
        dtype_equal = bool(left.dtype == right.dtype)
        jointly_finite = np.isfinite(left) & np.isfinite(right)
        reference_all_finite = bool(np.isfinite(left).all())
        candidate_all_finite = bool(np.isfinite(right).all())
        nan_free = bool(not np.isnan(left).any() and not np.isnan(right).any())
        posinf_free = bool(not np.isposinf(left).any() and not np.isposinf(right).any())
        neginf_free = bool(not np.isneginf(left).any() and not np.isneginf(right).any())
        finite_delta = np.abs(left[jointly_finite] - right[jointly_finite])
        nonfinite_pattern_equal = bool(
            np.array_equal(np.isnan(left), np.isnan(right))
            and np.array_equal(np.isposinf(left), np.isposinf(right))
            and np.array_equal(np.isneginf(left), np.isneginf(right))
        )
        neginf_pattern_equal = bool(
            np.array_equal(np.isneginf(left), np.isneginf(right))
        )
        allclose = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
        comparison_safety_pass = bool(
            reference_dtype_supported
            and candidate_dtype_supported
            and nan_free
            and posinf_free
            and neginf_pattern_equal
        )
        report[name] = {
            "shape_equal": True,
            "reference_dtype": str(left.dtype),
            "candidate_dtype": str(right.dtype),
            "dtype_equal": dtype_equal,
            "reference_dtype_supported": reference_dtype_supported,
            "candidate_dtype_supported": candidate_dtype_supported,
            "bit_exact": bool(np.array_equal(left, right, equal_nan=True)),
            "reference_all_finite": reference_all_finite,
            "candidate_all_finite": candidate_all_finite,
            "nan_free": nan_free,
            "posinf_free": posinf_free,
            "neginf_free": neginf_free,
            "nonfinite_pattern_equal": nonfinite_pattern_equal,
            "neginf_pattern_equal": neginf_pattern_equal,
            "comparison_safety_pass": comparison_safety_pass,
            "jointly_finite_count": int(jointly_finite.sum()),
            "max_abs_on_jointly_finite": (
                float(finite_delta.max()) if finite_delta.size else None
            ),
            "mean_abs_on_jointly_finite": (
                float(finite_delta.mean()) if finite_delta.size else None
            ),
            "allclose": allclose,
            "rtol": rtol,
            "atol": atol,
            "strict_equivalent": bool(
                comparison_safety_pass
                and dtype_equal
                and nonfinite_pattern_equal
                and allclose
            ),
        }
    report["comparison_safety_pass"] = all(
        bool(report[name].get("comparison_safety_pass", False))
        for name in OUTPUT_NAMES
    )
    report["strict_equivalent"] = all(
        bool(report[name].get("strict_equivalent", False)) for name in OUTPUT_NAMES
    )
    return report


def _postprocess(
    processor,
    torch,
    outputs: dict[str, np.ndarray],
    input_ids,
    target_sizes,
    threshold: float,
    text_threshold: float,
) -> list[list[dict[str, Any]]]:
    class Outputs:
        pass

    wrapped = Outputs()
    # The raw safety gate has already proven supported dtypes, no NaN/+Inf, and
    # an identical legal -Inf text-mask pattern.  Cast only this CPU
    # post-processing view to FP32 so thresholding support does not depend on
    # the candidate output dtype.
    wrapped.logits = torch.from_numpy(outputs["logits"]).float()
    wrapped.pred_boxes = torch.from_numpy(outputs["pred_boxes"]).float()
    processed = processor.post_process_grounded_object_detection(
        wrapped,
        input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )
    signatures: list[list[dict[str, Any]]] = []
    for result in processed:
        labels = result.get("text_labels", result.get("labels", []))
        if not (len(labels) == len(result["scores"]) == len(result["boxes"])):
            raise RuntimeError("post-processed label/score/box lengths differ")
        frame: list[dict[str, Any]] = []
        for index, (label, score, box) in enumerate(
            zip(labels, result["scores"], result["boxes"], strict=True)
        ):
            score_value = float(score)
            box_values = [float(value) for value in box]
            if not math.isfinite(score_value) or not all(
                math.isfinite(value) for value in box_values
            ):
                raise RuntimeError(
                    f"post-processed detection {len(signatures)}/{index} is non-finite"
                )
            frame.append(
                {
                    "index": index,
                    "label": str(label),
                    "score": score_value,
                    "box_xyxy_px": box_values,
                }
            )
        signatures.append(frame)
    return signatures


def _ordered_decision_diff(
    reference: list[list[dict[str, Any]]],
    candidate: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    structure_equal = len(reference) == len(candidate)
    max_score_delta = 0.0
    max_box_delta = 0.0
    for left_frame, right_frame in zip(reference, candidate):
        structure_equal = structure_equal and len(left_frame) == len(right_frame)
        for left_item, right_item in zip(left_frame, right_frame):
            structure_equal = structure_equal and left_item["label"] == right_item["label"]
            max_score_delta = max(
                max_score_delta,
                abs(left_item["score"] - right_item["score"]),
            )
            max_box_delta = max(
                max_box_delta,
                *(
                    abs(a - b)
                    for a, b in zip(
                        left_item["box_xyxy_px"],
                        right_item["box_xyxy_px"],
                        strict=True,
                    )
                ),
            )
    strict = bool(
        structure_equal and max_score_delta <= 1e-3 and max_box_delta <= 0.5
    )
    return {
        "structure_equal": bool(structure_equal),
        "max_score_delta": max_score_delta,
        "max_box_delta_px": max_box_delta,
        "strict_decision_equivalent_at_1e-3_and_half_px": strict,
        "reference_detection_counts": [len(items) for items in reference],
        "candidate_detection_counts": [len(items) for items in candidate],
    }


def _decision_set_diff(
    reference: Sequence[Sequence[Mapping[str, Any]]],
    candidate: Sequence[Sequence[Mapping[str, Any]]],
    *,
    transition_budget: int,
) -> dict[str, Any]:
    """Exact label-partitioned, permutation-invariant maximum-total-IoU match."""

    try:
        from gdino_capture_decision_compare import _match_image_decisions
    except ModuleNotFoundError:  # Supports ``python -m scripts...`` in helper tests.
        from scripts.gdino_capture_decision_compare import _match_image_decisions

    if len(reference) != len(candidate):
        return {
            "diagnostic_only": True,
            "acceptance_claim": False,
            "gate_name": DECISION_SET_GATE_NAME,
            "method": "exact_label_partitioned_maximum_total_iou_bitmask_dp",
            "strict_pass": False,
            "failure": "postprocess_batch_count_mismatch",
            "reference_batch_count": len(reference),
            "candidate_batch_count": len(candidate),
        }
    remaining = transition_budget
    rows: list[dict[str, Any]] = []
    for batch_index, (reference_frame, candidate_frame) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        try:
            row = _match_image_decisions(
                reference_frame,
                candidate_frame,
                transition_budget=remaining,
            )
        except RuntimeError as exc:
            return {
                "diagnostic_only": True,
                "acceptance_claim": False,
                "gate_name": DECISION_SET_GATE_NAME,
                "method": "exact_label_partitioned_maximum_total_iou_bitmask_dp",
                "strict_pass": False,
                "failure": "exact_matching_runtime_guard",
                "failure_batch_index": batch_index,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "completed_batches": rows,
                "transition_budget": {
                    "configured": transition_budget,
                    "remaining": remaining,
                },
            }
        remaining -= int(row["matching"]["estimated_transition_upper_bound"])
        rows.append({"batch_index": batch_index, **row})
    strict_pass = all(bool(row["gates"]["strict"]["pass"]) for row in rows)
    return {
        "diagnostic_only": True,
        "acceptance_claim": False,
        "gate_name": DECISION_SET_GATE_NAME,
        "method": "exact_label_partitioned_maximum_total_iou_bitmask_dp",
        "constraint": "exact_text_label_equality",
        "objective": "maximum_total_iou_with_complete_smaller_side_matching",
        "strict_pass": strict_pass,
        "criteria": {
            "complete_one_to_one_label_match": True,
            "iou_min_inclusive": 0.999,
            "score_abs_delta_max_inclusive": 1e-3,
            "box_abs_delta_px_max_inclusive": 0.5,
        },
        "transition_budget": {
            "configured": transition_budget,
            "conservative_estimate_consumed": transition_budget - remaining,
            "remaining": remaining,
        },
        "per_batch": rows,
        "acceptance_boundary": (
            "This decision-set gate is diagnostic evidence only and does not confer "
            "a PASS on the independent frozen SF1 acceptance scorer."
        ),
    }


def _selected_correctness_gate(
    *,
    selected: str,
    raw_diff: Mapping[str, Any],
    decision_set_diff: Mapping[str, Any],
    ordered_decision_diff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    safety_pass = bool(raw_diff.get("comparison_safety_pass", False))
    if selected == "raw":
        raw_pass = bool(raw_diff.get("strict_equivalent", False))
        ordered_pass = bool(
            ordered_decision_diff is None
            or ordered_decision_diff.get(
                "strict_decision_equivalent_at_1e-3_and_half_px", False
            )
        )
        gate_pass = safety_pass and raw_pass and ordered_pass
        criteria = (
            "shape + supported dtype + NaN/+Inf rejection + matching -Inf mask + "
            "dtype-equal raw allclose"
        )
        if ordered_decision_diff is not None:
            criteria += " + legacy ordered postprocess decision tolerance"
    elif selected == "decision-set":
        gate_pass = safety_pass and bool(decision_set_diff.get("strict_pass", False))
        criteria = (
            "raw shape/supported-dtype/nonfinite safety + diagnostic exact "
            "label-partitioned permutation-invariant decision-set strict gate"
        )
    else:  # argparse prevents this; retain fail-closed behavior for direct calls.
        raise ValueError(f"unknown correctness gate: {selected}")
    return {
        "selected": selected,
        "pass": bool(gate_pass),
        "raw_comparison_safety_pass": safety_pass,
        "criteria": criteria,
        "decision_set_is_diagnostic_only": selected == "decision-set",
        "acceptance_claim": False,
    }


def _benchmark(torch, module, inputs, *, warmup: int, runs: int) -> tuple[dict, dict]:
    with torch.inference_mode():
        for _ in range(warmup):
            module(*inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings_ms: list[float] = []
        outputs = None
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = module(*inputs)
            end.record()
            end.synchronize()
            timings_ms.append(float(start.elapsed_time(end)))
    if outputs is None:
        raise RuntimeError("benchmark produced no output")
    metrics = {
        "warmup": warmup,
        "runs": runs,
        "mean_ms": statistics.fmean(timings_ms),
        "p50_ms": _percentile(timings_ms, 0.50),
        "p95_ms": _percentile(timings_ms, 0.95),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "samples_ms": timings_ms,
    }
    return metrics, _arrays(outputs)


def _parse_partition_report(report_text: str) -> dict[str, Any]:
    coverage = re.search(
        r"graph consists of\s+(\d+)\s+Total Operators,\s+of which\s+"
        r"(\d+)\s+operators are supported,\s+([0-9.]+)%\s+coverage",
        report_text,
        flags=re.IGNORECASE,
    )
    engine_ids = sorted(
        {int(value) for value in re.findall(r"TRT Engine #(\d+)", report_text)}
    )
    engine_operator_counts = [
        int(value)
        for value in re.findall(
            r"Number of Operators in Engine:\s*(\d+)", report_text
        )
    ]
    engine_submodules = sorted(
        set(re.findall(r"Submodule name:\s*([A-Za-z0-9_.-]+)", report_text))
    )
    report_complete = bool(
        re.search(r"Graph Structure:", report_text, flags=re.IGNORECASE)
        and re.search(
            r"Aggregate Stats|Recommendations", report_text, flags=re.IGNORECASE
        )
    )
    parsed = bool(
        coverage
        and report_complete
        and len(engine_operator_counts) == len(engine_ids)
    )
    result: dict[str, Any] = {
        "parsed": parsed,
        "report_complete": report_complete,
        "trt_engine_ids": engine_ids,
        "trt_engine_count": len(engine_ids),
        "trt_engine_submodules": engine_submodules,
        "operators_per_trt_engine": engine_operator_counts,
    }
    if not coverage or not report_complete:
        result["parse_error"] = (
            "coverage summary not found"
            if not coverage
            else "dryrun graph structure or aggregate footer not found"
        )
        return result
    total = int(coverage.group(1))
    supported = int(coverage.group(2))
    partitioned = sum(engine_operator_counts)
    result.update(
        {
            "total_operator_count": total,
            "converter_supported_operator_count": supported,
            "converter_supported_percent": float(coverage.group(3)),
            "trt_partitioned_operator_count": partitioned,
            "torch_partitioned_operator_count": total - partitioned,
            "trt_partitioned_percent": (
                100.0 * partitioned / total if total else 0.0
            ),
            "mixed_execution": bool(partitioned and partitioned < total),
            "all_pytorch_fallback": partitioned == 0,
        }
    )
    if partitioned > total:
        result["parsed"] = False
        result["parse_error"] = "TRT engine operator sum exceeds total operators"
    elif engine_ids and len(engine_operator_counts) != len(engine_ids):
        result["parse_error"] = "could not map every TRT engine to an operator count"
    return result


def _compiled_graph_evidence(compiled) -> tuple[dict[str, Any], str]:
    graph_text = str(compiled.graph)
    node_rows = []
    trt_node_names: list[str] = []
    for node in compiled.graph.nodes:
        target = str(node.target)
        row = {"name": node.name, "op": node.op, "target": target}
        node_rows.append(row)
        lowered = f"{node.name} {target}".lower()
        if (
            "tensorrt" in lowered
            or "execute_engine" in lowered
            or re.search(r"_run_on_acc_\d+", lowered)
        ):
            trt_node_names.append(node.name)
    trt_submodules = []
    for name, module in compiled.named_modules():
        identity = f"{name} {type(module).__module__}.{type(module).__name__}".lower()
        if (
            "tensorrt" in identity
            or "execute_engine" in identity
            or re.search(r"_run_on_acc_\d+", identity)
        ):
            trt_submodules.append(name)
    graph_engine_names = sorted(set(re.findall(r"_run_on_acc_\d+", graph_text)))
    evidence_count = max(
        len(graph_engine_names),
        len(set(trt_node_names)),
        len(set(trt_submodules)),
    )
    summary = {
        "fx_node_count": len(node_rows),
        "trt_evidence_count": evidence_count,
        "trt_graph_engine_names": graph_engine_names,
        "trt_node_names": sorted(set(trt_node_names)),
        "trt_submodule_names": sorted(set(trt_submodules)),
        "all_pytorch_fallback": evidence_count == 0,
    }
    return summary, graph_text


def _profile_trt_events(torch, compiled, inputs) -> dict[str, Any]:
    """Collect corroborating runtime evidence without making it the sole gate."""
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode(), torch.profiler.profile(activities=activities) as prof:
            compiled(*inputs)
            torch.cuda.synchronize()
        events = []
        for event in prof.key_averages():
            lowered = event.key.lower()
            if "tensorrt" in lowered or "execute_engine" in lowered:
                events.append(
                    {
                        "key": event.key,
                        "count": int(event.count),
                        "cpu_time_total_us": float(event.cpu_time_total),
                        "device_time_total_us": float(event.device_time_total),
                    }
                )
        return {"available": True, "trt_events": events}
    except Exception as exc:  # profiling is supporting evidence, never a silent pass
        return {
            "available": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }


def _compile_kwargs(torch, inputs, args, *, mode: str, dryrun: str | bool) -> dict:
    kwargs: dict[str, Any] = {
        "arg_inputs": inputs,
        "require_full_compilation": False,
        "min_block_size": args.min_block_size,
        "disable_tf32": True,
        "pass_through_build_failures": False,
        "enable_experimental_decompositions": args.experimental_decompositions,
        "dryrun": dryrun,
        "enable_autocast": mode == "fp16",
    }
    if args.torch_executed_modules:
        kwargs["torch_executed_modules"] = list(args.torch_executed_modules)
    if mode == "fp16":
        kwargs["autocast_low_precision_type"] = torch.float16
    if args.offload_module_to_cpu:
        kwargs["offload_module_to_cpu"] = True
    if args.workspace_size_gib > 0:
        kwargs["workspace_size"] = int(args.workspace_size_gib * (1 << 30))
    return kwargs


def _run_partition_and_compile(
    torch,
    torch_tensorrt,
    exported,
    inputs,
    args,
    *,
    mode: str,
    report_path: Path,
    graph_path: Path,
) -> tuple[Any, dict[str, Any]]:
    if report_path.exists() or graph_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path} or {graph_path}")
    dryrun_started = time.perf_counter()
    torch_tensorrt.dynamo.compile(
        exported,
        **_compile_kwargs(torch, inputs, args, mode=mode, dryrun=str(report_path)),
    )
    dryrun_seconds = time.perf_counter() - dryrun_started
    if not report_path.is_file():
        raise ProbeAbort(
            f"NO_GO_{mode.upper()}_PARTITION_REPORT_MISSING",
            f"Torch-TensorRT dryrun did not create {report_path}",
        )
    report_path.chmod(0o644)
    partition = _parse_partition_report(report_path.read_text(errors="replace"))
    partition.update(
        {
            "report_path": str(report_path),
            "report_sha256": _sha256(report_path),
            "dryrun_seconds": dryrun_seconds,
            "min_block_size": args.min_block_size,
            "requested_torch_executed_modules": list(
                args.torch_executed_modules
            ),
        }
    )
    if not partition["parsed"]:
        raise ProbeAbort(
            f"NO_GO_{mode.upper()}_PARTITION_UNPROVEN",
            f"could not prove Torch-TensorRT {mode} partition coverage",
            evidence={"partition": partition},
        )
    if partition["all_pytorch_fallback"] or partition["trt_engine_count"] < 1:
        raise ProbeAbort(
            f"NO_GO_{mode.upper()}_ALL_PYTORCH_FALLBACK",
            f"Torch-TensorRT {mode} dryrun placed no operators in TensorRT",
            evidence={"partition": partition},
        )

    compile_started = time.perf_counter()
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        **_compile_kwargs(torch, inputs, args, mode=mode, dryrun=False),
    )
    compile_seconds = time.perf_counter() - compile_started
    graph_summary, graph_text = _compiled_graph_evidence(compiled)
    _atomic_write_text(graph_path, graph_text + "\n")
    graph_summary.update(
        {
            "path": str(graph_path),
            "sha256": _sha256(graph_path),
            "dryrun_engine_count": partition["trt_engine_count"],
            "dryrun_engine_count_proven_in_compiled_graph": bool(
                graph_summary["trt_evidence_count"]
                >= partition["trt_engine_count"]
            ),
        }
    )
    if (
        graph_summary["all_pytorch_fallback"]
        or not graph_summary["dryrun_engine_count_proven_in_compiled_graph"]
    ):
        raise ProbeAbort(
            f"NO_GO_{mode.upper()}_COMPILED_GRAPH_TRT_COVERAGE_UNPROVEN",
            f"compiled {mode} graph does not prove every dryrun TensorRT engine",
            evidence={"partition": partition, "compiled_graph": graph_summary},
        )
    return compiled, {
        "partition": partition,
        "compiled_graph": graph_summary,
        "compile_seconds": compile_seconds,
    }


def _write_result(
    output_path: Path,
    result: dict[str, Any],
    saved_outputs: dict[str, np.ndarray],
) -> None:
    if saved_outputs:
        outputs_path = output_path.with_suffix(".outputs.npz")
        if outputs_path.exists():
            raise FileExistsError(f"refusing to overwrite {outputs_path}")
        _atomic_savez(outputs_path, saved_outputs)
        result["saved_outputs"] = {
            "path": str(outputs_path),
            "sha256": _sha256(outputs_path),
            "keys": sorted(saved_outputs),
        }
    _atomic_write_text(
        output_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-outputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--min-block-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--fp32-rtol", type=float, default=1e-6)
    parser.add_argument("--fp32-atol", type=float, default=1e-7)
    parser.add_argument("--fp16-rtol", type=float, default=1e-2)
    parser.add_argument("--fp16-atol", type=float, default=1e-2)
    parser.add_argument(
        "--container-eager-gate",
        choices=("raw", "decision-set"),
        default="raw",
        help=(
            "Correctness gate used for frozen->container eager and each TRT mode. "
            "raw preserves the legacy query-slot gate; decision-set requires an "
            "audited no-TF32 top-k capture and uses exact permutation-invariant "
            "detection matching. Export rewrites always remain raw-gated."
        ),
    )
    parser.add_argument(
        "--max-match-transitions",
        type=int,
        default=DEFAULT_MATCH_TRANSITION_BUDGET,
        help="Fail closed before exact decision matching exceeds this DP budget.",
    )
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--experimental-decompositions", action="store_true")
    parser.add_argument(
        "--torch-executed-module",
        dest="torch_executed_modules",
        action="append",
        default=[],
        help=(
            "Exact exported module FQN to keep in PyTorch. Repeatable. The dry-run "
            "partition report remains the authority on whether the exclusion took effect."
        ),
    )
    parser.add_argument(
        "--offload-module-to-cpu",
        action="store_true",
        help=(
            "Pass offload_module_to_cpu=True to torch_tensorrt.dynamo.compile to "
            "reduce unified-memory pressure while the shared node keeps other "
            "model services resident."
        ),
    )
    parser.add_argument(
        "--workspace-size-gib",
        type=float,
        default=0.0,
        help=(
            "Cap the TensorRT builder workspace (GiB); 0 keeps the compiler "
            "default. Recorded in compile kwargs for provenance."
        ),
    )
    parser.add_argument(
        "--text-outside-export",
        action="store_true",
        help=(
            "Precompute the BERT text backbone eagerly and feed its last hidden "
            "state as an explicit export input, so the text embedding subgraph "
            "(including the aten.add converter failure site) never enters "
            "torch.export. The exported program must prove the text feature is a "
            "real placeholder and that no text-backbone parameters remain."
        ),
    )
    parser.add_argument("--code-commit")
    parser.add_argument(
        "--container-image",
        default="nvcr.io/nvidia/pytorch:26.06-py3",
        help="Provenance label only; the script does not launch Docker itself.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1 or args.min_block_size < 1:
        parser.error("runs/min-block-size must be positive and warmup non-negative")
    if args.max_match_transitions < 1:
        parser.error("max-match-transitions must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be in [0, 1]")
    if not 0.0 <= args.text_threshold <= 1.0:
        parser.error("text-threshold must be in [0, 1]")
    if min(args.fp32_rtol, args.fp32_atol, args.fp16_rtol, args.fp16_atol) < 0:
        parser.error("all numerical tolerances must be non-negative")
    if any(not value.strip() for value in args.torch_executed_modules):
        parser.error("torch-executed-module values must be non-empty")
    if len(set(args.torch_executed_modules)) != len(args.torch_executed_modules):
        parser.error("torch-executed-module values must be unique")

    output_path = Path(args.output)
    output_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_path.parent.chmod(0o755)
    if output_path.exists() or output_path.with_suffix(".outputs.npz").exists():
        parser.error(f"refusing to overwrite output artifact rooted at {output_path}")

    project = Path(__file__).resolve().parent.parent
    result: dict[str, Any] = {
        "schema_version": "1.1",
        "scope": "SF1_L2_TORCH_TENSORRT_HYBRID_FEASIBILITY_ONLY",
        "verdict_scope": "diagnostic_only_does_not_replace_frozen_acceptance",
        "acceptance_claim": False,
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or _git_commit(project),
        "container_image": args.container_image,
        "status": "RUNNING",
        "verdict": None,
        "gates": {
            "all_pytorch_fallback_is_failure": True,
            "selected_correctness_gate": args.container_eager_gate,
            "selected_gate_applies_to": [
                "frozen_reference_vs_container_eager_fp32",
                "same_process_eager_fp32_vs_torch_tensorrt_fp32",
                "same_process_eager_fp32_vs_torch_tensorrt_fp16",
            ],
            "export_rewrite_and_torch_export_always_require_same_process_raw": True,
            "raw_query_slot_report_always_recorded": True,
            "shape_dtype_and_nonfinite_safety_always_required": True,
            "requested_torch_executed_modules": list(args.torch_executed_modules),
            "text_outside_export_requested": bool(args.text_outside_export),
            "fp32_raw_tolerance": {"rtol": args.fp32_rtol, "atol": args.fp32_atol},
            "fp16_raw_tolerance": {"rtol": args.fp16_rtol, "atol": args.fp16_atol},
            "decision_tolerance": {
                "matching": (
                    "exact_label_partitioned_maximum_total_iou_bitmask_dp"
                ),
                "complete_one_to_one_label_match": True,
                "iou_min": 0.999,
                "score_abs": 1e-3,
                "box_abs_px": 0.5,
                "permutation_invariant": True,
                "diagnostic_only": True,
            },
        },
        "modes": {},
    }
    saved_outputs: dict[str, np.ndarray] = {}
    exit_code = 2

    try:
        import tensorrt
        import torch
        import torch_tensorrt
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.models.grounding_dino import modeling_grounding_dino

        if not torch.cuda.is_available():
            raise ProbeAbort("NO_GO_CUDA_UNAVAILABLE", "CUDA is required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result["platform"] = {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_tensorrt": getattr(torch_tensorrt, "__version__", "unknown"),
            "tensorrt": getattr(tensorrt, "__version__", "unknown"),
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "tf32_disabled_in_torch_and_tensorrt": True,
        }
        result["compiler_api"] = {
            "torch_export_export_signature": str(inspect.signature(torch.export.export)),
            "torch_tensorrt_dynamo_compile_signature": str(
                inspect.signature(torch_tensorrt.dynamo.compile)
            ),
            "fp16_route": (
                "enable_autocast=true with autocast_low_precision_type=torch.float16"
            ),
        }

        inputs_path = Path(args.inputs)
        manifest_path = Path(args.baseline_manifest)
        baseline_outputs_path = Path(args.baseline_outputs)
        model_dir = Path(args.model_dir)
        for path in (inputs_path, manifest_path, baseline_outputs_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)

        with np.load(inputs_path) as frozen_file:
            missing = [name for name in INPUT_NAMES if name not in frozen_file]
            if missing:
                raise KeyError(f"frozen inputs are missing: {missing}")
            frozen_inputs = {
                name: np.ascontiguousarray(frozen_file[name]) for name in INPUT_NAMES
            }
        frozen_reference, baseline_key_mapping, baseline_provenance = (
            _load_baseline_reference(
                baseline_outputs_path,
                inputs_path=inputs_path,
                manifest_path=manifest_path,
            )
        )
        if (
            args.container_eager_gate == "decision-set"
            and not baseline_provenance["verified_no_tf32_capture"]
        ):
            raise ProbeAbort(
                "NO_GO_DECISION_SET_REQUIRES_AUDITED_NO_TF32_CAPTURE",
                "decision-set mode requires final_logits/final_pred_boxes from an "
                "audited stage-boundaries capture with both TF32 policies disabled",
                evidence={"baseline_provenance": baseline_provenance},
            )
        manifest = json.loads(manifest_path.read_text())
        if int(manifest["batch_size"]) != int(frozen_inputs["pixel_values"].shape[0]):
            raise ProbeAbort(
                "NO_GO_FROZEN_INPUT_MANIFEST_MISMATCH",
                "manifest batch_size does not match frozen pixel_values",
            )
        result["inputs"] = {
            "path": str(inputs_path),
            "sha256": _sha256(inputs_path),
            "baseline_manifest_path": str(manifest_path),
            "baseline_manifest_sha256": _sha256(manifest_path),
            "baseline_outputs_path": str(baseline_outputs_path),
            "baseline_outputs_sha256": _sha256(baseline_outputs_path),
            "baseline_output_key_mapping": baseline_key_mapping,
            "baseline_reference_provenance": baseline_provenance,
            "batch_size": int(manifest["batch_size"]),
            "shapes": {name: list(value.shape) for name, value in frozen_inputs.items()},
            "dtypes": {name: str(value.dtype) for name, value in frozen_inputs.items()},
            "prompt": manifest.get("prompt"),
            "target_sizes": manifest.get("target_sizes"),
            "postprocess_thresholds": {
                "box": args.threshold,
                "text": args.text_threshold,
            },
        }

        tensors = {
            name: torch.from_numpy(frozen_inputs[name]).to("cuda") for name in INPUT_NAMES
        }
        positional_inputs = tuple(tensors[name] for name in INPUT_NAMES)
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_dir, local_files_only=True
        ).cuda().eval()

        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(
                self,
                pixel_values,
                input_ids,
                token_type_ids,
                attention_mask,
                pixel_mask,
            ):
                outputs = self.wrapped(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    pixel_mask=pixel_mask,
                    return_dict=True,
                )
                return outputs.logits, outputs.pred_boxes

        wrapper = Wrapper(model).cuda().eval()
        eager_metrics, eager_outputs = _benchmark(
            torch,
            wrapper,
            positional_inputs,
            warmup=args.warmup,
            runs=args.runs,
        )
        eager_metrics["throughput_images_per_second"] = int(manifest["batch_size"]) / (
            eager_metrics["mean_ms"] / 1000.0
        )
        eager_metrics["raw_diff_vs_frozen_fp32"] = _raw_diff(
            frozen_reference,
            eager_outputs,
            rtol=args.fp32_rtol,
            atol=args.fp32_atol,
        )
        saved_outputs["eager_fp32_logits"] = eager_outputs["logits"]
        saved_outputs["eager_fp32_pred_boxes"] = eager_outputs["pred_boxes"]
        eager_raw_diff = eager_metrics["raw_diff_vs_frozen_fp32"]
        if not eager_raw_diff["comparison_safety_pass"]:
            eager_metrics["decision_set_diff_vs_frozen_fp32"] = {
                "not_evaluated": True,
                "reason": "raw shape/dtype/nonfinite safety gate failed",
                "diagnostic_only": True,
            }
            eager_metrics["selected_correctness_gate"] = (
                _selected_correctness_gate(
                    selected=args.container_eager_gate,
                    raw_diff=eager_raw_diff,
                    decision_set_diff=eager_metrics[
                        "decision_set_diff_vs_frozen_fp32"
                    ],
                )
            )
            result["modes"]["eager_fp32"] = eager_metrics
            raise ProbeAbort(
                "NO_GO_CONTAINER_EAGER_FP32_UNSAFE_OUTPUT",
                "container eager FP32 failed shape/dtype/nonfinite safety checks",
                evidence={"raw_diff": eager_raw_diff},
            )
        cpu_input_ids = torch.from_numpy(frozen_inputs["input_ids"])
        frozen_reference_decisions = _postprocess(
            processor,
            torch,
            frozen_reference,
            cpu_input_ids,
            manifest["target_sizes"],
            args.threshold,
            args.text_threshold,
        )
        reference_decisions = _postprocess(
            processor,
            torch,
            eager_outputs,
            cpu_input_ids,
            manifest["target_sizes"],
            args.threshold,
            args.text_threshold,
        )
        eager_metrics["ordered_decision_diff_vs_frozen_fp32"] = (
            _ordered_decision_diff(frozen_reference_decisions, reference_decisions)
        )
        eager_metrics["decision_set_diff_vs_frozen_fp32"] = _decision_set_diff(
            frozen_reference_decisions,
            reference_decisions,
            transition_budget=args.max_match_transitions,
        )
        eager_metrics["selected_correctness_gate"] = _selected_correctness_gate(
            selected=args.container_eager_gate,
            raw_diff=eager_raw_diff,
            decision_set_diff=eager_metrics["decision_set_diff_vs_frozen_fp32"],
        )
        result["modes"]["eager_fp32"] = eager_metrics
        if not eager_metrics["selected_correctness_gate"]["pass"]:
            verdict = (
                "NO_GO_CONTAINER_EAGER_FP32_BASELINE_MISMATCH"
                if args.container_eager_gate == "raw"
                else "NO_GO_CONTAINER_EAGER_FP32_DECISION_SET_MISMATCH"
            )
            raise ProbeAbort(
                verdict,
                "container eager FP32 failed the selected frozen-reference gate",
                evidence={
                    "selected_correctness_gate": eager_metrics[
                        "selected_correctness_gate"
                    ],
                    "raw_diff": eager_raw_diff,
                    "decision_set_diff": eager_metrics[
                        "decision_set_diff_vs_frozen_fp32"
                    ],
                },
            )

        original_mask_builder = (
            modeling_grounding_dino.generate_masks_with_special_tokens_and_transfer_map
        )

        def exportable_mask_builder(input_ids):
            batch_size, sequence_length = input_ids.shape
            device = input_ids.device
            special_mask = (
                (input_ids == 101)
                | (input_ids == 102)
                | (input_ids == 1012)
                | (input_ids == 1029)
            )
            positions = torch.arange(sequence_length, device=device)
            query_positions = positions.view(1, sequence_length, 1)
            candidate_positions = positions.view(1, 1, sequence_length)
            special_candidates = special_mask.unsqueeze(1)
            previous_special = torch.where(
                special_candidates & (candidate_positions <= query_positions),
                candidate_positions,
                -1,
            ).amax(dim=2)
            next_special = torch.where(
                special_candidates & (candidate_positions >= query_positions),
                candidate_positions,
                sequence_length,
            ).amin(dim=2)
            valid_block = (
                (next_special != 0)
                & (next_special != sequence_length - 1)
                & (next_special != sequence_length)
            )
            attention_mask = (next_special.unsqueeze(2) == next_special.unsqueeze(1)) & (
                valid_block.unsqueeze(1)
            )
            identity = (query_positions == candidate_positions).expand(
                batch_size, -1, -1
            )
            attention_mask = identity | attention_mask
            position_ids = (
                positions.unsqueeze(0).expand(batch_size, -1) - previous_special - 1
            )
            position_ids = torch.where(
                valid_block, position_ids, torch.zeros_like(position_ids)
            )
            return attention_mask, torch.clamp(position_ids, min=0).to(torch.long)

        def exportable_sinusoidal_position_embedding(
            pos_tensor,
            num_pos_feats=128,
            temperature=10000,
        ):
            scale = torch.tensor(2 * math.pi, dtype=torch.float32, device=pos_tensor.device)
            dim_t = torch.arange(
                num_pos_feats, dtype=torch.float32, device=pos_tensor.device
            )
            dim_t = temperature ** (
                2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
            )
            embeddings = [
                coordinate[..., None] * scale / dim_t
                for coordinate in pos_tensor.unbind(-1)
            ]
            embeddings = [
                torch.stack((item[..., 0::2].sin(), item[..., 1::2].cos()), dim=-1)
                .flatten(-2)
                for item in embeddings
            ]
            if len(embeddings) >= 2:
                embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
            return torch.cat(embeddings, dim=-1).to(pos_tensor.dtype)

        original_masks = original_mask_builder(tensors["input_ids"])
        rewritten_masks = exportable_mask_builder(tensors["input_ids"])
        mask_patch_exact = all(
            torch.equal(left, right)
            for left, right in zip(original_masks, rewritten_masks)
        )
        if not mask_patch_exact:
            raise ProbeAbort(
                "NO_GO_EXPORT_MASK_REWRITE_MISMATCH",
                "exportable special-token mask rewrite is not bit-exact",
            )
        modeling_grounding_dino.generate_masks_with_special_tokens_and_transfer_map = (
            exportable_mask_builder
        )
        modeling_grounding_dino.encode_sinusoidal_position_embedding = (
            exportable_sinusoidal_position_embedding
        )
        scale_modules = [
            module
            for module in model.modules()
            if isinstance(
                module, modeling_grounding_dino.GroundingDinoSinePositionEmbedding
            )
        ]
        for module in scale_modules:
            module.scale = torch.tensor(float(module.scale), device="cuda")
        with torch.inference_mode():
            patched_outputs = _arrays(wrapper(*positional_inputs))
        patch_diff = _raw_diff(
            eager_outputs,
            patched_outputs,
            rtol=args.fp32_rtol,
            atol=args.fp32_atol,
        )
        result["export_compatibility"] = {
            "mask_rewrite_bit_exact": mask_patch_exact,
            "sinusoidal_scale_rewrite": "zero_dim_tensor_v1",
            "scale_module_count": len(scale_modules),
            "raw_diff_vs_native_eager_fp32": patch_diff,
            "site_packages_modified": False,
        }
        saved_outputs["patched_eager_fp32_logits"] = patched_outputs["logits"]
        saved_outputs["patched_eager_fp32_pred_boxes"] = patched_outputs["pred_boxes"]
        if not patch_diff["strict_equivalent"]:
            raise ProbeAbort(
                "NO_GO_EXPORT_COMPATIBILITY_REWRITE_MISMATCH",
                "export compatibility rewrite exceeds the FP32 tolerance",
            )

        export_wrapper = wrapper
        export_inputs = positional_inputs
        original_text_backbone = None
        if args.text_outside_export:
            core = model.model
            captured: dict[str, Any] = {}

            def _capture_text_hook(_module, _hook_args, _hook_kwargs, output):
                captured["output"] = output
                return None

            capture_handle = core.text_backbone.register_forward_hook(
                _capture_text_hook, with_kwargs=True
            )
            try:
                with torch.inference_mode():
                    wrapper(*positional_inputs)
            finally:
                capture_handle.remove()
            if "output" not in captured:
                raise ProbeAbort(
                    "NO_GO_TEXT_OUTSIDE_EXPORT_CAPTURE_FAILED",
                    "text backbone was not invoked during the capture forward",
                )
            raw_text_output = captured["output"]
            captured_hidden = (
                raw_text_output.last_hidden_state
                if hasattr(raw_text_output, "last_hidden_state")
                else raw_text_output[0]
            )
            # Clone outside inference_mode so the tensor is a plain example input.
            text_last_hidden = captured_hidden.detach().clone()

            class _TextBackboneOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state

                def __getitem__(self, index):
                    if index == 0:
                        return self.last_hidden_state
                    raise IndexError(index)

            class _TextBackboneStub(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.injected = None

                def forward(self, *_stub_args, **_stub_kwargs):
                    if self.injected is None:
                        raise RuntimeError(
                            "text features were not injected before text backbone forward"
                        )
                    return _TextBackboneOutput(self.injected)

            original_text_backbone = core.text_backbone
            text_stub = _TextBackboneStub()
            core.text_backbone = text_stub

            class TextOutsideWrapper(torch.nn.Module):
                def __init__(self, wrapped, stub):
                    super().__init__()
                    self.wrapped = wrapped
                    self.stub = stub

                def forward(
                    self,
                    pixel_values,
                    input_ids,
                    token_type_ids,
                    attention_mask,
                    pixel_mask,
                    text_last_hidden_state,
                ):
                    self.stub.injected = text_last_hidden_state
                    outputs = self.wrapped(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        pixel_mask=pixel_mask,
                        return_dict=True,
                    )
                    self.stub.injected = None
                    return outputs.logits, outputs.pred_boxes

            export_wrapper = TextOutsideWrapper(model, text_stub).cuda().eval()
            export_inputs = positional_inputs + (text_last_hidden,)
            with torch.inference_mode():
                stub_outputs = _arrays(export_wrapper(*export_inputs))
            stub_diff = _raw_diff(
                patched_outputs,
                stub_outputs,
                rtol=args.fp32_rtol,
                atol=args.fp32_atol,
            )
            result["export_compatibility"]["text_outside_export"] = {
                "requested": True,
                "capture_via_forward_hook": True,
                "text_last_hidden_state_shape": list(text_last_hidden.shape),
                "text_last_hidden_state_dtype": str(text_last_hidden.dtype),
                "raw_diff_vs_patched_eager_fp32": stub_diff,
            }
            if not stub_diff["strict_equivalent"]:
                raise ProbeAbort(
                    "NO_GO_TEXT_OUTSIDE_EXPORT_REWRITE_MISMATCH",
                    "text-outside-export stub does not reproduce patched eager outputs",
                )

        export_started = time.perf_counter()
        exported = torch.export.export(export_wrapper, export_inputs, strict=False)
        export_seconds = time.perf_counter() - export_started
        exported_graph_path = output_path.with_suffix(".exported_graph.txt")
        if exported_graph_path.exists():
            raise FileExistsError(f"refusing to overwrite {exported_graph_path}")
        _atomic_write_text(exported_graph_path, str(exported.graph_module.graph) + "\n")
        text_outside_proof = None
        if args.text_outside_export:
            user_input_count = len(exported.graph_signature.user_inputs)
            graph_text = exported_graph_path.read_text()
            text_params_remaining = "text_backbone_embeddings" in graph_text
            text_outside_proof = {
                "user_input_count": user_input_count,
                "expected_user_input_count": len(export_inputs),
                "text_backbone_embedding_params_in_graph": text_params_remaining,
            }
            if user_input_count != len(export_inputs) or text_params_remaining:
                raise ProbeAbort(
                    "NO_GO_TEXT_OUTSIDE_EXPORT_NOT_PROVEN",
                    "exported program lost the text-feature placeholder or still "
                    "contains text backbone parameters",
                    evidence=text_outside_proof,
                )
        with torch.inference_mode():
            exported_outputs = _arrays(exported.module()(*export_inputs))
        export_diff = _raw_diff(
            eager_outputs,
            exported_outputs,
            rtol=args.fp32_rtol,
            atol=args.fp32_atol,
        )
        result["torch_export"] = {
            "strict": False,
            "export_seconds": export_seconds,
            "graph_path": str(exported_graph_path),
            "graph_sha256": _sha256(exported_graph_path),
            "raw_diff_vs_native_eager_fp32": export_diff,
        }
        if text_outside_proof is not None:
            result["torch_export"]["text_outside_export_proof"] = text_outside_proof
        saved_outputs["exported_fp32_logits"] = exported_outputs["logits"]
        saved_outputs["exported_fp32_pred_boxes"] = exported_outputs["pred_boxes"]
        if not export_diff["strict_equivalent"]:
            raise ProbeAbort(
                "NO_GO_TORCH_EXPORT_FP32_MISMATCH",
                "torch.export module does not align with native eager FP32",
            )

        fp32_compiled, fp32_record = _run_partition_and_compile(
            torch,
            torch_tensorrt,
            exported,
            export_inputs,
            args,
            mode="fp32",
            report_path=output_path.with_suffix(".fp32.partition.txt"),
            graph_path=output_path.with_suffix(".fp32.compiled_graph.txt"),
        )
        # Release compile-time allocator caches before touching the engines again;
        # the shared node keeps a 51 GiB vLLM service resident and v7 died OOM here.
        gc.collect()
        torch.cuda.empty_cache()
        fp32_record["runtime_profile"] = _profile_trt_events(
            torch, fp32_compiled, export_inputs
        )
        fp32_metrics, fp32_outputs = _benchmark(
            torch,
            fp32_compiled,
            export_inputs,
            warmup=args.warmup,
            runs=args.runs,
        )
        fp32_metrics["throughput_images_per_second"] = int(manifest["batch_size"]) / (
            fp32_metrics["mean_ms"] / 1000.0
        )
        fp32_record.update(fp32_metrics)
        fp32_record["speedup_vs_same_process_eager_fp32_p50"] = (
            eager_metrics["p50_ms"] / fp32_metrics["p50_ms"]
        )
        fp32_record["raw_diff_vs_native_eager_fp32"] = _raw_diff(
            eager_outputs,
            fp32_outputs,
            rtol=args.fp32_rtol,
            atol=args.fp32_atol,
        )
        fp32_raw_diff = fp32_record["raw_diff_vs_native_eager_fp32"]
        if fp32_raw_diff["comparison_safety_pass"]:
            fp32_decisions = _postprocess(
                processor,
                torch,
                fp32_outputs,
                cpu_input_ids,
                manifest["target_sizes"],
                args.threshold,
                args.text_threshold,
            )
            fp32_record["ordered_decision_diff_vs_native_eager_fp32"] = (
                _ordered_decision_diff(reference_decisions, fp32_decisions)
            )
            fp32_record["decision_set_diff_vs_native_eager_fp32"] = (
                _decision_set_diff(
                    reference_decisions,
                    fp32_decisions,
                    transition_budget=args.max_match_transitions,
                )
            )
        else:
            not_evaluated = {
                "not_evaluated": True,
                "reason": "raw shape/dtype/nonfinite safety gate failed",
                "diagnostic_only": True,
                "strict_pass": False,
            }
            fp32_record["ordered_decision_diff_vs_native_eager_fp32"] = (
                not_evaluated
            )
            fp32_record["decision_set_diff_vs_native_eager_fp32"] = not_evaluated
        fp32_record["selected_correctness_gate"] = _selected_correctness_gate(
            selected=args.container_eager_gate,
            raw_diff=fp32_raw_diff,
            decision_set_diff=fp32_record[
                "decision_set_diff_vs_native_eager_fp32"
            ],
            ordered_decision_diff=fp32_record[
                "ordered_decision_diff_vs_native_eager_fp32"
            ],
        )
        fp32_record["gate_pass"] = fp32_record["selected_correctness_gate"]["pass"]
        result["modes"]["torch_tensorrt_hybrid_fp32"] = fp32_record
        saved_outputs["trt_hybrid_fp32_logits"] = fp32_outputs["logits"]
        saved_outputs["trt_hybrid_fp32_pred_boxes"] = fp32_outputs["pred_boxes"]
        if not fp32_record["gate_pass"]:
            raise ProbeAbort(
                "NO_GO_TORCH_TENSORRT_FP32_MISMATCH",
                "Torch-TensorRT hybrid FP32 failed the selected correctness gate",
                evidence={
                    "selected_correctness_gate": fp32_record[
                        "selected_correctness_gate"
                    ],
                    "raw_diff": fp32_raw_diff,
                    "decision_set_diff": fp32_record[
                        "decision_set_diff_vs_native_eager_fp32"
                    ],
                },
            )

        # Re-run eager after engine construction so compilation side effects cannot
        # be mistaken for a candidate speed/correctness result.
        if original_text_backbone is not None:
            model.model.text_backbone = original_text_backbone
        if args.offload_module_to_cpu:
            wrapper.cuda()
        with torch.inference_mode():
            eager_after_compile = _arrays(wrapper(*positional_inputs))
        eager_stability = _raw_diff(
            eager_outputs,
            eager_after_compile,
            rtol=args.fp32_rtol,
            atol=args.fp32_atol,
        )
        result["eager_fp32_stability_after_compile"] = eager_stability
        if not eager_stability["strict_equivalent"]:
            raise ProbeAbort(
                "NO_GO_EAGER_FP32_UNSTABLE_AFTER_COMPILE",
                "native eager FP32 changed after TensorRT compilation",
            )

        if args.skip_fp16:
            result["status"] = "PASS"
            result["verdict"] = "PASS_TORCH_TENSORRT_HYBRID_FP32_FP16_NOT_REQUESTED"
            exit_code = 0
        else:
            fp16_compiled, fp16_record = _run_partition_and_compile(
                torch,
                torch_tensorrt,
                exported,
                export_inputs,
                args,
                mode="fp16",
                report_path=output_path.with_suffix(".fp16.partition.txt"),
                graph_path=output_path.with_suffix(".fp16.compiled_graph.txt"),
            )
            fp16_record["runtime_profile"] = _profile_trt_events(
                torch, fp16_compiled, export_inputs
            )
            fp16_metrics, fp16_outputs = _benchmark(
                torch,
                fp16_compiled,
                export_inputs,
                warmup=args.warmup,
                runs=args.runs,
            )
            fp16_metrics["throughput_images_per_second"] = int(
                manifest["batch_size"]
            ) / (fp16_metrics["mean_ms"] / 1000.0)
            fp16_record.update(fp16_metrics)
            fp16_record["speedup_vs_same_process_eager_fp32_p50"] = (
                eager_metrics["p50_ms"] / fp16_metrics["p50_ms"]
            )
            fp16_record["raw_diff_vs_native_eager_fp32"] = _raw_diff(
                eager_outputs,
                fp16_outputs,
                rtol=args.fp16_rtol,
                atol=args.fp16_atol,
            )
            fp16_raw_diff = fp16_record["raw_diff_vs_native_eager_fp32"]
            if fp16_raw_diff["comparison_safety_pass"]:
                fp16_decisions = _postprocess(
                    processor,
                    torch,
                    fp16_outputs,
                    cpu_input_ids,
                    manifest["target_sizes"],
                    args.threshold,
                    args.text_threshold,
                )
                fp16_record["ordered_decision_diff_vs_native_eager_fp32"] = (
                    _ordered_decision_diff(reference_decisions, fp16_decisions)
                )
                fp16_record["decision_set_diff_vs_native_eager_fp32"] = (
                    _decision_set_diff(
                        reference_decisions,
                        fp16_decisions,
                        transition_budget=args.max_match_transitions,
                    )
                )
            else:
                not_evaluated = {
                    "not_evaluated": True,
                    "reason": "raw shape/dtype/nonfinite safety gate failed",
                    "diagnostic_only": True,
                    "strict_pass": False,
                }
                fp16_record["ordered_decision_diff_vs_native_eager_fp32"] = (
                    not_evaluated
                )
                fp16_record["decision_set_diff_vs_native_eager_fp32"] = (
                    not_evaluated
                )
            fp16_record["selected_correctness_gate"] = _selected_correctness_gate(
                selected=args.container_eager_gate,
                raw_diff=fp16_raw_diff,
                decision_set_diff=fp16_record[
                    "decision_set_diff_vs_native_eager_fp32"
                ],
                ordered_decision_diff=fp16_record[
                    "ordered_decision_diff_vs_native_eager_fp32"
                ],
            )
            fp16_record["gate_pass"] = fp16_record["selected_correctness_gate"][
                "pass"
            ]
            result["modes"]["torch_tensorrt_hybrid_fp16"] = fp16_record
            saved_outputs["trt_hybrid_fp16_logits"] = fp16_outputs["logits"]
            saved_outputs["trt_hybrid_fp16_pred_boxes"] = fp16_outputs["pred_boxes"]
            if not fp16_record["gate_pass"]:
                raise ProbeAbort(
                    "FP32_PASS_FP16_REJECTED_NUMERICAL_OR_DECISION_DRIFT",
                    "hybrid FP32 passed, but hybrid FP16 exceeded a correctness gate",
                    evidence={
                        "selected_correctness_gate": fp16_record[
                            "selected_correctness_gate"
                        ],
                        "raw_diff": fp16_raw_diff,
                        "decision_set_diff": fp16_record[
                            "decision_set_diff_vs_native_eager_fp32"
                        ],
                    },
                )
            result["status"] = "PASS"
            result["verdict"] = "PASS_TORCH_TENSORRT_HYBRID_FP32_AND_FP16"
            exit_code = 0
    except ProbeAbort as exc:
        result["status"] = "NO_GO"
        result["verdict"] = exc.verdict
        result["failure"] = {"exception_type": type(exc).__name__, "message": str(exc)}
        if exc.evidence is not None:
            result["failure"]["evidence"] = exc.evidence
    except Exception as exc:
        result["status"] = "NO_GO"
        result["verdict"] = "NO_GO_UNCONTROLLED_PROBE_FAILURE"
        result["failure"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        result["process_peak_rss_bytes"] = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        )
        result["completed_at_unix"] = int(time.time())
        _write_result(output_path, result, saved_outputs)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

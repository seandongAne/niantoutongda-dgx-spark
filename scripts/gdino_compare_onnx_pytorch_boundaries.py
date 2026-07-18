#!/usr/bin/env python
"""Compare captured Grounding DINO PyTorch and ONNX decision boundaries.

This tool consumes, but never regenerates, two independently captured artifacts:

* a ``gdino_topk_stage_probe.py`` capture directory; and
* a successful ``gdino_onnx_intermediate_probe.py`` ``result.json``.

It verifies the locally available artifact hashes and array descriptors before
comparing proposal scores, proposal coordinate logits, TopK outputs, the first
proposal gather, and final logits/boxes.  Final outputs are reported both in raw
query-rank order and after matching query rows by their originating encoder
proposal ID.

The report is descriptive only.  In particular, a differing boundary does not
identify the operator that caused the difference.  ONNX GridSample sentinels
have no matching tensor in the PyTorch capture schema and are therefore never
used for causal attribution by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"
SCOPE = "SF1-L2_PYTORCH_ONNX_BOUNDARY_COMPARISON_ONLY"
REQUIRED_PT_ARRAYS = {
    "encoder_proposal_class_logits_flat_indices",
    "encoder_proposal_class_logits_sample",
    "encoder_proposal_coord_logits",
    "encoder_proposal_scores",
    "final_logits",
    "final_pred_boxes",
    "topk_indices",
    "topk_selected_coord_logits",
    "topk_values",
}
REQUIRED_ONNX_ROLES = {
    "encoder_class_logits_before_topk_reduce",
    "final_logits",
    "final_pred_boxes",
    "topk_gather_0_data_before_selection",
    "topk_gather_0_output_after_selection",
    "topk_indices",
    "topk_input_scores",
    "topk_values",
}
INPUT_NAMES = (
    "pixel_values",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "pixel_mask",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and compare PyTorch topk-stage and ONNX intermediate "
            "Grounding DINO captures."
        )
    )
    parser.add_argument(
        "--pytorch-capture",
        required=True,
        help="Directory containing summary.json and stage-boundaries.npz.",
    )
    parser.add_argument(
        "--onnx-result",
        required=True,
        help="result.json emitted by gdino_onnx_intermediate_probe.py.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Frozen local model directory used by the PyTorch capture/export.",
    )
    parser.add_argument(
        "--processor-dir",
        help="Local AutoProcessor directory; defaults to --model-dir.",
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Frozen sample_inputs.npz used by both captures.",
    )
    parser.add_argument(
        "--baseline-manifest",
        required=True,
        help="Frozen ONNX export manifest referenced by the PyTorch capture.",
    )
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--max-match-transitions",
        type=int,
        default=5_000_000,
        help="Fail closed before exact label-constrained matching exceeds this budget.",
    )
    parser.add_argument("--output", required=True, help="New comparison JSON path.")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be in [0, 1]")
    if not 0.0 <= args.text_threshold <= 1.0:
        parser.error("text-threshold must be in [0, 1]")
    if args.max_match_transitions < 1:
        parser.error("max-match-transitions must be positive")
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return value


def _json_float(value: Any) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.size),
        "sha256": _sha256_array(value),
    }


def _verify_npz_descriptor(
    name: str, value: np.ndarray, recorded: dict[str, Any]
) -> dict[str, Any]:
    actual = _array_descriptor(value)
    expected = {
        "shape": recorded.get("shape"),
        "dtype": recorded.get("dtype"),
        "numel": recorded.get("numel"),
        "sha256": recorded.get("sha256"),
    }
    if actual != expected:
        raise RuntimeError(
            f"PyTorch NPZ descriptor mismatch for {name}: "
            f"recorded={expected}, actual={actual}"
        )
    return actual


def _candidate_roots(anchor: Path) -> list[Path]:
    roots = [Path.cwd(), anchor.parent]
    for parent in (anchor.parent, *anchor.parents):
        if (parent / ".git").exists():
            roots.append(parent)
            break
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _resolve_declared_file(
    declared: str,
    *,
    anchor: Path,
    fallback_directory: Path | None = None,
) -> Path | None:
    path = Path(declared)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(root / path for root in _candidate_roots(anchor))
        candidates.append(anchor.parent / path)
    if fallback_directory is not None:
        candidates.append(fallback_directory / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _verify_npy_file(
    entry: dict[str, Any], *, result_path: Path, required: bool
) -> tuple[np.ndarray | None, dict[str, Any]]:
    declared = entry.get("file")
    if not isinstance(declared, str) or not declared:
        if required:
            raise ValueError(f"ONNX tensor entry has no file path: {entry.get('roles')}")
        return None, {"available_locally": False, "reason": "missing file descriptor"}
    path = _resolve_declared_file(
        declared,
        anchor=result_path,
        fallback_directory=result_path.parent / "tensors",
    )
    if path is None:
        if required:
            raise FileNotFoundError(f"required ONNX tensor is not local: {declared}")
        return None, {
            "available_locally": False,
            "declared_file": declared,
            "reason": "sentinel file is not available locally",
        }

    recorded_hash = _require_sha256(
        entry.get("file_sha256"), f"ONNX tensor {declared} file_sha256"
    )
    actual_hash = _sha256_file(path)
    if recorded_hash != actual_hash:
        raise RuntimeError(
            f"ONNX tensor hash mismatch for {declared}: "
            f"recorded={recorded_hash}, actual={actual_hash}"
        )
    actual_size = path.stat().st_size
    if int(entry.get("file_size_bytes", -1)) != actual_size:
        raise RuntimeError(
            f"ONNX tensor file-size mismatch for {declared}: "
            f"recorded={entry.get('file_size_bytes')}, actual={actual_size}"
        )
    value = np.load(path, allow_pickle=False)
    if value.dtype.hasobject:
        raise ValueError(f"object array is forbidden: {path}")
    array_record = entry.get("array")
    if not isinstance(array_record, dict):
        raise ValueError(f"ONNX tensor has no array descriptor: {declared}")
    descriptor_checks = {
        "shape": (array_record.get("shape"), list(value.shape)),
        "dtype": (array_record.get("dtype"), str(value.dtype)),
        "nbytes": (array_record.get("nbytes"), int(value.nbytes)),
        "element_count": (array_record.get("element_count"), int(value.size)),
    }
    wrong = {
        key: {"recorded": pair[0], "actual": pair[1]}
        for key, pair in descriptor_checks.items()
        if pair[0] != pair[1]
    }
    if wrong:
        raise RuntimeError(f"ONNX tensor descriptor mismatch for {declared}: {wrong}")
    return value, {
        "available_locally": True,
        "resolved_file": str(path),
        "file_sha256": actual_hash,
        "file_size_bytes": actual_size,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": int(value.nbytes),
        "element_count": int(value.size),
    }


def _finite_array_diff(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_equal": reference.shape == candidate.shape,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "dtype_equal": reference.dtype == candidate.dtype,
    }
    if reference.shape != candidate.shape:
        result["comparison_valid"] = False
        return result
    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    jointly_finite = reference_finite & candidate_finite
    delta = np.abs(
        reference[jointly_finite].astype(np.float64)
        - candidate[jointly_finite].astype(np.float64)
    )
    nonfinite_pattern_equal = bool(
        np.array_equal(np.isnan(reference), np.isnan(candidate))
        and np.array_equal(np.isposinf(reference), np.isposinf(candidate))
        and np.array_equal(np.isneginf(reference), np.isneginf(candidate))
    )
    result.update(
        {
            "comparison_valid": True,
            "bit_exact": bool(np.array_equal(reference, candidate)),
            "nonfinite_pattern_equal": nonfinite_pattern_equal,
            "reference_finite_count": int(reference_finite.sum()),
            "candidate_finite_count": int(candidate_finite.sum()),
            "jointly_finite_count": int(jointly_finite.sum()),
            "max_abs_on_jointly_finite": (
                _json_float(delta.max()) if delta.size else None
            ),
            "mean_abs_on_jointly_finite": (
                _json_float(delta.mean()) if delta.size else None
            ),
        }
    )
    return result


def _require_same_shape(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape:
        raise RuntimeError(f"{label} shapes differ: {left.shape} != {right.shape}")


def _validate_topk_capture(
    arrays: dict[str, np.ndarray], *, label: str
) -> dict[str, Any]:
    scores = arrays["encoder_proposal_scores"]
    indices = arrays["topk_indices"]
    values = arrays["topk_values"]
    coords = arrays["encoder_proposal_coord_logits"]
    gathered = arrays["topk_selected_coord_logits"]
    if scores.ndim != 2 or indices.ndim != 2 or values.shape != indices.shape:
        raise RuntimeError(f"{label} TopK arrays have incompatible ranks/shapes")
    if coords.shape[:2] != scores.shape or coords.ndim != 3:
        raise RuntimeError(f"{label} proposal coordinate shape is incompatible")
    if gathered.shape != (indices.shape[0], indices.shape[1], coords.shape[2]):
        raise RuntimeError(f"{label} gathered coordinate shape is incompatible")
    if not np.issubdtype(indices.dtype, np.integer):
        raise RuntimeError(f"{label} TopK indices are not integers")

    per_batch = []
    for batch_index in range(indices.shape[0]):
        row = indices[batch_index].astype(np.int64, copy=False)
        if row.size != np.unique(row).size:
            raise RuntimeError(f"{label} TopK indices are not unique in batch {batch_index}")
        if row.size and (int(row.min()) < 0 or int(row.max()) >= scores.shape[1]):
            raise RuntimeError(f"{label} TopK index is out of bounds in batch {batch_index}")
        expected_values = scores[batch_index, row]
        expected_gather = coords[batch_index, row]
        if not np.array_equal(expected_values, values[batch_index]):
            raise RuntimeError(f"{label} TopK values do not gather exactly from scores")
        finite_adjacent = np.isfinite(values[batch_index, :-1]) & np.isfinite(
            values[batch_index, 1:]
        )
        if np.any(
            values[batch_index, :-1][finite_adjacent]
            < values[batch_index, 1:][finite_adjacent]
        ):
            raise RuntimeError(f"{label} TopK values are not sorted descending")
        if not np.array_equal(expected_gather, gathered[batch_index]):
            raise RuntimeError(f"{label} gathered coordinates do not match TopK indices")
        selected_mask = np.zeros(scores.shape[1], dtype=bool)
        selected_mask[row] = True
        selected = scores[batch_index, selected_mask]
        rejected = scores[batch_index, ~selected_mask]
        finite_selected = selected[np.isfinite(selected)]
        finite_rejected = rejected[np.isfinite(rejected)]
        min_selected = float(finite_selected.min()) if finite_selected.size else None
        max_rejected = float(finite_rejected.max()) if finite_rejected.size else None
        if (
            min_selected is not None
            and max_rejected is not None
            and min_selected < max_rejected
        ):
            raise RuntimeError(
                f"{label} selected set is not a valid largest-score TopK in batch "
                f"{batch_index}: {min_selected} < {max_rejected}"
            )
        per_batch.append(
            {
                "batch_index": batch_index,
                "proposal_count": int(scores.shape[1]),
                "selected_count": int(row.size),
                "indices_unique": True,
                "values_gather_exact": True,
                "values_sorted_descending": True,
                "coordinate_gather_exact": True,
                "lowest_finite_selected_score": min_selected,
                "highest_finite_rejected_score": max_rejected,
                "selected_rejected_gap": (
                    _json_float(min_selected - max_rejected)
                    if min_selected is not None and max_rejected is not None
                    else None
                ),
            }
        )
    return {"valid": True, "per_batch": per_batch}


def _topk_overlap(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    _require_same_shape(reference, candidate, "TopK index")
    per_batch = []
    for batch_index in range(reference.shape[0]):
        left = reference[batch_index]
        right = candidate[batch_index]
        left_map = {int(proposal): rank for rank, proposal in enumerate(left)}
        right_map = {int(proposal): rank for rank, proposal in enumerate(right)}
        left_set = set(left_map)
        right_set = set(right_map)
        common = left_set & right_set
        union = left_set | right_set
        displacements = np.asarray(
            [abs(left_map[item] - right_map[item]) for item in common], dtype=np.int64
        )
        left_only = sorted(left_set - right_set)
        right_only = sorted(right_set - left_set)
        per_batch.append(
            {
                "batch_index": batch_index,
                "ordered_indices_equal": bool(np.array_equal(left, right)),
                "same_rank_count": int((left == right).sum()),
                "same_rank_fraction": float((left == right).mean()),
                "set_equal": left_set == right_set,
                "set_overlap_count": len(common),
                "set_overlap_fraction_of_k": len(common) / len(left_set),
                "set_jaccard": len(common) / len(union) if union else 1.0,
                "reference_only_count": len(left_only),
                "candidate_only_count": len(right_only),
                "reference_only_first_32": left_only[:32],
                "candidate_only_first_32": right_only[:32],
                "common_proposal_rank_displacement_mean": (
                    float(displacements.mean()) if displacements.size else None
                ),
                "common_proposal_rank_displacement_max": (
                    int(displacements.max()) if displacements.size else None
                ),
            }
        )
    return {
        "ordered_indices_equal_all_batches": all(
            row["ordered_indices_equal"] for row in per_batch
        ),
        "sets_equal_all_batches": all(row["set_equal"] for row in per_batch),
        "per_batch": per_batch,
    }


def _proposal_aligned_diff(
    reference_tensor: np.ndarray,
    candidate_tensor: np.ndarray,
    reference_indices: np.ndarray,
    candidate_indices: np.ndarray,
) -> dict[str, Any]:
    if reference_tensor.ndim < 2 or candidate_tensor.ndim < 2:
        raise RuntimeError("proposal-aligned tensors must have batch and query axes")
    if reference_tensor.shape[0] != candidate_tensor.shape[0]:
        raise RuntimeError("proposal-aligned tensors have different batch sizes")
    if reference_tensor.shape[2:] != candidate_tensor.shape[2:]:
        raise RuntimeError("proposal-aligned tensors have incompatible trailing shapes")
    if reference_tensor.shape[:2] != reference_indices.shape:
        raise RuntimeError("reference tensor query shape does not match TopK indices")
    if candidate_tensor.shape[:2] != candidate_indices.shape:
        raise RuntimeError("candidate tensor query shape does not match TopK indices")

    aligned_reference = []
    aligned_candidate = []
    per_batch = []
    for batch_index in range(reference_indices.shape[0]):
        reference_map = {
            int(proposal): rank
            for rank, proposal in enumerate(reference_indices[batch_index])
        }
        candidate_map = {
            int(proposal): rank
            for rank, proposal in enumerate(candidate_indices[batch_index])
        }
        common = sorted(set(reference_map) & set(candidate_map))
        reference_rows = np.asarray([reference_map[item] for item in common], dtype=np.int64)
        candidate_rows = np.asarray([candidate_map[item] for item in common], dtype=np.int64)
        left = reference_tensor[batch_index, reference_rows]
        right = candidate_tensor[batch_index, candidate_rows]
        aligned_reference.append(left)
        aligned_candidate.append(right)
        per_batch.append(
            {
                "batch_index": batch_index,
                "common_proposal_count": len(common),
                "reference_query_count": int(reference_indices.shape[1]),
                "candidate_query_count": int(candidate_indices.shape[1]),
                "common_proposal_fraction_of_reference": (
                    len(common) / reference_indices.shape[1]
                ),
                "diff": _finite_array_diff(left, right),
            }
        )
    if aligned_reference:
        left_all = np.concatenate(aligned_reference, axis=0)
        right_all = np.concatenate(aligned_candidate, axis=0)
        aggregate = _finite_array_diff(left_all, right_all)
    else:
        aggregate = {"comparison_valid": False, "reason": "no batches"}
    return {
        "alignment_key": "encoder_proposal_id_from_topk_indices",
        "causal_interpretation": "NONE; alignment changes comparison order only",
        "aggregate": aggregate,
        "per_batch": per_batch,
    }


def _write_json_no_overwrite(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        temporary.chmod(0o644)
        # Atomic exclusive creation: unlike replace(), link() cannot overwrite a
        # result that appeared after the initial existence check.
        os.link(temporary, path)
        temporary.unlink()
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def _main(args: argparse.Namespace) -> dict[str, Any]:
    capture_dir = Path(args.pytorch_capture).resolve()
    summary_path = capture_dir / "summary.json"
    npz_path = capture_dir / "stage-boundaries.npz"
    result_path = Path(args.onnx_result).resolve()
    model_dir = Path(args.model_dir).resolve()
    processor_dir = Path(args.processor_dir or args.model_dir).resolve()
    inputs_path = Path(args.inputs).resolve()
    explicit_baseline_path = Path(args.baseline_manifest).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    for path in (
        summary_path,
        npz_path,
        result_path,
        inputs_path,
        explicit_baseline_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (model_dir, processor_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)

    summary = _load_json(summary_path)
    if summary.get("probe") != "gdino_topk_stage_probe":
        raise ValueError(f"unexpected PyTorch probe: {summary.get('probe')!r}")
    if summary.get("precision") != "eager_fp32":
        raise ValueError("PyTorch capture must be eager_fp32")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("PyTorch summary has no artifact manifest")
    recorded_npz_hash = _require_sha256(
        artifacts.get("npz_sha256"), "PyTorch artifacts.npz_sha256"
    )
    actual_npz_hash = _sha256_file(npz_path)
    if recorded_npz_hash != actual_npz_hash:
        raise RuntimeError(
            "PyTorch NPZ hash mismatch: "
            f"recorded={recorded_npz_hash}, actual={actual_npz_hash}"
        )
    if artifacts.get("npz_bytes") != npz_path.stat().st_size:
        raise RuntimeError("PyTorch NPZ byte count does not match summary")
    npz_descriptors = artifacts.get("npz_arrays")
    if not isinstance(npz_descriptors, dict):
        raise ValueError("PyTorch summary has no per-array descriptors")
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(npz_descriptors):
            raise RuntimeError(
                "PyTorch NPZ key set differs from its manifest: "
                f"unrecorded={sorted(set(loaded.files) - set(npz_descriptors))}, "
                f"missing={sorted(set(npz_descriptors) - set(loaded.files))}"
            )
        pt_arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    missing_pt = sorted(REQUIRED_PT_ARRAYS - set(pt_arrays))
    if missing_pt:
        raise KeyError(f"PyTorch capture is missing required arrays: {missing_pt}")
    verified_pt_descriptors = {
        name: _verify_npz_descriptor(name, value, npz_descriptors[name])
        for name, value in pt_arrays.items()
    }

    result = _load_json(result_path)
    if result.get("scope") != "SF1-L2_ONNX_INTERMEDIATE_SEMANTICS_ONLY":
        raise ValueError(f"unexpected ONNX result scope: {result.get('scope')!r}")
    if not result.get("intermediate_outputs_valid"):
        raise RuntimeError("ONNX result did not pass its instrumentation gate")
    if not result.get("instrumentation_equivalence", {}).get("equivalent"):
        raise RuntimeError("ONNX instrumentation is not equivalent to its original graph")
    output_manifest = result.get("outputs")
    if not isinstance(output_manifest, dict) or not isinstance(
        output_manifest.get("tensors"), list
    ):
        raise ValueError("ONNX result has no tensor manifest")
    role_entries: dict[str, dict[str, Any]] = {}
    for entry in output_manifest["tensors"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("roles"), list):
            raise ValueError("invalid ONNX tensor manifest entry")
        for role in entry["roles"]:
            if role in role_entries:
                raise RuntimeError(f"ONNX tensor role is ambiguous: {role}")
            role_entries[role] = entry
    missing_roles = sorted(REQUIRED_ONNX_ROLES - set(role_entries))
    if missing_roles:
        raise KeyError(f"ONNX result is missing required roles: {missing_roles}")

    onnx_arrays: dict[str, np.ndarray] = {}
    verified_onnx_descriptors: dict[str, dict[str, Any]] = {}
    for role in sorted(REQUIRED_ONNX_ROLES):
        value, descriptor = _verify_npy_file(
            role_entries[role], result_path=result_path, required=True
        )
        if value is None:
            raise AssertionError(f"required ONNX role unexpectedly resolved to None: {role}")
        onnx_arrays[role] = value
        verified_onnx_descriptors[role] = descriptor

    inspection_declared = result.get("inspection")
    if not isinstance(inspection_declared, str):
        raise ValueError("ONNX result has no inspection path")
    inspection_path = _resolve_declared_file(
        inspection_declared,
        anchor=result_path,
        fallback_directory=result_path.parent,
    )
    if inspection_path is None:
        raise FileNotFoundError(f"ONNX inspection is not local: {inspection_declared}")
    inspection = _load_json(inspection_path)
    if inspection.get("verdict") != "GO":
        raise RuntimeError(f"ONNX inspection verdict is not GO: {inspection.get('verdict')}")

    pt_input_hash = _require_sha256(
        summary.get("inputs", {}).get("file_sha256"), "PyTorch frozen input hash"
    )
    explicit_input_hash = _sha256_file(inputs_path)
    if pt_input_hash != explicit_input_hash:
        raise RuntimeError(
            "explicit frozen input hash differs from the PyTorch capture: "
            f"{explicit_input_hash} != {pt_input_hash}"
        )
    onnx_input_hash = _require_sha256(
        inspection.get("inputs", {}).get("sha256"), "ONNX frozen input hash"
    )
    if pt_input_hash != onnx_input_hash:
        raise RuntimeError(
            f"frozen input hashes differ: PyTorch={pt_input_hash}, ONNX={onnx_input_hash}"
        )

    original_onnx_hash = _require_sha256(
        result.get("onnx", {}).get("original_sha256"), "ONNX original graph hash"
    )
    inspection_onnx_hash = _require_sha256(
        inspection.get("model", {}).get("sha256"), "inspection graph hash"
    )
    if original_onnx_hash != inspection_onnx_hash:
        raise RuntimeError("ONNX result and inspection graph hashes differ")

    baseline = summary.get("inputs", {}).get("baseline_manifest")
    if not isinstance(baseline, dict) or not isinstance(baseline.get("path"), str):
        raise ValueError("PyTorch summary has no baseline export manifest descriptor")
    baseline_recorded_hash = _require_sha256(
        baseline.get("sha256"), "PyTorch baseline manifest hash"
    )
    baseline_actual_hash = _sha256_file(explicit_baseline_path)
    if baseline_recorded_hash != baseline_actual_hash:
        raise RuntimeError("baseline export manifest hash differs from PyTorch summary")
    export_manifest = _load_json(explicit_baseline_path)
    export_onnx_hash = _require_sha256(
        export_manifest.get("onnx", {}).get("sha256"), "export manifest ONNX hash"
    )
    if export_onnx_hash != original_onnx_hash:
        raise RuntimeError("export manifest and ONNX probe graph hashes differ")
    pt_model_path = summary.get("model", {}).get("directory")
    export_model_path = export_manifest.get("model_dir")
    if pt_model_path != export_model_path:
        raise RuntimeError(
            f"model directory identity differs: {pt_model_path!r} != {export_model_path!r}"
        )
    pt_model_hash = _require_sha256(
        summary.get("model", {}).get("artifact_hashes", {}).get("combined_sha256"),
        "PyTorch model artifact hash",
    )

    # Reuse the already-audited processor and exact label-constrained IoU matcher
    # rather than introducing a second decision-comparison implementation.
    try:
        from gdino_capture_decision_compare import (
            _match_image_decisions,
            _model_artifact_hashes,
            _postprocess,
            _validate_target_sizes,
        )
    except ImportError as error:
        raise RuntimeError(
            "gdino_capture_decision_compare.py must be deployed beside this script"
        ) from error

    explicit_model_artifacts = _model_artifact_hashes(model_dir)
    if explicit_model_artifacts["combined_sha256"] != pt_model_hash:
        raise RuntimeError(
            "explicit model-dir artifact hash differs from the PyTorch capture: "
            f"{explicit_model_artifacts['combined_sha256']} != {pt_model_hash}"
        )
    matcher_module = sys.modules[_match_image_decisions.__module__]
    matcher_path = Path(str(matcher_module.__file__)).resolve()

    with np.load(inputs_path, allow_pickle=False) as frozen:
        missing_inputs = sorted(set(INPUT_NAMES) - set(frozen.files))
        if missing_inputs:
            raise KeyError(f"frozen inputs are missing arrays: {missing_inputs}")
        input_ids = np.array(frozen["input_ids"], copy=True)
        frozen_descriptors = {
            name: _array_descriptor(np.asarray(frozen[name])) for name in INPUT_NAMES
        }
    recorded_input_descriptors = summary.get("inputs", {}).get("arrays")
    if not isinstance(recorded_input_descriptors, dict):
        raise ValueError("PyTorch summary has no frozen input array descriptors")
    for name in INPUT_NAMES:
        if recorded_input_descriptors.get(name) != frozen_descriptors[name]:
            raise RuntimeError(
                f"explicit frozen input descriptor differs for {name}"
            )

    pt_internal = _validate_topk_capture(pt_arrays, label="PyTorch")
    onnx_topk_view = {
        "encoder_proposal_scores": onnx_arrays["topk_input_scores"],
        "encoder_proposal_coord_logits": onnx_arrays[
            "topk_gather_0_data_before_selection"
        ],
        "topk_indices": onnx_arrays["topk_indices"],
        "topk_values": onnx_arrays["topk_values"],
        "topk_selected_coord_logits": onnx_arrays[
            "topk_gather_0_output_after_selection"
        ],
    }
    onnx_internal = _validate_topk_capture(onnx_topk_view, label="ONNX")

    onnx_class_logits = onnx_arrays["encoder_class_logits_before_topk_reduce"]
    sampled_shape = summary.get("tensor_summaries", {}).get(
        "encoder_proposal_class_logits", {}
    ).get("shape")
    if sampled_shape != list(onnx_class_logits.shape):
        raise RuntimeError(
            "PyTorch recorded encoder class-logit shape differs from ONNX full tensor: "
            f"{sampled_shape} != {list(onnx_class_logits.shape)}"
        )
    flat_indices = pt_arrays["encoder_proposal_class_logits_flat_indices"]
    if flat_indices.ndim != 1 or not np.issubdtype(flat_indices.dtype, np.integer):
        raise RuntimeError("PyTorch class-logit sample indices are not a 1-D integer array")
    if flat_indices.size and (
        int(flat_indices.min()) < 0 or int(flat_indices.max()) >= onnx_class_logits.size
    ):
        raise RuntimeError("PyTorch class-logit sample index is out of ONNX tensor bounds")
    onnx_class_sample = onnx_class_logits.reshape(-1)[flat_indices.astype(np.int64)]
    _require_same_shape(
        pt_arrays["encoder_proposal_class_logits_sample"],
        onnx_class_sample,
        "sampled encoder class logits",
    )

    # Prove that the exposed ONNX ReduceMax input and output are internally aligned.
    reduced_scores = np.max(onnx_class_logits, axis=-1)
    if not np.array_equal(reduced_scores, onnx_arrays["topk_input_scores"]):
        raise RuntimeError("ONNX TopK scores do not exactly reduce from exposed class logits")

    pt_indices = pt_arrays["topk_indices"]
    onnx_indices = onnx_arrays["topk_indices"]
    topk_overlap = _topk_overlap(pt_indices, onnx_indices)
    raw_final_logits = _finite_array_diff(
        pt_arrays["final_logits"], onnx_arrays["final_logits"]
    )
    raw_final_boxes = _finite_array_diff(
        pt_arrays["final_pred_boxes"], onnx_arrays["final_pred_boxes"]
    )
    aligned_final_logits = _proposal_aligned_diff(
        pt_arrays["final_logits"],
        onnx_arrays["final_logits"],
        pt_indices,
        onnx_indices,
    )
    aligned_final_boxes = _proposal_aligned_diff(
        pt_arrays["final_pred_boxes"],
        onnx_arrays["final_pred_boxes"],
        pt_indices,
        onnx_indices,
    )

    batch_size = int(pt_arrays["final_logits"].shape[0])
    if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
        raise RuntimeError(
            f"input_ids batch does not match outputs: {input_ids.shape} vs {batch_size}"
        )
    if export_manifest.get("batch_size") != batch_size:
        raise RuntimeError(
            "export manifest batch size differs from captured outputs: "
            f"{export_manifest.get('batch_size')!r} != {batch_size}"
        )
    target_sizes = _validate_target_sizes(
        export_manifest.get("target_sizes"), batch_size
    )

    import torch
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
    pt_decisions = _postprocess(
        processor,
        torch,
        {
            "final_logits": pt_arrays["final_logits"],
            "final_pred_boxes": pt_arrays["final_pred_boxes"],
        },
        input_ids,
        target_sizes,
        args.threshold,
        args.text_threshold,
    )
    onnx_decisions = _postprocess(
        processor,
        torch,
        {
            "final_logits": onnx_arrays["final_logits"],
            "final_pred_boxes": onnx_arrays["final_pred_boxes"],
        },
        input_ids,
        target_sizes,
        args.threshold,
        args.text_threshold,
    )
    if len(pt_decisions) != batch_size or len(onnx_decisions) != batch_size:
        raise RuntimeError("AutoProcessor output batch size differs from captured outputs")
    remaining_match_transitions = args.max_match_transitions
    decision_batches = []
    for batch_index, (pt_batch, onnx_batch) in enumerate(
        zip(pt_decisions, onnx_decisions, strict=True)
    ):
        matched = _match_image_decisions(
            pt_batch,
            onnx_batch,
            transition_budget=remaining_match_transitions,
        )
        remaining_match_transitions -= matched["matching"][
            "estimated_transition_upper_bound"
        ]
        decision_batches.append(
            {
                "batch_index": batch_index,
                "target_size_hw": target_sizes[batch_index],
                "pytorch_detections": pt_batch,
                "onnx_detections": onnx_batch,
                **matched,
            }
        )
    all_pairs = [
        pair for batch in decision_batches for pair in batch["matched_pairs"]
    ]
    aggregate_min_iou = min((pair["iou"] for pair in all_pairs), default=None)
    aggregate_max_score_delta = max(
        (pair["score_abs_delta"] for pair in all_pairs), default=None
    )
    aggregate_max_box_delta = max(
        (pair["box_max_abs_delta_px"] for pair in all_pairs), default=None
    )
    postprocess_comparison = {
        "verdict_scope": "diagnostic_only_does_not_replace_frozen_acceptance",
        "acceptance_claim": False,
        "reference_runtime": "pytorch_capture",
        "candidate_runtime": "onnxruntime_capture",
        "processor": {
            "directory": str(processor_dir),
            "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
            "transformers_version": transformers.__version__,
        },
        "threshold": args.threshold,
        "text_threshold": args.text_threshold,
        "target_sizes_hw": target_sizes,
        "matching": {
            "constraint": "exact_text_label_equality",
            "objective": "exact_maximum_total_iou_within_each_label",
            "implementation": "gdino_capture_decision_compare._match_image_decisions",
            "configured_transition_budget": args.max_match_transitions,
            "remaining_transition_budget": remaining_match_transitions,
        },
        "aggregate": {
            "pytorch_detection_count": sum(len(batch) for batch in pt_decisions),
            "onnx_detection_count": sum(len(batch) for batch in onnx_decisions),
            "matched_detection_count": len(all_pairs),
            "min_iou": aggregate_min_iou,
            "max_score_abs_delta": aggregate_max_score_delta,
            "max_box_abs_delta_px": aggregate_max_box_delta,
            "complete_one_to_one_label_match_all_batches": all(
                not batch["unmatched_reference_indices"]
                and not batch["unmatched_candidate_indices"]
                for batch in decision_batches
            ),
        },
        "per_batch": decision_batches,
    }

    grid_entries = {
        role: entry
        for role, entry in role_entries.items()
        if role.startswith("grid_sample_")
    }
    grid_descriptors = []
    for role, entry in sorted(grid_entries.items()):
        _, descriptor = _verify_npy_file(entry, result_path=result_path, required=False)
        grid_descriptors.append({"role": role, **descriptor})
    grid_all_local = bool(grid_descriptors) and all(
        row["available_locally"] for row in grid_descriptors
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "verdict": "BOUNDARIES_COMPARED_NO_CAUSAL_ATTRIBUTION",
        "created_at_unix": int(time.time()),
        "interpretation_boundary": (
            "This report locates observed boundary differences but does not identify "
            "the operator that caused them."
        ),
        "artifact_verification": {
            "frozen_input": {
                "same_sha256": True,
                "explicit_file": str(inputs_path),
                "sha256": pt_input_hash,
                "all_array_descriptors_verified": True,
                "arrays": frozen_descriptors,
            },
            "pytorch_capture": {
                "summary": str(summary_path),
                "summary_sha256": _sha256_file(summary_path),
                "npz": str(npz_path),
                "npz_sha256": actual_npz_hash,
                "npz_bytes": npz_path.stat().st_size,
                "all_array_descriptors_verified": True,
                "array_count": len(verified_pt_descriptors),
            },
            "onnx_capture": {
                "result": str(result_path),
                "result_sha256": _sha256_file(result_path),
                "inspection": str(inspection_path),
                "inspection_sha256": _sha256_file(inspection_path),
                "original_onnx_sha256": original_onnx_hash,
                "required_tensor_descriptors_verified": True,
                "required_tensor_count": len(verified_onnx_descriptors),
                "instrumentation_equivalence_passed": True,
            },
            "export_chain": {
                "baseline_manifest": str(explicit_baseline_path),
                "baseline_manifest_sha256": baseline_actual_hash,
                "baseline_manifest_hash_matches_pytorch_capture": True,
                "onnx_hash_matches_manifest_inspection_and_result": True,
            },
            "model_identity": {
                "pytorch_model_directory": pt_model_path,
                "export_model_directory": export_model_path,
                "directory_equal": True,
                "explicit_model_directory": str(model_dir),
                "explicit_model_artifact_hash_matches_pytorch_capture": True,
                "pytorch_model_artifact_combined_sha256": pt_model_hash,
                "onnx_export_manifest_records_source_weight_sha256": False,
                "source_weight_identity_cryptographically_proven_across_captures": False,
                "limitation": (
                    "The export manifest records the model directory but no source-weight "
                    "digest; path equality is not a cryptographic weight-identity proof."
                ),
            },
            "decision_matcher": {
                "source": str(matcher_path),
                "source_sha256": _sha256_file(matcher_path),
            },
        },
        "internal_consistency": {
            "pytorch_topk_and_gather": pt_internal,
            "onnx_reduce_max_exact": True,
            "onnx_topk_and_gather": onnx_internal,
        },
        "boundary_comparisons": {
            "encoder_class_logits_sampled_at_pytorch_flat_indices": {
                "sample_count": int(flat_indices.size),
                "pytorch_flat_indices_sha256": _sha256_array(flat_indices),
                "onnx_full_tensor_shape": list(onnx_class_logits.shape),
                "diff": _finite_array_diff(
                    pt_arrays["encoder_proposal_class_logits_sample"],
                    onnx_class_sample,
                ),
            },
            "proposal_scores_full": _finite_array_diff(
                pt_arrays["encoder_proposal_scores"],
                onnx_arrays["topk_input_scores"],
            ),
            "proposal_coord_logits_full": _finite_array_diff(
                pt_arrays["encoder_proposal_coord_logits"],
                onnx_arrays["topk_gather_0_data_before_selection"],
            ),
            "topk": {
                "indices": topk_overlap,
                "values_raw_rank_order": _finite_array_diff(
                    pt_arrays["topk_values"], onnx_arrays["topk_values"]
                ),
            },
            "topk_gather_coord_logits": {
                "raw_rank_order": _finite_array_diff(
                    pt_arrays["topk_selected_coord_logits"],
                    onnx_arrays["topk_gather_0_output_after_selection"],
                ),
                "proposal_id_aligned": _proposal_aligned_diff(
                    pt_arrays["topk_selected_coord_logits"],
                    onnx_arrays["topk_gather_0_output_after_selection"],
                    pt_indices,
                    onnx_indices,
                ),
            },
            "final_logits": {
                "raw_rank_order": raw_final_logits,
                "proposal_id_aligned": aligned_final_logits,
            },
            "final_pred_boxes": {
                "raw_rank_order": raw_final_boxes,
                "proposal_id_aligned": aligned_final_boxes,
            },
        },
        "postprocessed_detection_comparison": postprocess_comparison,
        "grid_sample": {
            "onnx_sentinel_role_count": len(grid_entries),
            "onnx_sentinels_all_available_locally": grid_all_local,
            "onnx_sentinel_descriptors": grid_descriptors,
            "pytorch_counterpart_available": False,
            "compared_across_runtimes": False,
            "not_compared_reason": (
                "The PyTorch capture does not contain tensors corresponding to the ONNX "
                "GridSample sentinels. If sentinel files are not local they are likewise "
                "not compared."
            ),
            "causal_attribution": "NONE",
        },
    }


def main() -> int:
    args = _parse_args()
    try:
        report = _main(args)
        output_path = Path(args.output).resolve()
        _write_json_no_overwrite(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI must fail closed with a concise error.
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

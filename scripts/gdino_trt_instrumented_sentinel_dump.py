#!/usr/bin/env python
"""Build a TensorRT FP32 no-TF32 engine from the INSTRUMENTED Grounding DINO
ONNX graph, dump every sentinel boundary tensor, and compare against the
already-validated ONNX Runtime sentinel references.

Context (DAY-07): ORT passes the strict detection-set gate on the frozen pair,
so the exported graph is decision-equivalent, while the clean TensorRT FP32
no-TF32 engine still loses detections ([5,5]->[2,3]).  Both consume the same
ONNX, which makes an ORT-vs-TRT sentinel bisect well-posed for the first time:
the first boundary where TensorRT drifts beyond the known ORT-vs-host scale
localizes the guilty region without guessing.

Diagnostic only.  Marking sentinel tensors as outputs blocks TensorRT fusions,
so this engine's latency is NOT comparable to the clean engine and no speedup
is reported from it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    from gdino_compare_onnx_pytorch_boundaries import (
        _finite_array_diff,
        _topk_overlap,
    )
    from gdino_trt_runtime_bench import _postprocess, _decision_diff, _torch_dtype
    from gdino_capture_decision_compare import _match_image_decisions
except ModuleNotFoundError:  # Supports ``python -m scripts...`` in helper tests.
    from scripts.gdino_compare_onnx_pytorch_boundaries import (
        _finite_array_diff,
        _topk_overlap,
    )
    from scripts.gdino_trt_runtime_bench import _postprocess, _decision_diff, _torch_dtype
    from scripts.gdino_capture_decision_compare import _match_image_decisions

# Classification scale: ORT CPU vs host no-TF32 measured 7.44e-5 on proposal
# scores and 2.11e-4 on proposal coord logits (onnx-pytorch-boundary
# comparison).  A region is called consistent when TRT stays within ~10x of
# that scale and divergent when it exceeds ~100x; in between is inconclusive.
CONSISTENT_MAX_ABS = {
    "topk_input_scores": 1e-3,
    "encoder_class_logits_before_topk_reduce": 5e-3,
    "topk_gather_0_data_before_selection": 5e-3,
}
DIVERGENT_MAX_ABS = {
    "topk_input_scores": 1e-2,
    "encoder_class_logits_before_topk_reduce": 5e-2,
    "topk_gather_0_data_before_selection": 5e-2,
}
DEFAULT_TRANSITION_BUDGET = 1_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference(entry: dict, reference_root: Path) -> np.ndarray:
    declared = Path(entry["file"])
    candidates = [declared]
    if not declared.is_absolute():
        candidates.append(reference_root / declared)
    resolved = None
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            break
        gz = candidate.with_suffix(candidate.suffix + ".gz")
        if gz.exists():
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
                with gzip.open(gz, "rb") as source:
                    shutil.copyfileobj(source, handle)
                resolved = Path(handle.name)
            break
    if resolved is None:
        raise FileNotFoundError(f"sentinel reference missing: {entry['file']}")
    declared_sha = entry.get("sha256")
    if declared_sha and resolved.suffix == ".npy" and resolved.name == declared.name:
        actual = _sha256(resolved)
        if actual != declared_sha:
            raise RuntimeError(
                f"sentinel reference sha256 mismatch for {declared}: "
                f"{actual} != {declared_sha}"
            )
    return np.load(resolved)


def _proposal_aligned_final_diff(
    reference_indices: np.ndarray,
    candidate_indices: np.ndarray,
    reference_final: np.ndarray,
    candidate_final: np.ndarray,
) -> dict:
    per_batch = []
    for batch in range(reference_indices.shape[0]):
        reference_slot = {
            int(proposal): slot
            for slot, proposal in enumerate(reference_indices[batch])
        }
        candidate_slot = {
            int(proposal): slot
            for slot, proposal in enumerate(candidate_indices[batch])
        }
        common = sorted(set(reference_slot) & set(candidate_slot))
        if not common:
            per_batch.append({"batch_index": batch, "common_count": 0})
            continue
        left = reference_final[batch][[reference_slot[p] for p in common]]
        right = candidate_final[batch][[candidate_slot[p] for p in common]]
        jointly_finite = np.isfinite(left) & np.isfinite(right)
        delta = np.abs(
            left[jointly_finite].astype(np.float64)
            - right[jointly_finite].astype(np.float64)
        )
        per_batch.append(
            {
                "batch_index": batch,
                "common_count": len(common),
                "max_abs_on_jointly_finite": float(delta.max()) if delta.size else None,
                "mean_abs_on_jointly_finite": float(delta.mean()) if delta.size else None,
            }
        )
    return {"per_batch": per_batch}


def _build_engine(args, result: dict) -> None:
    engine_path = Path(args.engine)
    if engine_path.exists() and args.reuse_engine:
        result["engine_build"] = {
            "reused_existing_engine": True,
            "trtexec_invoked": False,
        }
        return
    if engine_path.exists():
        raise SystemExit(
            f"refusing to overwrite existing engine without --reuse-engine: {engine_path}"
        )
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = engine_path.with_suffix(".trtexec.log")
    command = [
        args.trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={engine_path}",
        "--noTF32",
        "--skipInference",
        f"--memPoolSize=workspace:{args.workspace_mib}M",
    ]
    started = time.time()
    with log_path.open("w") as handle:
        completed = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    duration = time.time() - started
    tail = log_path.read_text(errors="replace").splitlines()[-40:]
    result["engine_build"] = {
        "reused_existing_engine": False,
        "trtexec_invoked": True,
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": duration,
        "log_path": str(log_path),
        "log_tail": tail if completed.returncode != 0 else tail[-8:],
    }
    if completed.returncode != 0 or not engine_path.exists():
        raise RuntimeError(
            f"trtexec engine build failed (rc={completed.returncode}); see {log_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="instrumented ONNX path")
    parser.add_argument("--sentinel-manifest", required=True, help="onnx-intermediate result.json")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--torch-outputs", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--trtexec", required=True)
    parser.add_argument("--reference-root", default=".")
    parser.add_argument("--workspace-mib", type=int, default=16384)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--transition-budget", type=int, default=DEFAULT_TRANSITION_BUDGET
    )
    parser.add_argument("--reuse-engine", action="store_true")
    parser.add_argument("--code-commit")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {output_path}")
    tensors_dir = output_path.parent / "tensors"

    sentinel_manifest = json.loads(Path(args.sentinel_manifest).read_text())
    manifest_entries = sentinel_manifest["outputs"]["tensors"]
    if not sentinel_manifest.get("instrumentation_equivalence", {}).get("equivalent"):
        raise RuntimeError(
            "sentinel manifest does not attest instrumentation equivalence; "
            "refusing to build diagnostics on an unvalidated instrumented graph"
        )
    name_to_entry = {entry["name"]: entry for entry in manifest_entries}
    role_by_name = {entry["name"]: entry["roles"][0] for entry in manifest_entries}

    onnx_path = Path(args.onnx)
    declared_onnx_sha = sentinel_manifest.get("onnx", {}).get("instrumented_sha256")
    onnx_sha = _sha256(onnx_path)
    if declared_onnx_sha and onnx_sha != declared_onnx_sha:
        raise RuntimeError(
            f"instrumented ONNX sha256 mismatch: {onnx_sha} != {declared_onnx_sha}"
        )

    result: dict = {
        "schema_version": "1.0",
        "scope": "SF1_DIAGNOSTIC_TRT_SENTINEL_LOCALIZATION_NOT_ACCEPTANCE",
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or "unknown",
        "onnx": {"path": str(onnx_path), "sha256": onnx_sha},
        "timing_boundary": (
            "Instrumented engines block TensorRT fusions; latency here is not "
            "comparable to the clean engine and confers no speedup claim."
        ),
    }

    _build_engine(args, result)

    import tensorrt as trt
    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result["platform"] = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "tensorrt": trt.__version__,
    }

    engine_path = Path(args.engine)
    logger = trt.Logger(trt.Logger.WARNING)
    with engine_path.open("rb") as handle:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")
    result["engine"] = {
        "path": str(engine_path),
        "sha256": _sha256(engine_path),
        "size_bytes": engine_path.stat().st_size,
        "device_memory_size_bytes": int(
            engine.device_memory_size_v2
            if hasattr(engine, "device_memory_size_v2")
            else engine.device_memory_size
        ),
    }

    frozen_inputs = np.load(args.inputs)
    tensors = {}
    input_names = []
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
            if name not in frozen_inputs:
                raise KeyError(f"engine input missing from frozen NPZ: {name}")
            dtype = _torch_dtype(trt, torch, engine.get_tensor_dtype(name))
            tensor = torch.as_tensor(
                frozen_inputs[name], dtype=dtype, device="cuda"
            ).contiguous()
            if not context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"rejected shape for {name}: {tuple(tensor.shape)}")
            tensors[name] = tensor
        else:
            output_names.append(name)

    expected_outputs = set(name_to_entry)
    actual_outputs = set(output_names)
    result["io"] = {
        "inputs": input_names,
        "outputs": sorted(actual_outputs),
        "expected_sentinel_outputs": sorted(expected_outputs),
        "output_names_match_manifest": actual_outputs == expected_outputs,
    }
    if actual_outputs != expected_outputs:
        result["io"]["missing"] = sorted(expected_outputs - actual_outputs)
        result["io"]["unexpected"] = sorted(actual_outputs - expected_outputs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(
            "TensorRT engine outputs do not match the sentinel manifest; "
            f"see {output_path}"
        )

    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"TensorRT could not infer shapes for: {unresolved}")
    engine_output_dtypes = {}
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(f"unresolved output shape for {name}: {shape}")
        trt_dtype = engine.get_tensor_dtype(name)
        engine_output_dtypes[name] = str(trt_dtype)
        dtype = _torch_dtype(trt, torch, trt_dtype)
        tensors[name] = torch.empty(shape, dtype=dtype, device="cuda")
    for name, tensor in tensors.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"failed to bind TensorRT tensor: {name}")

    stream = torch.cuda.Stream()
    wall_samples = []
    for _ in range(5):
        started = time.perf_counter()
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        stream.synchronize()
        wall_samples.append((time.perf_counter() - started) * 1000.0)
    result["diagnostic_run"] = {
        "runs": len(wall_samples),
        "wall_ms_samples": wall_samples,
        "not_a_performance_claim": True,
    }

    trt_arrays = {}
    for name in output_names:
        array = tensors[name].cpu().numpy()
        entry = name_to_entry[name]
        if entry["dtype"] == "int64" and array.dtype != np.int64:
            array = array.astype(np.int64)
        trt_arrays[name] = array

    tensors_dir.mkdir(parents=True, exist_ok=True)
    dumped = {}
    for ordinal, entry in enumerate(manifest_entries):
        role = entry["roles"][0]
        array = trt_arrays[entry["name"]]
        file_path = tensors_dir / f"{ordinal:02d}-{role}-trt.npy"
        np.save(file_path, array)
        dumped[role] = {
            "file": str(file_path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "engine_dtype": engine_output_dtypes[entry["name"]],
            "sha256": _sha256(file_path),
        }
    result["trt_sentinels"] = dumped

    references = {
        entry["roles"][0]: _load_reference(entry, Path(args.reference_root))
        for entry in manifest_entries
    }

    comparisons = {}
    for entry in manifest_entries:
        role = entry["roles"][0]
        candidate = trt_arrays[entry["name"]]
        reference = references[role]
        if role == "topk_indices":
            comparisons[role] = {
                "kind": "topk_index_overlap",
                **_topk_overlap(
                    reference.astype(np.int64), candidate.astype(np.int64)
                ),
            }
        else:
            comparisons[role] = {
                "kind": "finite_array_diff",
                **_finite_array_diff(reference, candidate),
                "sentinel_only": role.startswith("grid_sample_"),
            }
    result["ort_vs_trt"] = comparisons

    result["proposal_id_aligned_finals_vs_ort"] = {
        "final_logits": _proposal_aligned_final_diff(
            references["topk_indices"].astype(np.int64),
            trt_arrays[
                next(
                    entry["name"]
                    for entry in manifest_entries
                    if entry["roles"][0] == "topk_indices"
                )
            ].astype(np.int64),
            references["final_logits"],
            trt_arrays[
                next(
                    entry["name"]
                    for entry in manifest_entries
                    if entry["roles"][0] == "final_logits"
                )
            ],
        ),
        "final_pred_boxes": _proposal_aligned_final_diff(
            references["topk_indices"].astype(np.int64),
            trt_arrays[
                next(
                    entry["name"]
                    for entry in manifest_entries
                    if entry["roles"][0] == "topk_indices"
                )
            ].astype(np.int64),
            references["final_pred_boxes"],
            trt_arrays[
                next(
                    entry["name"]
                    for entry in manifest_entries
                    if entry["roles"][0] == "final_pred_boxes"
                )
            ],
        ),
    }

    baseline_manifest = json.loads(Path(args.baseline_manifest).read_text())
    processor = AutoProcessor.from_pretrained(args.model_dir)
    input_ids = torch.from_numpy(np.array(frozen_inputs["input_ids"]))
    target_sizes = baseline_manifest["target_sizes"]

    final_logits_name = next(
        entry["name"] for entry in manifest_entries if entry["roles"][0] == "final_logits"
    )
    final_boxes_name = next(
        entry["name"]
        for entry in manifest_entries
        if entry["roles"][0] == "final_pred_boxes"
    )

    def decisions_from(logits, boxes):
        return _postprocess(
            processor,
            torch.from_numpy(np.asarray(logits, dtype=np.float32)),
            torch.from_numpy(np.asarray(boxes, dtype=np.float32)),
            input_ids,
            target_sizes,
            args.threshold,
            args.text_threshold,
        )

    trt_decisions = decisions_from(
        trt_arrays[final_logits_name], trt_arrays[final_boxes_name]
    )
    ort_decisions = decisions_from(
        references["final_logits"], references["final_pred_boxes"]
    )
    with np.load(args.torch_outputs) as torch_outputs:
        torch_decisions = decisions_from(
            np.array(torch_outputs["logits"]), np.array(torch_outputs["pred_boxes"])
        )

    def to_items(frames):
        return [
            [
                {
                    "label": item["label"],
                    "score": item["score"],
                    "box_xyxy_px": list(item["box"]),
                }
                for item in frame
            ]
            for frame in frames
        ]

    def set_gates(reference_frames, candidate_frames):
        remaining = args.transition_budget
        rows = []
        for batch_index, (reference, candidate) in enumerate(
            zip(reference_frames, candidate_frames)
        ):
            row = _match_image_decisions(
                reference, candidate, transition_budget=remaining
            )
            remaining -= int(row["matching"]["estimated_transition_upper_bound"])
            rows.append(
                {
                    "batch_index": batch_index,
                    "counts": row["counts"],
                    "delta_summary": row["delta_summary"],
                    "gates": row["gates"],
                    "unmatched_reference_indices": row["unmatched_reference_indices"],
                    "unmatched_candidate_indices": row["unmatched_candidate_indices"],
                }
            )
        return {
            "strict_pass": all(row["gates"]["strict"]["pass"] for row in rows),
            "diagnostic_pass": all(row["gates"]["diagnostic"]["pass"] for row in rows),
            "per_batch": rows,
        }

    result["decisions"] = {
        "trt_detection_counts": [len(frame) for frame in trt_decisions],
        "ort_detection_counts": [len(frame) for frame in ort_decisions],
        "torch_baseline_detection_counts": [len(frame) for frame in torch_decisions],
        "trt_vs_ort_positional": _decision_diff(ort_decisions, trt_decisions),
        "trt_vs_ort_set_gates": set_gates(to_items(ort_decisions), to_items(trt_decisions)),
        "trt_vs_torch_baseline_positional": _decision_diff(
            torch_decisions, trt_decisions
        ),
        "trt_vs_torch_baseline_set_gates": set_gates(
            to_items(torch_decisions), to_items(trt_decisions)
        ),
        "trt_detections": trt_decisions,
    }

    def classify() -> str:
        pre_topk_roles = (
            "topk_input_scores",
            "encoder_class_logits_before_topk_reduce",
            "topk_gather_0_data_before_selection",
        )
        divergent = []
        inconclusive = []
        for role in pre_topk_roles:
            max_abs = comparisons[role].get("max_abs_on_jointly_finite")
            if max_abs is None:
                inconclusive.append(role)
            elif max_abs > DIVERGENT_MAX_ABS[role]:
                divergent.append(role)
            elif max_abs > CONSISTENT_MAX_ABS[role]:
                inconclusive.append(role)
        set_pass = result["decisions"]["trt_vs_ort_set_gates"]["diagnostic_pass"]
        if divergent:
            return "DIVERGES_BEFORE_TOPK:" + ",".join(divergent)
        aligned = result["proposal_id_aligned_finals_vs_ort"]["final_pred_boxes"][
            "per_batch"
        ]
        aligned_max = max(
            (row.get("max_abs_on_jointly_finite") or 0.0) for row in aligned
        )
        if not set_pass and not inconclusive:
            return (
                "PRE_TOPK_CONSISTENT_DIVERGES_AFTER_TOPK"
                f"(aligned_final_boxes_max_abs={aligned_max:.6g})"
            )
        if set_pass and not inconclusive:
            return "CONSISTENT_WITH_ORT_WITHIN_DIAGNOSTIC_TOLERANCE"
        return "INCONCLUSIVE:" + ",".join(inconclusive)

    result["classification_thresholds"] = {
        "consistent_max_abs": CONSISTENT_MAX_ABS,
        "divergent_max_abs": DIVERGENT_MAX_ABS,
        "scale_reference": (
            "ORT CPU vs host no-TF32: proposal scores 7.44e-5, coord logits 2.11e-4"
        ),
    }
    result["verdict"] = classify()
    result["acceptance_boundary"] = (
        "Sentinel localization on the frozen two-image workload; grid-sample "
        "captures are sentinels only and do not by themselves prove a causal "
        "operator.  No performance or acceptance claim follows from this run."
    )
    result["exit_code_semantics"] = "0 = diagnostic completed; 2 = failed to complete"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "trt_detection_counts": result["decisions"]["trt_detection_counts"],
                "ort_detection_counts": result["decisions"]["ort_detection_counts"],
                "pre_topk_max_abs": {
                    role: comparisons[role].get("max_abs_on_jointly_finite")
                    for role in (
                        "topk_input_scores",
                        "encoder_class_logits_before_topk_reduce",
                        "topk_gather_0_data_before_selection",
                    )
                },
                "topk_sets_equal": comparisons["topk_indices"][
                    "sets_equal_all_batches"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

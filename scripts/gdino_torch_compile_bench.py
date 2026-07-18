#!/usr/bin/env python
"""Benchmark Grounding DINO eager and ``torch.compile`` FP32/BF16 modes.

The probe deliberately keeps the eager FP32 forward from this process as the
correctness oracle.  Frozen inputs, post-processing, non-finite-pattern checks,
and the strict detection-decision gate are shared with
``gdino_torch_precision_bench.py``.  Lazy compilation and post-compile warmup
are reported separately and are never included in steady-state timings.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    from gdino_torch_precision_bench import (
        INPUT_NAMES,
        _decision_diff,
        _percentile,
        _postprocess,
        _raw_diff,
        _raw_outputs_have_valid_nonfinite_patterns,
        _sha256,
    )
except ModuleNotFoundError as exc:
    if exc.name != "gdino_torch_precision_bench":
        raise
    from scripts.gdino_torch_precision_bench import (
        INPUT_NAMES,
        _decision_diff,
        _percentile,
        _postprocess,
        _raw_diff,
        _raw_outputs_have_valid_nonfinite_patterns,
        _sha256,
    )


OUTPUT_NAMES = ("logits", "pred_boxes")
REQUESTED_CANDIDATES = ("eager_bf16", "compile_fp32", "compile_bf16")


def _arrays(outputs: Any) -> dict[str, np.ndarray]:
    """Normalize a two-tensor model result to auditable float32 arrays."""

    if isinstance(outputs, dict):
        tensors = (outputs["logits"], outputs["pred_boxes"])
    elif hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
        tensors = (outputs.logits, outputs.pred_boxes)
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        tensors = (outputs[0], outputs[1])
    else:
        raise TypeError(f"unexpected model output type: {type(outputs).__name__}")
    return {
        name: tensor.detach().float().cpu().numpy()
        for name, tensor in zip(OUTPUT_NAMES, tensors)
    }


def _autocast_context(torch, dtype):
    return (
        torch.autocast(device_type="cuda", dtype=dtype)
        if dtype is not None
        else contextlib.nullcontext()
    )


def _timing_summary(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms:
        return {"count": 0, "samples_ms": []}
    return {
        "count": len(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p50_ms": _percentile(samples_ms, 0.50),
        "p95_ms": _percentile(samples_ms, 0.95),
        "samples_ms": samples_ms,
    }


def _benchmark_eager(
    torch,
    module,
    inputs,
    *,
    autocast_dtype,
    warmup: int,
    runs: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure an eager mode after unmeasured warmup."""

    with torch.inference_mode(), _autocast_context(torch, autocast_dtype):
        for _ in range(warmup):
            module(*inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cuda_samples_ms: list[float] = []
        wall_samples_ms: list[float] = []
        outputs = None
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            outputs = module(*inputs)
            end.record()
            end.synchronize()
            wall_samples_ms.append((time.perf_counter() - wall_start) * 1000.0)
            cuda_samples_ms.append(float(start.elapsed_time(end)))
    assert outputs is not None
    cuda_summary = _timing_summary(cuda_samples_ms)
    metrics = {
        "status": "OK",
        "autocast_dtype": (
            str(autocast_dtype) if autocast_dtype is not None else None
        ),
        "native_output_dtypes": [str(outputs[0].dtype), str(outputs[1].dtype)],
        "warmup": warmup,
        "runs": runs,
        "mean_ms": cuda_summary["mean_ms"],
        "p50_ms": cuda_summary["p50_ms"],
        "p95_ms": cuda_summary["p95_ms"],
        "throughput_images_per_second": batch_size
        / (cuda_summary["mean_ms"] / 1000.0),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "samples_cuda_ms": cuda_samples_ms,
        "samples_wall_ms": wall_samples_ms,
        "wall_timing": _timing_summary(wall_samples_ms),
    }
    return metrics, _arrays(outputs)


def _single_forward(torch, module, inputs) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        outputs = module(*inputs)
        torch.cuda.synchronize()
    return _arrays(outputs)


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)


def _snapshot_dynamo_counters(torch) -> dict[str, Any]:
    """Read private Dynamo counters defensively across PyTorch versions."""

    try:
        counters = torch._dynamo.utils.counters
        groups: dict[str, dict[str, Any]] = {}
        for group, counter in sorted(counters.items(), key=lambda item: str(item[0])):
            groups[str(group)] = {
                str(key): _json_scalar(value)
                for key, value in sorted(
                    counter.items(), key=lambda item: str(item[0])
                )
            }
        return {"available": True, "groups": groups}
    except Exception as exc:  # pragma: no cover - version-dependent fallback
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _graph_break_summary(counter_snapshot: dict[str, Any]) -> dict[str, Any]:
    reasons = counter_snapshot.get("groups", {}).get("graph_break", {})
    count = sum(value for value in reasons.values() if isinstance(value, int))
    return {"count": count, "reasons": reasons}


def _reset_compile_state(torch) -> list[str]:
    """Reset process-local graph state without touching the persistent cache."""

    warnings: list[str] = []
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
        else:  # pragma: no cover - old PyTorch fallback
            torch._dynamo.reset()
    except Exception as exc:  # pragma: no cover - version-dependent fallback
        warnings.append(f"compiler reset failed: {type(exc).__name__}: {exc}")
    try:
        torch._dynamo.utils.counters.clear()
    except Exception as exc:  # pragma: no cover - version-dependent fallback
        warnings.append(f"counter reset failed: {type(exc).__name__}: {exc}")
    return warnings


def _compile_and_benchmark(
    torch,
    module,
    inputs,
    *,
    autocast_dtype,
    backend: str,
    fullgraph: bool,
    compile_warmup: int,
    runs: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Compile lazily, isolate compilation/warmup, then measure steady state."""

    record: dict[str, Any] = {
        "status": "STARTED",
        "backend": backend,
        "fullgraph": fullgraph,
        "dynamic": False,
        "autocast_dtype": (
            str(autocast_dtype) if autocast_dtype is not None else None
        ),
        "compile_warmup": compile_warmup,
        "runs": runs,
        "compilation_time_excluded_from_steady_state": True,
        "counter_reset_warnings": _reset_compile_state(torch),
        "counter_snapshots": {},
    }
    phase = "torch_compile_wrapper_creation"
    try:
        creation_start = time.perf_counter()
        compiled = torch.compile(
            module,
            backend=backend,
            fullgraph=fullgraph,
            dynamic=False,
        )
        record["compile_wrapper_creation_wall_ms"] = (
            time.perf_counter() - creation_start
        ) * 1000.0
        record["counter_snapshots"]["after_wrapper_creation"] = (
            _snapshot_dynamo_counters(torch)
        )

        phase = "first_lazy_compile_invocation"
        with torch.inference_mode(), _autocast_context(torch, autocast_dtype):
            torch.cuda.synchronize()
            first_start = time.perf_counter()
            outputs = compiled(*inputs)
            torch.cuda.synchronize()
            record["first_invocation_wall_ms"] = (
                time.perf_counter() - first_start
            ) * 1000.0
            record["first_invocation_includes_lazy_compilation"] = True
            record["counter_snapshots"]["after_first_invocation"] = (
                _snapshot_dynamo_counters(torch)
            )

            phase = "post_compile_warmup"
            warmup_wall_ms: list[float] = []
            for _ in range(compile_warmup):
                torch.cuda.synchronize()
                warmup_start = time.perf_counter()
                outputs = compiled(*inputs)
                torch.cuda.synchronize()
                warmup_wall_ms.append(
                    (time.perf_counter() - warmup_start) * 1000.0
                )
            record["post_first_invocation_warmup_wall"] = _timing_summary(
                warmup_wall_ms
            )
            record["counter_snapshots"]["after_post_compile_warmup"] = (
                _snapshot_dynamo_counters(torch)
            )

            phase = "steady_state_benchmark"
            torch.cuda.reset_peak_memory_stats()
            cuda_samples_ms: list[float] = []
            wall_samples_ms: list[float] = []
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                outputs = compiled(*inputs)
                end.record()
                end.synchronize()
                wall_samples_ms.append(
                    (time.perf_counter() - wall_start) * 1000.0
                )
                cuda_samples_ms.append(float(start.elapsed_time(end)))
        cuda_summary = _timing_summary(cuda_samples_ms)
        record.update(
            {
                "status": "OK",
                "native_output_dtypes": [
                    str(outputs[0].dtype),
                    str(outputs[1].dtype),
                ],
                "mean_ms": cuda_summary["mean_ms"],
                "p50_ms": cuda_summary["p50_ms"],
                "p95_ms": cuda_summary["p95_ms"],
                "throughput_images_per_second": batch_size
                / (cuda_summary["mean_ms"] / 1000.0),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "samples_cuda_ms": cuda_samples_ms,
                "samples_wall_ms": wall_samples_ms,
                "wall_timing": _timing_summary(wall_samples_ms),
            }
        )
        final_counters = _snapshot_dynamo_counters(torch)
        record["counter_snapshots"]["after_steady_state"] = final_counters
        record["graph_breaks"] = _graph_break_summary(final_counters)
        return record, _arrays(outputs)
    except Exception as exc:
        final_counters = _snapshot_dynamo_counters(torch)
        record.update(
            {
                "status": "FAILED",
                "failure_phase": phase,
                "failure": {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "graph_breaks": _graph_break_summary(final_counters),
            }
        )
        record["counter_snapshots"]["after_failure"] = final_counters
        return record, None


def _attach_correctness_gate(
    record: dict[str, Any],
    *,
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    reference_decisions: list[list[dict]],
    processor,
    torch,
    input_ids_cpu,
    target_sizes,
    threshold: float,
    text_threshold: float,
) -> None:
    raw = _raw_diff(reference, candidate)
    candidate_decisions = _postprocess(
        processor,
        torch,
        candidate,
        input_ids_cpu,
        target_sizes,
        threshold,
        text_threshold,
    )
    decision = _decision_diff(reference_decisions, candidate_decisions)
    raw_pattern_pass = _raw_outputs_have_valid_nonfinite_patterns(raw)
    decision_pass = bool(
        decision["strict_decision_equivalent_at_1e-3_and_half_px"]
    )
    record["raw_output_diff_vs_eager_fp32_pre"] = raw
    record["decision_diff_vs_eager_fp32_pre"] = decision
    record["correctness_gate"] = {
        "raw_shape_and_nonfinite_pattern_pass": raw_pattern_pass,
        "strict_detection_decision_pass": decision_pass,
        "pass": bool(raw_pattern_pass and decision_pass),
    }


def _outputs_bit_exact(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> bool:
    return all(
        reference[name].shape == candidate[name].shape
        and np.array_equal(reference[name], candidate[name], equal_nan=True)
        for name in OUTPUT_NAMES
    )


def _failure_record(exc: Exception, *, phase: str) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "failure_phase": phase,
        "failure": {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-outputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--compile-warmup",
        type=int,
        help="post-first-invocation warmup; defaults to --warmup",
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument(
        "--fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="require one full graph; default permits graph breaks and records them",
    )
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--code-commit")
    return parser


def main() -> int:
    import resource

    parser = _argument_parser()
    args = parser.parse_args()
    compile_warmup = (
        args.warmup if args.compile_warmup is None else args.compile_warmup
    )
    if args.warmup < 0 or compile_warmup < 0 or args.runs < 1:
        parser.error("runs must be positive and warmups non-negative")

    output_path = Path(args.output)
    outputs_path = output_path.with_suffix(".outputs.npz")
    existing = [str(path) for path in (output_path, outputs_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")

    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    inputs_path = Path(args.inputs)
    manifest_path = Path(args.baseline_manifest)
    baseline_outputs_path = Path(args.baseline_outputs)
    model_dir = Path(args.model_dir)
    with np.load(inputs_path) as loaded:
        missing = [name for name in INPUT_NAMES if name not in loaded]
        if missing:
            raise KeyError(f"frozen inputs are missing: {missing}")
        frozen = {name: np.asarray(loaded[name]).copy() for name in INPUT_NAMES}
    tensors = {
        name: torch.from_numpy(frozen[name]).to("cuda") for name in INPUT_NAMES
    }
    positional_inputs = tuple(tensors[name] for name in INPUT_NAMES)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).cuda().eval()

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
    batch_size = int(manifest["batch_size"])
    input_ids_cpu = torch.from_numpy(frozen["input_ids"])
    target_sizes = manifest["target_sizes"]
    saved_outputs: dict[str, np.ndarray] = {}
    modes: dict[str, dict[str, Any]] = {}

    eager_fp32, reference = _benchmark_eager(
        torch,
        wrapper,
        positional_inputs,
        autocast_dtype=None,
        warmup=args.warmup,
        runs=args.runs,
        batch_size=batch_size,
    )
    modes["eager_fp32_pre"] = eager_fp32
    for name in OUTPUT_NAMES:
        saved_outputs[f"eager_fp32_pre_{name}"] = reference[name]
    reference_decisions = _postprocess(
        processor,
        torch,
        reference,
        input_ids_cpu,
        target_sizes,
        args.threshold,
        args.text_threshold,
    )
    _attach_correctness_gate(
        modes["eager_fp32_pre"],
        reference=reference,
        candidate=reference,
        reference_decisions=reference_decisions,
        processor=processor,
        torch=torch,
        input_ids_cpu=input_ids_cpu,
        target_sizes=target_sizes,
        threshold=args.threshold,
        text_threshold=args.text_threshold,
    )

    if not torch.cuda.is_bf16_supported():
        modes["eager_bf16"] = {
            "status": "UNSUPPORTED",
            "reason": "torch.cuda.is_bf16_supported=false",
        }
    else:
        try:
            modes["eager_bf16"], eager_bf16_outputs = _benchmark_eager(
                torch,
                wrapper,
                positional_inputs,
                autocast_dtype=torch.bfloat16,
                warmup=args.warmup,
                runs=args.runs,
                batch_size=batch_size,
            )
            modes["eager_bf16"]["speedup_vs_eager_fp32_pre_p50"] = (
                eager_fp32["p50_ms"] / modes["eager_bf16"]["p50_ms"]
            )
            _attach_correctness_gate(
                modes["eager_bf16"],
                reference=reference,
                candidate=eager_bf16_outputs,
                reference_decisions=reference_decisions,
                processor=processor,
                torch=torch,
                input_ids_cpu=input_ids_cpu,
                target_sizes=target_sizes,
                threshold=args.threshold,
                text_threshold=args.text_threshold,
            )
            for name in OUTPUT_NAMES:
                saved_outputs[f"eager_bf16_{name}"] = eager_bf16_outputs[name]
        except Exception as exc:
            modes["eager_bf16"] = _failure_record(
                exc, phase="eager_bf16_benchmark"
            )

    compile_specs = (
        ("compile_fp32", None),
        ("compile_bf16", torch.bfloat16),
    )
    for mode, autocast_dtype in compile_specs:
        if autocast_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            modes[mode] = {
                "status": "UNSUPPORTED",
                "reason": "torch.cuda.is_bf16_supported=false",
            }
            continue
        record, candidate = _compile_and_benchmark(
            torch,
            wrapper,
            positional_inputs,
            autocast_dtype=autocast_dtype,
            backend=args.backend,
            fullgraph=args.fullgraph,
            compile_warmup=compile_warmup,
            runs=args.runs,
            batch_size=batch_size,
        )
        modes[mode] = record
        if candidate is None:
            continue
        record["speedup_vs_eager_fp32_pre_p50"] = (
            eager_fp32["p50_ms"] / record["p50_ms"]
        )
        _attach_correctness_gate(
            record,
            reference=reference,
            candidate=candidate,
            reference_decisions=reference_decisions,
            processor=processor,
            torch=torch,
            input_ids_cpu=input_ids_cpu,
            target_sizes=target_sizes,
            threshold=args.threshold,
            text_threshold=args.text_threshold,
        )
        for name in OUTPUT_NAMES:
            saved_outputs[f"{mode}_{name}"] = candidate[name]

    try:
        eager_post_outputs = _single_forward(
            torch, wrapper, positional_inputs
        )
        modes["eager_fp32_post"] = {"status": "OK"}
        _attach_correctness_gate(
            modes["eager_fp32_post"],
            reference=reference,
            candidate=eager_post_outputs,
            reference_decisions=reference_decisions,
            processor=processor,
            torch=torch,
            input_ids_cpu=input_ids_cpu,
            target_sizes=target_sizes,
            threshold=args.threshold,
            text_threshold=args.text_threshold,
        )
        for name in OUTPUT_NAMES:
            saved_outputs[f"eager_fp32_post_{name}"] = eager_post_outputs[name]
        eager_stability = {
            "bit_exact": _outputs_bit_exact(reference, eager_post_outputs),
            "raw_output_diff": _raw_diff(reference, eager_post_outputs),
            "correctness_gate_pass": modes["eager_fp32_post"][
                "correctness_gate"
            ]["pass"],
        }
        eager_stability["pass"] = bool(
            eager_stability["bit_exact"]
            and eager_stability["correctness_gate_pass"]
        )
    except Exception as exc:
        modes["eager_fp32_post"] = _failure_record(
            exc, phase="eager_fp32_post_stability"
        )
        eager_stability = {"pass": False, "failure": modes["eager_fp32_post"]}

    with np.load(baseline_outputs_path) as loaded:
        missing_baseline = [name for name in OUTPUT_NAMES if name not in loaded]
        if missing_baseline:
            raise KeyError(
                f"frozen baseline outputs are missing: {missing_baseline}"
            )
        frozen_reference = {
            name: np.asarray(loaded[name]).copy() for name in OUTPUT_NAMES
        }
    frozen_reference_diff = _raw_diff(frozen_reference, reference)
    frozen_reference_decisions = _postprocess(
        processor,
        torch,
        frozen_reference,
        input_ids_cpu,
        target_sizes,
        args.threshold,
        args.text_threshold,
    )
    frozen_reference_decision_diff = _decision_diff(
        frozen_reference_decisions, reference_decisions
    )
    frozen_reference_gate = {
        "finite_values_bit_exact_with_matching_nonfinite_values": (
            _outputs_bit_exact(frozen_reference, reference)
        ),
        "raw_shape_and_nonfinite_pattern_pass": (
            _raw_outputs_have_valid_nonfinite_patterns(frozen_reference_diff)
        ),
        "strict_detection_decision_pass": bool(
            frozen_reference_decision_diff[
                "strict_decision_equivalent_at_1e-3_and_half_px"
            ]
        ),
    }
    frozen_reference_gate["pass"] = all(frozen_reference_gate.values())

    candidate_summary = {
        mode: {
            "status": modes[mode]["status"],
            "correctness_gate_pass": bool(
                modes[mode].get("correctness_gate", {}).get("pass", False)
            ),
        }
        for mode in REQUESTED_CANDIDATES
    }
    all_candidates_pass = all(
        item["status"] == "OK" and item["correctness_gate_pass"]
        for item in candidate_summary.values()
    )
    overall_pass = bool(
        frozen_reference_gate["pass"]
        and all_candidates_pass
        and eager_stability["pass"]
    )
    if not frozen_reference_gate["pass"]:
        verdict = "NO_GO_EAGER_FP32_DOES_NOT_MATCH_FROZEN_ORACLE"
    elif not eager_stability["pass"]:
        verdict = "NO_GO_EAGER_FP32_UNSTABLE_AFTER_CANDIDATES"
    elif any(item["status"] != "OK" for item in candidate_summary.values()):
        verdict = "NO_GO_CANDIDATE_EXECUTION_FAILED_OR_UNSUPPORTED"
    elif not all_candidates_pass:
        verdict = "NO_GO_CANDIDATE_CORRECTNESS_GATE_FAILED"
    else:
        verdict = "PASS_ALL_EAGER_AND_TORCH_COMPILE_MODES"

    allow_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    cudnn_allow_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
    suppress_errors = getattr(torch._dynamo.config, "suppress_errors", None)
    result = {
        "schema_version": "1.0",
        "scope": "SF1_L2_PYTORCH_COMPILE_FP32_BF16_FEASIBILITY_ONLY",
        "created_at_unix": int(time.time()),
        "status": "PASS" if overall_pass else "NO_GO",
        "verdict": verdict,
        "code_commit": args.code_commit or manifest.get("code_commit", "unknown"),
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "compile_configuration": {
            "backend": args.backend,
            "fullgraph": args.fullgraph,
            "dynamic": False,
            "torch_dynamo_suppress_errors": suppress_errors,
            "cuda_matmul_allow_tf32": allow_tf32,
            "cudnn_allow_tf32": cudnn_allow_tf32,
        },
        "inputs": {
            "path": str(inputs_path),
            "sha256": _sha256(inputs_path),
            "batch_size": batch_size,
            "baseline_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "prompt": manifest.get("prompt"),
            "target_sizes": target_sizes,
        },
        "postprocess_thresholds": {
            "box": args.threshold,
            "text": args.text_threshold,
        },
        "frozen_fp32_reference": {
            "path": str(baseline_outputs_path),
            "sha256": _sha256(baseline_outputs_path),
            "raw_output_diff_vs_eager_fp32_pre": frozen_reference_diff,
            "decision_diff_vs_eager_fp32_pre": frozen_reference_decision_diff,
            "gate": frozen_reference_gate,
        },
        "modes": modes,
        "candidate_summary": candidate_summary,
        "eager_fp32_pre_post_stability": eager_stability,
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        * 1024,
        "measurement_note": (
            "the first lazy compile invocation and all post-compile warmup calls "
            "are excluded from steady-state samples; every candidate is gated "
            "against the same-process eager FP32 raw shape/non-finite pattern and "
            "strict post-processed detection decisions; that eager reference must "
            "itself remain bit-exact with the frozen FP32 oracle"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with outputs_path.open("xb") as handle:
        np.savez_compressed(handle, **saved_outputs)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

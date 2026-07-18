#!/usr/bin/env python
"""Benchmark a Grounding DINO TensorRT engine and compare frozen outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _torch_dtype(trt, torch, dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


def _postprocess(processor, logits, pred_boxes, input_ids, target_sizes, threshold, text_threshold):
    class Outputs:
        pass

    outputs = Outputs()
    outputs.logits = logits
    outputs.pred_boxes = pred_boxes
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )
    signatures = []
    for result in results:
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        signatures.append(
            [
                {
                    "label": str(label),
                    "score": float(score),
                    "box": [float(value) for value in box],
                }
                for label, score, box in zip(labels, result["scores"], result["boxes"])
            ]
        )
    return signatures


def _decision_diff(left, right):
    structure_equal = len(left) == len(right)
    max_score_delta = 0.0
    max_box_delta = 0.0
    for left_frame, right_frame in zip(left, right):
        structure_equal = structure_equal and len(left_frame) == len(right_frame)
        for left_item, right_item in zip(left_frame, right_frame):
            structure_equal = structure_equal and left_item["label"] == right_item["label"]
            max_score_delta = max(
                max_score_delta,
                abs(left_item["score"] - right_item["score"]),
            )
            max_box_delta = max(
                max_box_delta,
                *(abs(a - b) for a, b in zip(left_item["box"], right_item["box"])),
            )
    return {
        "structure_equal": structure_equal,
        "max_score_delta": max_score_delta,
        "max_box_delta_px": max_box_delta,
        "strict_decision_equivalent_at_1e-3_and_half_px": (
            structure_equal and max_score_delta <= 1e-3 and max_box_delta <= 0.5
        ),
        "torch_detection_counts": [len(items) for items in left],
        "tensorrt_detection_counts": [len(items) for items in right],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--torch-outputs", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("runs must be positive and warmup non-negative")

    import tensorrt as trt
    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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

    frozen_inputs = np.load(args.inputs)
    tensors = {}
    input_names = []
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
            if name not in frozen_inputs:
                raise KeyError(f"engine input missing from frozen NPZ: {name}")
            dtype = _torch_dtype(trt, torch, engine.get_tensor_dtype(name))
            tensor = torch.as_tensor(frozen_inputs[name], dtype=dtype, device="cuda").contiguous()
            if not context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"rejected shape for {name}: {tuple(tensor.shape)}")
            tensors[name] = tensor
        else:
            output_names.append(name)

    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"TensorRT could not infer shapes for: {unresolved}")
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(f"unresolved output shape for {name}: {shape}")
        dtype = _torch_dtype(trt, torch, engine.get_tensor_dtype(name))
        tensors[name] = torch.empty(shape, dtype=dtype, device="cuda")

    for name, tensor in tensors.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"failed to bind TensorRT tensor: {name}")

    torch.cuda.current_stream().synchronize()
    stream = torch.cuda.Stream()
    for _ in range(args.warmup):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT warmup execution failed")
    stream.synchronize()

    timings_ms: list[float] = []
    for _ in range(args.runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        end.record(stream)
        end.synchronize()
        timings_ms.append(float(start.elapsed_time(end)))

    trt_outputs = {name: tensors[name].float().cpu().numpy() for name in output_names}
    np.savez_compressed(Path(args.output).with_suffix(".outputs.npz"), **trt_outputs)
    baseline_outputs = np.load(args.torch_outputs)
    raw_diff = {}
    for name in ("logits", "pred_boxes"):
        baseline_finite = np.isfinite(baseline_outputs[name])
        trt_finite = np.isfinite(trt_outputs[name])
        jointly_finite = baseline_finite & trt_finite
        delta = np.abs(
            baseline_outputs[name][jointly_finite] - trt_outputs[name][jointly_finite]
        )
        raw_diff[name] = {
            "baseline_nonfinite_count": int((~baseline_finite).sum()),
            "tensorrt_nonfinite_count": int((~trt_finite).sum()),
            "jointly_finite_count": int(jointly_finite.sum()),
            "max_abs_on_jointly_finite": float(delta.max()) if delta.size else None,
            "mean_abs_on_jointly_finite": float(delta.mean()) if delta.size else None,
        }

    baseline_manifest = json.loads(Path(args.baseline_manifest).read_text())
    processor = AutoProcessor.from_pretrained(args.model_dir)
    input_ids = torch.from_numpy(frozen_inputs["input_ids"])
    torch_decisions = _postprocess(
        processor,
        torch.from_numpy(baseline_outputs["logits"]),
        torch.from_numpy(baseline_outputs["pred_boxes"]),
        input_ids,
        baseline_manifest["target_sizes"],
        args.threshold,
        args.text_threshold,
    )
    trt_decisions = _postprocess(
        processor,
        torch.from_numpy(trt_outputs["logits"]),
        torch.from_numpy(trt_outputs["pred_boxes"]),
        input_ids,
        baseline_manifest["target_sizes"],
        args.threshold,
        args.text_threshold,
    )
    decision_diff = _decision_diff(torch_decisions, trt_decisions)
    decision_diff["all_tensorrt_outputs_finite"] = all(
        item["tensorrt_nonfinite_count"] == 0 for item in raw_diff.values()
    )
    decision_diff["strict_decision_equivalent_at_1e-3_and_half_px"] = (
        decision_diff["strict_decision_equivalent_at_1e-3_and_half_px"]
        and decision_diff["all_tensorrt_outputs_finite"]
    )

    mean_ms = statistics.fmean(timings_ms)
    batch_size = int(baseline_manifest["batch_size"])
    baseline_p50 = float(baseline_manifest["pytorch_core"]["p50_ms"])
    result = {
        "schema_version": "1.0",
        "scope": "SF1-L2_TENSORRT_FEASIBILITY_ONLY",
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or baseline_manifest.get("code_commit", "unknown"),
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "tensorrt": trt.__version__,
        },
        "engine": {
            "path": str(engine_path),
            "sha256": _sha256(engine_path),
            "size_bytes": engine_path.stat().st_size,
            "device_memory_size_bytes": int(
                engine.device_memory_size_v2
                if hasattr(engine, "device_memory_size_v2")
                else engine.device_memory_size
            ),
            "inputs": input_names,
            "outputs": output_names,
        },
        "tensorrt_core": {
            "cuda_stream": "non_default",
            "warmup": args.warmup,
            "runs": args.runs,
            "mean_ms": mean_ms,
            "p50_ms": _percentile(timings_ms, 0.50),
            "p95_ms": _percentile(timings_ms, 0.95),
            "throughput_images_per_second": batch_size / (mean_ms / 1000.0),
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            * 1024,
            "samples_ms": timings_ms,
        },
        "speedup_vs_pytorch_p50": baseline_p50 / _percentile(timings_ms, 0.50),
        "raw_output_diff": raw_diff,
        "decision_diff": decision_diff,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if decision_diff["strict_decision_equivalent_at_1e-3_and_half_px"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

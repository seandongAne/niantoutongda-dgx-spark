#!/usr/bin/env python
"""Benchmark Grounding DINO eager FP32 and CUDA autocast precision modes.

This is a feasibility probe.  It reuses the frozen inputs produced by
``gdino_onnx_probe.py`` and compares every candidate against an FP32 forward
from the same process before reporting speedup.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


INPUT_NAMES = (
    "pixel_values",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "pixel_mask",
)


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


def _raw_diff(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict:
    report = {}
    for name in ("logits", "pred_boxes"):
        left = reference[name]
        right = candidate[name]
        if left.shape != right.shape:
            report[name] = {
                "shape_equal": False,
                "reference_shape": list(left.shape),
                "candidate_shape": list(right.shape),
            }
            continue
        left_finite = np.isfinite(left)
        right_finite = np.isfinite(right)
        jointly_finite = left_finite & right_finite
        delta = np.abs(left[jointly_finite] - right[jointly_finite])
        report[name] = {
            "shape_equal": True,
            "reference_all_finite": bool(left_finite.all()),
            "candidate_all_finite": bool(right_finite.all()),
            "reference_nan_count": int(np.isnan(left).sum()),
            "candidate_nan_count": int(np.isnan(right).sum()),
            "reference_posinf_count": int(np.isposinf(left).sum()),
            "candidate_posinf_count": int(np.isposinf(right).sum()),
            "reference_neginf_count": int(np.isneginf(left).sum()),
            "candidate_neginf_count": int(np.isneginf(right).sum()),
            "nonfinite_pattern_equal": bool(
                np.array_equal(np.isnan(left), np.isnan(right))
                and np.array_equal(np.isposinf(left), np.isposinf(right))
                and np.array_equal(np.isneginf(left), np.isneginf(right))
            ),
            "jointly_finite_count": int(jointly_finite.sum()),
            "max_abs_on_jointly_finite": float(delta.max()) if delta.size else None,
            "mean_abs_on_jointly_finite": float(delta.mean()) if delta.size else None,
        }
    return report


def _raw_outputs_have_valid_nonfinite_patterns(report: dict) -> bool:
    """Permit matching -inf masks, but reject NaN, +inf, or mask drift."""

    return bool(
        report
        and all(
            item.get("shape_equal", False)
            and item.get("nonfinite_pattern_equal", False)
            and item.get("reference_nan_count", 1) == 0
            and item.get("candidate_nan_count", 1) == 0
            and item.get("reference_posinf_count", 1) == 0
            and item.get("candidate_posinf_count", 1) == 0
            for item in report.values()
        )
    )


def _postprocess(
    processor,
    torch,
    outputs: dict[str, np.ndarray],
    input_ids,
    target_sizes,
    threshold: float,
    text_threshold: float,
) -> list[list[dict]]:
    class Outputs:
        pass

    wrapped = Outputs()
    wrapped.logits = torch.from_numpy(outputs["logits"])
    wrapped.pred_boxes = torch.from_numpy(outputs["pred_boxes"])
    processed = processor.post_process_grounded_object_detection(
        wrapped,
        input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )
    signatures = []
    for result in processed:
        labels = result.get("text_labels", result.get("labels", []))
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


def _decision_diff(reference: list[list[dict]], candidate: list[list[dict]]) -> dict:
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
                *(abs(a - b) for a, b in zip(left_item["box"], right_item["box"])),
            )
    strict = structure_equal and max_score_delta <= 1e-3 and max_box_delta <= 0.5
    return {
        "structure_equal": structure_equal,
        "max_score_delta": max_score_delta,
        "max_box_delta_px": max_box_delta,
        "strict_decision_equivalent_at_1e-3_and_half_px": strict,
        "reference_detection_counts": [len(items) for items in reference],
        "candidate_detection_counts": [len(items) for items in candidate],
    }


def main() -> int:
    import resource

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-outputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("runs must be positive and warmup non-negative")

    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    inputs_path = Path(args.inputs)
    manifest_path = Path(args.baseline_manifest)
    baseline_outputs_path = Path(args.baseline_outputs)
    model_dir = Path(args.model_dir)
    frozen = np.load(inputs_path)
    missing = [name for name in INPUT_NAMES if name not in frozen]
    if missing:
        raise KeyError(f"frozen inputs are missing: {missing}")
    tensors = {
        name: torch.from_numpy(frozen[name]).to("cuda")
        for name in INPUT_NAMES
    }
    positional_inputs = tuple(tensors[name] for name in INPUT_NAMES)
    manifest = json.loads(manifest_path.read_text())
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).cuda().eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask):
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

    modes = [
        ("fp32", None),
        ("amp_fp16", torch.float16),
        ("amp_bf16", torch.bfloat16),
        ("fp32_repeat", None),
    ]
    measured = {}
    saved_outputs: dict[str, np.ndarray] = {}
    for mode, autocast_dtype in modes:
        if mode == "amp_bf16" and not torch.cuda.is_bf16_supported():
            measured[mode] = {"supported": False, "reason": "torch.cuda.is_bf16_supported=false"}
            continue
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), context:
            for _ in range(args.warmup):
                wrapper(*positional_inputs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings_ms: list[float] = []
            outputs = None
            for _ in range(args.runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                outputs = wrapper(*positional_inputs)
                end.record()
                end.synchronize()
                timings_ms.append(float(start.elapsed_time(end)))
        assert outputs is not None
        arrays = {
            "logits": outputs[0].detach().float().cpu().numpy(),
            "pred_boxes": outputs[1].detach().float().cpu().numpy(),
        }
        saved_outputs[f"{mode}_logits"] = arrays["logits"]
        saved_outputs[f"{mode}_pred_boxes"] = arrays["pred_boxes"]
        measured[mode] = {
            "supported": True,
            "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
            "native_output_dtypes": [str(outputs[0].dtype), str(outputs[1].dtype)],
            "warmup": args.warmup,
            "runs": args.runs,
            "mean_ms": statistics.fmean(timings_ms),
            "p50_ms": _percentile(timings_ms, 0.50),
            "p95_ms": _percentile(timings_ms, 0.95),
            "throughput_images_per_second": int(manifest["batch_size"])
            / (statistics.fmean(timings_ms) / 1000.0),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "samples_ms": timings_ms,
        }

    reference = {
        "logits": saved_outputs["fp32_logits"],
        "pred_boxes": saved_outputs["fp32_pred_boxes"],
    }
    input_ids_cpu = torch.from_numpy(frozen["input_ids"])
    reference_decisions = _postprocess(
        processor,
        torch,
        reference,
        input_ids_cpu,
        manifest["target_sizes"],
        args.threshold,
        args.text_threshold,
    )
    for mode, _ in modes:
        if not measured[mode].get("supported"):
            continue
        candidate = {
            "logits": saved_outputs[f"{mode}_logits"],
            "pred_boxes": saved_outputs[f"{mode}_pred_boxes"],
        }
        raw = _raw_diff(reference, candidate)
        decisions = _postprocess(
            processor,
            torch,
            candidate,
            input_ids_cpu,
            manifest["target_sizes"],
            args.threshold,
            args.text_threshold,
        )
        decision = _decision_diff(reference_decisions, decisions)
        decision["nonfinite_patterns_equal"] = all(
            item.get("nonfinite_pattern_equal", False) for item in raw.values()
        )
        decision["output_nonfinite_patterns_valid"] = (
            _raw_outputs_have_valid_nonfinite_patterns(raw)
        )
        decision["strict_decision_equivalent_at_1e-3_and_half_px"] = bool(
            decision["strict_decision_equivalent_at_1e-3_and_half_px"]
            and decision["nonfinite_patterns_equal"]
            and decision["output_nonfinite_patterns_valid"]
        )
        measured[mode]["raw_output_diff_vs_in_process_fp32"] = raw
        measured[mode]["decision_diff_vs_in_process_fp32"] = decision
        measured[mode]["speedup_vs_in_process_fp32_p50"] = (
            measured["fp32"]["p50_ms"] / measured[mode]["p50_ms"]
        )

    frozen_reference = {
        name: np.asarray(value)
        for name, value in np.load(baseline_outputs_path).items()
        if name in {"logits", "pred_boxes"}
    }
    result = {
        "schema_version": "1.0",
        "scope": "SF1_L2_PYTORCH_PRECISION_FEASIBILITY_ONLY",
        "created_at_unix": int(time.time()),
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
        "inputs": {
            "path": str(inputs_path),
            "sha256": _sha256(inputs_path),
            "batch_size": int(manifest["batch_size"]),
            "baseline_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "prompt": manifest.get("prompt"),
            "target_sizes": manifest.get("target_sizes"),
        },
        "postprocess_thresholds": {
            "box": args.threshold,
            "text": args.text_threshold,
        },
        "frozen_fp32_reference": {
            "path": str(baseline_outputs_path),
            "sha256": _sha256(baseline_outputs_path),
            "raw_output_diff_vs_in_process_fp32": _raw_diff(frozen_reference, reference),
        },
        "modes": measured,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "measurement_note": (
            "model weights remain FP32; autocast applies only to forward; "
            "speedups use the FP32 pass from this process"
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output_path.with_suffix(".outputs.npz"), **saved_outputs)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    candidates = [mode for mode in ("amp_fp16", "amp_bf16") if measured[mode].get("supported")]
    return 0 if all(
        measured[mode]["decision_diff_vs_in_process_fp32"][
            "strict_decision_equivalent_at_1e-3_and_half_px"
        ]
        for mode in candidates
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())

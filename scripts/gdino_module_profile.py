#!/usr/bin/env python
"""Profile non-overlapping Grounding DINO modules with CUDA events.

This diagnostic is pinned to the Hugging Face Transformers 5.13.1 Grounding
DINO module hierarchy.  It measures eager-FP32 whole-forward latency and the
text backbone, visual backbone, encoder, and decoder as mutually exclusive
regions.  The residual contains projection, proposal/top-k, prediction heads,
Python dispatch, and any other work outside those four regions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


INPUT_NAMES = (
    "pixel_values",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "pixel_mask",
)
OUTPUT_NAMES = ("logits", "pred_boxes")
EXPECTED_TRANSFORMERS_VERSION = "5.13.1"
COMPONENTS = ("text_backbone", "visual_backbone", "encoder", "decoder")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summary(values: list[float], *, include_samples: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }
    if include_samples:
        result["samples"] = values
    return result


def _output_arrays(outputs) -> dict[str, np.ndarray]:
    return {
        "logits": outputs.logits.detach().float().cpu().numpy(),
        "pred_boxes": outputs.pred_boxes.detach().float().cpu().numpy(),
    }


def _array_diff(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "equivalent": False,
        }
    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    jointly_finite = reference_finite & candidate_finite
    delta = np.abs(reference[jointly_finite] - candidate[jointly_finite])
    nonfinite_pattern_equal = bool(
        np.array_equal(np.isnan(reference), np.isnan(candidate))
        and np.array_equal(np.isposinf(reference), np.isposinf(candidate))
        and np.array_equal(np.isneginf(reference), np.isneginf(candidate))
    )
    finite_bit_exact = bool(
        np.array_equal(reference[jointly_finite], candidate[jointly_finite])
    )
    return {
        "shape_equal": True,
        "nonfinite_pattern_equal": nonfinite_pattern_equal,
        "finite_values_bit_exact": finite_bit_exact,
        "max_abs_on_jointly_finite": float(delta.max()) if delta.size else None,
        "mean_abs_on_jointly_finite": float(delta.mean()) if delta.size else None,
        "equivalent": bool(nonfinite_pattern_equal and finite_bit_exact),
    }


def _output_diff(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, Any]:
    comparisons = {
        name: _array_diff(reference[name], candidate[name]) for name in OUTPUT_NAMES
    }
    return {
        "comparisons": comparisons,
        "equivalent": all(item["equivalent"] for item in comparisons.values()),
    }


def _benchmark_whole(torch, model, tensors, runs: int):
    samples: list[float] = []
    outputs = None
    with torch.inference_mode():
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = model(**tensors, return_dict=True)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    if outputs is None:
        raise RuntimeError("whole-forward benchmark produced no output")
    return samples, outputs


def _module_path(model, target) -> str:
    matches = [name or "<root>" for name, module in model.named_modules() if module is target]
    if len(matches) != 1:
        raise RuntimeError(
            f"target module must occur exactly once in named_modules; found {matches}"
        )
    return matches[0]


def _validate_hierarchy(model, modeling_grounding_dino, BertModel):
    outer_type = modeling_grounding_dino.GroundingDinoForObjectDetection
    core_type = modeling_grounding_dino.GroundingDinoModel
    required_types = {
        "text_backbone": BertModel,
        "visual_backbone": modeling_grounding_dino.GroundingDinoConvModel,
        "encoder": modeling_grounding_dino.GroundingDinoEncoder,
        "decoder": modeling_grounding_dino.GroundingDinoDecoder,
    }
    if type(model) is not outer_type:
        raise RuntimeError(f"expected exact {outer_type}, found {type(model)}")
    core = getattr(model, "model", None)
    if type(core) is not core_type:
        raise RuntimeError(f"expected exact {core_type}, found {type(core)}")

    modules = {
        "text_backbone": getattr(core, "text_backbone", None),
        "visual_backbone": getattr(core, "backbone", None),
        "encoder": getattr(core, "encoder", None),
        "decoder": getattr(core, "decoder", None),
    }
    for label, expected_type in required_types.items():
        if type(modules[label]) is not expected_type:
            raise RuntimeError(
                f"expected core.{label} to have exact type {expected_type}; "
                f"found {type(modules[label])}"
            )

    paths = {label: _module_path(model, module) for label, module in modules.items()}
    for left in COMPONENTS:
        for right in COMPONENTS:
            if left == right:
                continue
            if paths[right].startswith(paths[left] + "."):
                raise RuntimeError(
                    f"profile regions are nested, not exclusive: {paths[left]} owns {paths[right]}"
                )
    if len({id(module) for module in modules.values()}) != len(modules):
        raise RuntimeError("profile component identities are not unique")
    return core, modules, paths


class _CudaRegionTracker:
    """Record exclusive module regions and fail on nesting or call-count drift."""

    def __init__(self, torch, modules: dict[str, Any]):
        self.torch = torch
        self.modules = modules
        self.recording = False
        self.active: str | None = None
        self.sequence: list[str] = []
        self.calls = {name: 0 for name in COMPONENTS}
        self.events: dict[str, list[Any]] = {}
        self.handles = []
        for name, module in modules.items():
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._post_hook(name)))

    def _pre_hook(self, name: str):
        def hook(_module, _inputs):
            if not self.recording:
                return
            if self.active is not None:
                raise RuntimeError(
                    f"profile regions overlap dynamically: {name} began inside {self.active}"
                )
            self.active = name
            self.calls[name] += 1
            self.sequence.append(f"begin:{name}")
            self.events[name][0].record()

        return hook

    def _post_hook(self, name: str):
        def hook(_module, _inputs, _output):
            if not self.recording:
                return
            if self.active != name:
                raise RuntimeError(
                    f"profile region ended out of order: {name}, active={self.active}"
                )
            self.events[name][1].record()
            self.sequence.append(f"end:{name}")
            self.active = None

        return hook

    def begin(self) -> None:
        if self.recording or self.active is not None:
            raise RuntimeError("region tracker was not idle at iteration start")
        self.calls = {name: 0 for name in COMPONENTS}
        self.events = {
            name: [
                self.torch.cuda.Event(enable_timing=True),
                self.torch.cuda.Event(enable_timing=True),
            ]
            for name in COMPONENTS
        }
        self.sequence = []
        self.recording = True

    def finish(self) -> tuple[dict[str, list[Any]], list[str], dict[str, int]]:
        self.recording = False
        if self.active is not None:
            raise RuntimeError(f"unterminated profile region: {self.active}")
        if self.calls != {name: 1 for name in COMPONENTS}:
            raise RuntimeError(f"expected one call per component; observed {self.calls}")
        pairs = [self.sequence[index : index + 2] for index in range(0, len(self.sequence), 2)]
        observed_order: list[str] = []
        for pair in pairs:
            if len(pair) != 2 or not pair[0].startswith("begin:"):
                raise RuntimeError(f"malformed region event sequence: {self.sequence}")
            name = pair[0].removeprefix("begin:")
            if pair[1] != f"end:{name}":
                raise RuntimeError(f"non-exclusive region event sequence: {self.sequence}")
            observed_order.append(name)
        if sorted(observed_order) != sorted(COMPONENTS):
            raise RuntimeError(
                f"component event sequence is incomplete or duplicated: {self.sequence}"
            )
        return dict(self.events), list(self.sequence), dict(self.calls)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True, help="frozen sample_inputs.npz")
    parser.add_argument(
        "--baseline-outputs",
        required=True,
        help="frozen torch_outputs.npz; profiled eager FP32 must match exactly",
    )
    parser.add_argument("--output", required=True, help="new JSON artifact; overwrite refused")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--tf32-policy",
        choices=("disabled", "pytorch-default", "enabled"),
        default="pytorch-default",
        help=(
            "disabled=false/false, pytorch-default=matmul-false/cudnn-true, "
            "enabled=true/true"
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_path}")
    inputs_path = Path(args.inputs).resolve()
    baseline_outputs_path = Path(args.baseline_outputs).resolve()
    model_dir = Path(args.model_dir).resolve()
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not baseline_outputs_path.is_file():
        raise FileNotFoundError(baseline_outputs_path)

    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection
    from transformers.models.bert.modeling_bert import BertModel
    from transformers.models.grounding_dino import modeling_grounding_dino

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"this profiler is pinned to transformers=={EXPECTED_TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with np.load(inputs_path, allow_pickle=False) as loaded:
        missing = [name for name in INPUT_NAMES if name not in loaded]
        extras = sorted(set(loaded.files) - set(INPUT_NAMES))
        if missing:
            raise KeyError(f"frozen inputs are missing required arrays: {missing}")
        frozen = {name: np.array(loaded[name], copy=True) for name in INPUT_NAMES}

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tf32_states = {
        "disabled": (False, False, "highest"),
        "pytorch-default": (False, True, "highest"),
        "enabled": (True, True, "high"),
    }
    matmul_tf32, cudnn_tf32, matmul_precision = tf32_states[args.tf32_policy]
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)

    tensors = {
        name: torch.from_numpy(value).to(device="cuda") for name, value in frozen.items()
    }
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).cuda().eval()
    core, modules, module_paths = _validate_hierarchy(
        model, modeling_grounding_dino, BertModel
    )

    with np.load(baseline_outputs_path, allow_pickle=False) as loaded:
        missing_outputs = [name for name in OUTPUT_NAMES if name not in loaded]
        if missing_outputs:
            raise KeyError(
                f"frozen baseline is missing outputs: {missing_outputs}"
            )
        frozen_outputs = {
            name: np.array(loaded[name], copy=True) for name in OUTPUT_NAMES
        }

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(**tensors, return_dict=True)
    torch.cuda.synchronize()
    uninstrumented_samples, uninstrumented_outputs = _benchmark_whole(
        torch, model, tensors, args.runs
    )
    uninstrumented_arrays = _output_arrays(uninstrumented_outputs)
    frozen_oracle_gate = _output_diff(frozen_outputs, uninstrumented_arrays)
    if not frozen_oracle_gate["equivalent"]:
        raise RuntimeError(
            "uninstrumented eager FP32 does not match the frozen oracle under "
            f"tf32-policy={args.tf32_policy}: {frozen_oracle_gate}"
        )

    tracker = _CudaRegionTracker(torch, modules)
    whole_samples: list[float] = []
    component_samples = {name: [] for name in COMPONENTS}
    remainder_samples: list[float] = []
    share_samples = {name: [] for name in (*COMPONENTS, "remainder")}
    sequences: list[list[str]] = []
    call_counts: list[dict[str, int]] = []
    try:
        with torch.inference_mode():
            allocated_before = int(torch.cuda.memory_allocated())
            reserved_before = int(torch.cuda.memory_reserved())
            torch.cuda.reset_peak_memory_stats()

            for _ in range(args.runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                tracker.begin()
                start.record()
                outputs = model(**tensors, return_dict=True)
                end.record()
                events, sequence, calls = tracker.finish()
                end.synchronize()
                if (
                    outputs.logits.dtype != torch.float32
                    or outputs.pred_boxes.dtype != torch.float32
                ):
                    raise RuntimeError(
                        "eager FP32 contract violated: "
                        f"logits={outputs.logits.dtype}, boxes={outputs.pred_boxes.dtype}"
                    )
                whole_ms = float(start.elapsed_time(end))
                per_component = {
                    name: float(pair[0].elapsed_time(pair[1]))
                    for name, pair in events.items()
                }
                remainder_ms = whole_ms - sum(per_component.values())
                tolerance_ms = max(0.05, whole_ms * 1e-4)
                if remainder_ms < -tolerance_ms:
                    raise RuntimeError(
                        "exclusive component time exceeds whole forward: "
                        f"whole={whole_ms:.6f}, components={per_component}"
                    )
                remainder_ms = max(0.0, remainder_ms)
                whole_samples.append(whole_ms)
                remainder_samples.append(remainder_ms)
                sequences.append(sequence)
                call_counts.append(calls)
                for name, value in per_component.items():
                    component_samples[name].append(value)
                    share_samples[name].append(value / whole_ms)
                share_samples["remainder"].append(remainder_ms / whole_ms)
    finally:
        tracker.close()

    config_dict = core.config.to_dict()
    uninstrumented_summary = _summary(uninstrumented_samples)
    instrumented_summary = _summary(whole_samples)
    instrumentation_output_gate = _output_diff(
        uninstrumented_arrays, _output_arrays(outputs)
    )
    if not instrumentation_output_gate["equivalent"]:
        raise RuntimeError(
            "profiling hooks changed eager FP32 outputs: "
            f"{instrumentation_output_gate}"
        )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    result = {
        "schema_version": 1,
        "probe": "gdino_module_profile",
        "scope": "diagnostic_only_eager_fp32_component_profile",
        "created_at_unix": int(time.time()),
        "precision": {
            "mode": "eager_fp32",
            "autocast": False,
            "tf32_policy": args.tf32_policy,
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
                if hasattr(torch, "get_float32_matmul_precision")
                else None
            ),
        },
        "inputs": {
            "path": str(inputs_path),
            "sha256": _sha256_file(inputs_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in frozen.items()
            },
            "unexpected_arrays_ignored": extras,
            "baseline_outputs": {
                "path": str(baseline_outputs_path),
                "sha256": _sha256_file(baseline_outputs_path),
            },
        },
        "model": {
            "directory": str(model_dir),
            "outer_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "core_class": f"{type(core).__module__}.{type(core).__qualname__}",
            "config_sha256": _canonical_sha256(config_dict),
            "config": config_dict,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "device_capability": list(torch.cuda.get_device_capability()),
        },
        "measurement": {
            "timer": "torch.cuda.Event on current stream",
            "warmup": args.warmup,
            "runs": args.runs,
            "whole_forward_ms": uninstrumented_summary,
            "instrumented_whole_forward_ms": instrumented_summary,
            "instrumentation_p50_ratio": (
                instrumented_summary["p50"] / uninstrumented_summary["p50"]
            ),
            "components": {
                name: {
                    "module_path": module_paths[name],
                    "class": (
                        f"{type(modules[name]).__module__}."
                        f"{type(modules[name]).__qualname__}"
                    ),
                    "latency_ms": _summary(component_samples[name]),
                    "share_of_whole": _summary(share_samples[name]),
                    "p50_ms_over_whole_p50": _summary(
                        component_samples[name], include_samples=False
                    )["p50"]
                    / instrumented_summary["p50"],
                }
                for name in COMPONENTS
            },
            "remainder": {
                "definition": "whole - text_backbone - visual_backbone - encoder - decoder",
                "includes": [
                    "input projections",
                    "encoder proposal generation",
                    "topk",
                    "prediction heads",
                    "other work outside the four measured modules",
                ],
                "latency_ms": _summary(remainder_samples),
                "share_of_whole": _summary(share_samples["remainder"]),
                "p50_ms_over_whole_p50": _summary(remainder_samples, include_samples=False)["p50"]
                / instrumented_summary["p50"],
            },
        },
        "validation": {
            "frozen_oracle_gate": frozen_oracle_gate,
            "instrumentation_output_gate": instrumentation_output_gate,
            "transformers_version_exact": True,
            "module_types_exact": True,
            "module_identities_unique": True,
            "module_paths_non_nested": True,
            "dynamic_regions_non_overlapping": True,
            "execution_order_stable_across_iterations": all(
                sequence == sequences[0] for sequence in sequences
            ),
            "observed_component_order": [
                item.removeprefix("begin:")
                for item in sequences[0]
                if item.startswith("begin:")
            ],
            "expected_call_count_per_iteration": 1,
            "call_counts": call_counts,
            "event_sequences": sequences,
        },
        "memory": {
            "allocated_after_warmup_bytes": allocated_before,
            "reserved_after_warmup_bytes": reserved_before,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_incremental_allocated_bytes": max(0, peak_allocated - allocated_before),
            "peak_incremental_reserved_bytes": max(0, peak_reserved - reserved_before),
        },
        "code": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "limitations": [
            (
                "CUDA event regions measure work ordered on the hook's current stream; "
                "independently launched streams may not be fully attributed."
            ),
            (
                "Remainder is subtraction, not a separately instrumented module, and "
                "includes dispatch gaps between CUDA events."
            ),
            "This diagnostic profile does not establish correctness or backend equivalence.",
        ],
    }
    if not result["validation"]["execution_order_stable_across_iterations"]:
        raise RuntimeError(f"component execution order changed across runs: {sequences}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

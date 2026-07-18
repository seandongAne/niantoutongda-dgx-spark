#!/usr/bin/env python
"""Locate the first stage-level Grounding DINO drift under CUDA autocast."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from scripts.gdino_torch_precision_bench import (
    INPUT_NAMES,
    _decision_diff,
    _postprocess,
    _raw_diff,
)


def _iter_tensors(value, prefix: str = ""):
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}.{index}" if prefix else str(index))


def _snapshot(tensor, sample_size: int) -> dict:
    import torch

    detached = tensor.detach()
    flat = detached.reshape(-1)
    count = int(flat.numel())
    if count:
        if count <= sample_size:
            sample = flat
        else:
            indices = torch.linspace(0, count - 1, steps=sample_size, device=flat.device).long()
            sample = flat.index_select(0, indices)
        sample = sample.float().cpu().numpy()
    else:
        sample = np.empty((0,), dtype=np.float32)
    finite = torch.isfinite(detached)
    finite_values = detached[finite].float()
    return {
        "shape": tuple(int(value) for value in detached.shape),
        "dtype": str(detached.dtype),
        "numel": count,
        "nan_count": int(torch.isnan(detached).sum()),
        "posinf_count": int(torch.isposinf(detached).sum()),
        "neginf_count": int(torch.isneginf(detached).sum()),
        "finite_mean": float(finite_values.mean()) if finite_values.numel() else None,
        "finite_std": float(finite_values.std(unbiased=False)) if finite_values.numel() else None,
        "sample": sample,
    }


def _compare_snapshot(reference: dict, candidate: dict) -> dict:
    shape_equal = reference["shape"] == candidate["shape"]
    count_pattern_equal = all(
        reference[key] == candidate[key]
        for key in ("nan_count", "posinf_count", "neginf_count")
    )
    comparable_samples = shape_equal and len(reference["sample"]) == len(candidate["sample"])
    sample_pattern_equal = bool(
        comparable_samples
        and np.array_equal(np.isnan(reference["sample"]), np.isnan(candidate["sample"]))
        and np.array_equal(np.isposinf(reference["sample"]), np.isposinf(candidate["sample"]))
        and np.array_equal(np.isneginf(reference["sample"]), np.isneginf(candidate["sample"]))
    )
    jointly_finite = (
        np.isfinite(reference["sample"]) & np.isfinite(candidate["sample"])
        if comparable_samples
        else np.zeros((0,), dtype=bool)
    )
    delta = np.abs(
        reference["sample"][jointly_finite] - candidate["sample"][jointly_finite]
    )
    return {
        "reference_shape": list(reference["shape"]),
        "candidate_shape": list(candidate["shape"]),
        "shape_equal": shape_equal,
        "reference_dtype": reference["dtype"],
        "candidate_dtype": candidate["dtype"],
        "nonfinite_counts_equal": count_pattern_equal,
        "sample_nonfinite_pattern_equal": sample_pattern_equal,
        "sample_count": int(delta.size),
        "sample_max_abs": float(delta.max()) if delta.size else None,
        "sample_mean_abs": float(delta.mean()) if delta.size else None,
        "reference_finite_mean": reference["finite_mean"],
        "candidate_finite_mean": candidate["finite_mean"],
        "reference_finite_std": reference["finite_std"],
        "candidate_finite_std": candidate["finite_std"],
    }


def _selected_module(name: str, module) -> bool:
    exact = {
        "model.text_projection",
        "model.enc_output",
        "model.enc_output_norm",
        "model.encoder_output_bbox_embed",
        "model.encoder_output_class_embed",
    }
    prefixes = (
        "model.backbone.conv_encoder.model.swin.encoder.layers.",
        "model.input_proj_vision.",
        "model.text_backbone.encoder.layer.",
        "model.encoder.layers.",
        "model.decoder.layers.",
    )
    if name in exact:
        return True
    for prefix in prefixes:
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return True
    return name == ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("sample-size must be positive")

    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    frozen = np.load(args.inputs)
    positional_inputs = tuple(torch.from_numpy(frozen[name]).to("cuda") for name in INPUT_NAMES)
    manifest = json.loads(Path(args.baseline_manifest).read_text())
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_dir).cuda().eval()

    snapshots: dict[str, list[dict]] = {}
    execution_order: list[str] = []
    handles = []

    def hook(name):
        label = name or "<root>"

        def capture(_module, _inputs, output):
            rows = [
                {"path": path, **_snapshot(tensor, args.sample_size)}
                for path, tensor in _iter_tensors(output)
            ]
            call_index = sum(key.startswith(label + "#") for key in snapshots)
            key = f"{label}#{call_index}"
            snapshots[key] = rows
            execution_order.append(key)

        return capture

    for name, module in model.named_modules():
        if _selected_module(name, module):
            handles.append(module.register_forward_hook(hook(name)))

    def run(dtype):
        nonlocal snapshots, execution_order
        snapshots = {}
        execution_order = []
        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if dtype is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), context:
            outputs = model(
                pixel_values=positional_inputs[0],
                input_ids=positional_inputs[1],
                token_type_ids=positional_inputs[2],
                attention_mask=positional_inputs[3],
                pixel_mask=positional_inputs[4],
                return_dict=True,
            )
        arrays = {
            "logits": outputs.logits.detach().float().cpu().numpy(),
            "pred_boxes": outputs.pred_boxes.detach().float().cpu().numpy(),
        }
        return arrays, snapshots, execution_order

    reference_outputs, reference_snapshots, reference_order = run(None)
    reference_decisions = _postprocess(
        processor,
        torch,
        reference_outputs,
        torch.from_numpy(frozen["input_ids"]),
        manifest["target_sizes"],
        args.threshold,
        args.text_threshold,
    )
    modes = {}
    for name, dtype in (("amp_fp16", torch.float16), ("amp_bf16", torch.bfloat16)):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            modes[name] = {"supported": False}
            continue
        outputs, candidate_snapshots, candidate_order = run(dtype)
        stage_rows = []
        for index, key in enumerate(reference_order):
            candidate_rows = candidate_snapshots.get(key, [])
            for tensor_index, reference in enumerate(reference_snapshots[key]):
                candidate = (
                    candidate_rows[tensor_index]
                    if tensor_index < len(candidate_rows)
                    else None
                )
                comparison = (
                    _compare_snapshot(reference, candidate)
                    if candidate is not None
                    else {
                        "shape_equal": False,
                        "nonfinite_counts_equal": False,
                        "sample_nonfinite_pattern_equal": False,
                        "sample_max_abs": None,
                    }
                )
                stage_rows.append(
                    {
                        "execution_index": index,
                        "module_call": key,
                        "tensor_path": reference["path"],
                        **comparison,
                    }
                )
        material = [
            row
            for row in stage_rows
            if not row["shape_equal"]
            or not row["nonfinite_counts_equal"]
            or not row["sample_nonfinite_pattern_equal"]
            or (row["sample_max_abs"] is not None and row["sample_max_abs"] > 1e-3)
        ]
        large = [
            row
            for row in stage_rows
            if row["sample_max_abs"] is not None and row["sample_max_abs"] > 0.1
        ]
        decisions = _postprocess(
            processor,
            torch,
            outputs,
            torch.from_numpy(frozen["input_ids"]),
            manifest["target_sizes"],
            args.threshold,
            args.text_threshold,
        )
        modes[name] = {
            "supported": True,
            "execution_order_equal": candidate_order == reference_order,
            "first_material_drift": material[0] if material else None,
            "first_large_drift": large[0] if large else None,
            "stage_diffs": stage_rows,
            "raw_output_diff": _raw_diff(reference_outputs, outputs),
            "decision_diff": _decision_diff(reference_decisions, decisions),
        }

    for handle in handles:
        handle.remove()
    result = {
        "schema_version": "1.0",
        "scope": "SF1_L2_PYTORCH_AUTOCAST_STAGE_DRIFT",
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or manifest.get("code_commit", "unknown"),
        "sample_size_per_tensor": args.sample_size,
        "module_calls": len(reference_order),
        "reference_execution_order": reference_order,
        "modes": modes,
        "interpretation": (
            "stage samples localize the earliest observable divergence; they do not by "
            "themselves prove a single operator is causal"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                mode: {
                    "first_material_drift": value.get("first_material_drift"),
                    "first_large_drift": value.get("first_large_drift"),
                }
                for mode, value in modes.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

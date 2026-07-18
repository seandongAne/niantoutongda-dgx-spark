#!/usr/bin/env python
"""Selective BF16 autocast bench for Grounding DINO.

Runs the visual backbone (and optionally the deformable encoder) under CUDA
BF16 autocast while keeping the decoder, contrastive head, and everything else
in FP32.  Motivated by the DAY-07 margin analysis: full-model compile BF16 lost
exactly one detection that sits 0.010 above the 0.22 box threshold, so the
question is whether confining BF16 to the ~90% of runtime spent in
backbone+encoder keeps detection decisions inside the frozen gates.

Precision policy is sealed explicitly (matmul TF32 off, cuDNN TF32 off,
float32 matmul precision "highest") and recorded in the manifest; the frozen
old-default oracle is compared as diagnostics only.

This is a two-image diagnostic bench, not hero acceptance evidence.
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

try:
    from gdino_torch_precision_bench import (
        INPUT_NAMES,
        _decision_diff,
        _postprocess,
        _raw_diff,
        _raw_outputs_have_valid_nonfinite_patterns,
    )
    from gdino_capture_decision_compare import _match_image_decisions
except ModuleNotFoundError:  # Supports ``python -m scripts...`` in helper tests.
    from scripts.gdino_torch_precision_bench import (
        INPUT_NAMES,
        _decision_diff,
        _postprocess,
        _raw_diff,
        _raw_outputs_have_valid_nonfinite_patterns,
    )
    from scripts.gdino_capture_decision_compare import _match_image_decisions

REQUIRED_TRANSFORMERS_VERSION = "5.13.1"
SEALED_PRECISION_POLICY = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
}
MARGIN_WINDOW = 0.07
MARGIN_MAX_ROWS = 16
DEFAULT_TRANSITION_BUDGET = 1_000_000


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


def _cast_floating_to_fp32(torch, value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.dtype != torch.float32:
            return value.float()
        return value
    if isinstance(value, tuple):
        items = [_cast_floating_to_fp32(torch, item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items)
    if isinstance(value, list):
        return [_cast_floating_to_fp32(torch, item) for item in value]
    if isinstance(value, dict):
        for key in list(value.keys()):
            value[key] = _cast_floating_to_fp32(torch, value[key])
        return value
    return value


class _RegionAutocast:
    """Enter BF16 autocast on selected submodules only; recast outputs to FP32.

    Forward pre-hooks open ``torch.autocast`` just before the region runs and
    forward hooks close it immediately after, then cast every floating tensor
    in the region output back to FP32 so downstream modules genuinely run in
    FP32 with FP32 operands.  Regions were shown non-nested and single-call by
    gdino_module_profile.py, and both invariants are re-asserted here.
    """

    def __init__(self, torch, regions: dict, dtype):
        self._torch = torch
        self._regions = regions
        self._dtype = dtype
        self._handles = []
        self._open = {}
        self.invocations = {name: 0 for name in regions}

    def __enter__(self):
        for name, module in self._regions.items():
            self._handles.append(
                module.register_forward_pre_hook(self._make_pre(name))
            )
            self._handles.append(module.register_forward_hook(self._make_post(name)))
        return self

    def _make_pre(self, name):
        def hook(_module, _inputs):
            if self._open:
                raise RuntimeError(
                    f"autocast region {name!r} opened while {sorted(self._open)} still open"
                )
            context = self._torch.autocast(device_type="cuda", dtype=self._dtype)
            context.__enter__()
            self._open[name] = context
            self.invocations[name] += 1
            return None

        return hook

    def _make_post(self, name):
        def hook(_module, _inputs, output):
            context = self._open.pop(name, None)
            if context is None:
                raise RuntimeError(f"autocast region {name!r} closed without opening")
            context.__exit__(None, None, None)
            return _cast_floating_to_fp32(self._torch, output)

        return hook

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        for name, context in list(self._open.items()):
            context.__exit__(None, None, None)
        dangling = sorted(self._open)
        self._open = {}
        if exc_type is None and dangling:
            raise RuntimeError(f"autocast regions never closed: {dangling}")
        return False


def _benchmark(torch, wrapper, inputs, *, warmup: int, runs: int):
    ordered = [inputs[name] for name in INPUT_NAMES]
    with torch.inference_mode():
        for _ in range(warmup):
            wrapper(*ordered)
        torch.cuda.synchronize()
        timings_ms = []
        outputs = None
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = wrapper(*ordered)
            end.record()
            end.synchronize()
            timings_ms.append(float(start.elapsed_time(end)))
    arrays = {
        "logits": outputs[0].detach().float().cpu().numpy(),
        "pred_boxes": outputs[1].detach().float().cpu().numpy(),
    }
    stats = {
        "warmup": warmup,
        "runs": runs,
        "mean_ms": statistics.fmean(timings_ms),
        "p50_ms": _percentile(timings_ms, 0.50),
        "p95_ms": _percentile(timings_ms, 0.95),
    }
    return arrays, stats


def _detection_items(decisions):
    frames = []
    for frame in decisions:
        frames.append(
            [
                {
                    "label": item["label"],
                    "score": item["score"],
                    "box_xyxy_px": list(item["box"]),
                }
                for item in frame
            ]
        )
    return frames


def _set_gates(reference_frames, candidate_frames, transition_budget: int):
    remaining = transition_budget
    per_batch = []
    for batch_index, (reference, candidate) in enumerate(
        zip(reference_frames, candidate_frames)
    ):
        row = _match_image_decisions(
            reference, candidate, transition_budget=remaining
        )
        remaining -= int(row["matching"]["estimated_transition_upper_bound"])
        per_batch.append(
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
        "method": "exact_label_partitioned_maximum_total_iou_bitmask_dp",
        "strict_pass": all(row["gates"]["strict"]["pass"] for row in per_batch),
        "diagnostic_pass": all(
            row["gates"]["diagnostic"]["pass"] for row in per_batch
        ),
        "per_batch": per_batch,
    }


def _margin_report(reference_logits, candidate_logits, threshold: float):
    def scores(logits):
        finite = np.where(np.isfinite(logits), logits, -np.inf)
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp(-finite.max(axis=-1)))

    reference_scores = scores(reference_logits)
    candidate_scores = scores(candidate_logits)
    report = []
    for batch_index in range(reference_scores.shape[0]):
        keep = np.where(
            (reference_scores[batch_index] >= threshold - MARGIN_WINDOW)
            | (candidate_scores[batch_index] >= threshold - MARGIN_WINDOW)
        )[0]
        rows = sorted(
            (
                {
                    "query": int(query),
                    "reference_score": float(reference_scores[batch_index, query]),
                    "candidate_score": float(candidate_scores[batch_index, query]),
                    "reference_above_threshold": bool(
                        reference_scores[batch_index, query] >= threshold
                    ),
                    "candidate_above_threshold": bool(
                        candidate_scores[batch_index, query] >= threshold
                    ),
                }
                for query in keep
            ),
            key=lambda row: row["reference_score"],
            reverse=True,
        )
        report.append(
            {
                "batch_index": batch_index,
                "threshold": threshold,
                "window": MARGIN_WINDOW,
                "flips_reference_to_below": [
                    row
                    for row in rows
                    if row["reference_above_threshold"]
                    and not row["candidate_above_threshold"]
                ],
                "rows": rows[:MARGIN_MAX_ROWS],
            }
        )
    return {
        "interpretation_boundary": (
            "Per-query rows compare identical query slots; encoder TopK slot "
            "permutation can move a proposal between slots, so slot rows are "
            "diagnostics while the set gates carry the decision comparison."
        ),
        "per_batch": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-outputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--transition-budget", type=int, default=DEFAULT_TRANSITION_BUDGET
    )
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("runs must be positive and warmup non-negative")

    output_path = Path(args.output)
    outputs_npz_path = output_path.with_suffix(".outputs.npz")
    for existing in (output_path, outputs_npz_path):
        if existing.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {existing}")

    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "transformers version drifted: "
            f"{transformers.__version__} != {REQUIRED_TRANSFORMERS_VERSION}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this device")

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = SEALED_PRECISION_POLICY[
        "cuda_matmul_allow_tf32"
    ]
    torch.backends.cudnn.allow_tf32 = SEALED_PRECISION_POLICY["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(
        SEALED_PRECISION_POLICY["float32_matmul_precision"]
    )

    inputs_path = Path(args.inputs)
    baseline_manifest_path = Path(args.baseline_manifest)
    baseline_outputs_path = Path(args.baseline_outputs)
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    with np.load(inputs_path) as frozen:
        missing = [name for name in INPUT_NAMES if name not in frozen.files]
        if missing:
            raise RuntimeError(f"frozen inputs missing arrays: {missing}")
        inputs = {
            name: torch.from_numpy(np.array(frozen[name])).to("cuda")
            for name in INPUT_NAMES
        }
    with np.load(baseline_outputs_path) as frozen_outputs:
        baseline_outputs = {
            name: np.array(frozen_outputs[name]) for name in ("logits", "pred_boxes")
        }

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(args.model_dir)
        .to("cuda")
        .eval()
    )

    core = model.model
    regions = {
        "visual_backbone": core.backbone,
        "encoder": core.encoder,
    }
    region_validation = {
        "visual_backbone": type(core.backbone).__name__,
        "encoder": type(core.encoder).__name__,
        "decoder_fp32": type(core.decoder).__name__,
        "text_backbone_fp32": type(core.text_backbone).__name__,
    }

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

    wrapper = Wrapper(model)
    target_sizes = baseline_manifest["target_sizes"]
    input_ids_cpu = inputs["input_ids"].detach().cpu()

    def decisions_for(arrays):
        return _postprocess(
            processor,
            torch,
            arrays,
            input_ids_cpu,
            target_sizes,
            args.threshold,
            args.text_threshold,
        )

    mode_specs = [
        ("fp32", ()),
        ("selective_bf16_backbone", ("visual_backbone",)),
        ("selective_bf16_encoder", ("encoder",)),
        ("selective_bf16_backbone_encoder", ("visual_backbone", "encoder")),
        ("fp32_repeat", ()),
    ]

    saved_arrays = {}
    mode_reports = {}
    reference_arrays = None
    reference_decisions = None
    reference_frames = None
    reference_p50 = None

    for mode_name, region_names in mode_specs:
        selected = {name: regions[name] for name in region_names}
        context = (
            _RegionAutocast(torch, selected, torch.bfloat16)
            if selected
            else contextlib.nullcontext()
        )
        with context as active:
            arrays, stats = _benchmark(
                torch, wrapper, inputs, warmup=args.warmup, runs=args.runs
            )
        if selected:
            expected = args.warmup + args.runs
            bad = {
                name: count
                for name, count in active.invocations.items()
                if count != expected
            }
            if bad:
                raise RuntimeError(
                    f"unexpected region invocation counts (expected {expected}): {bad}"
                )
        saved_arrays[f"{mode_name}_logits"] = arrays["logits"]
        saved_arrays[f"{mode_name}_pred_boxes"] = arrays["pred_boxes"]
        decisions = decisions_for(arrays)
        report = {
            "autocast_bf16_regions": list(region_names),
            "latency": stats,
            "detections": decisions,
        }

        if mode_name == "fp32":
            reference_arrays = arrays
            reference_decisions = decisions
            reference_frames = _detection_items(decisions)
            reference_p50 = stats["p50_ms"]
            frozen_raw = _raw_diff(baseline_outputs, arrays)
            report["vs_frozen_old_default_oracle"] = {
                "raw_diff": frozen_raw,
                "expected_nonbitexact_due_to_sealed_policy_change": True,
                "positional_decision_diff": _decision_diff(
                    decisions_for(baseline_outputs), decisions
                ),
                "set_gates": _set_gates(
                    _detection_items(decisions_for(baseline_outputs)),
                    _detection_items(decisions),
                    args.transition_budget,
                ),
            }
        else:
            raw = _raw_diff(reference_arrays, arrays)
            report["raw_diff_vs_fp32"] = raw
            report["raw_safety_pass"] = _raw_outputs_have_valid_nonfinite_patterns(raw)
            report["positional_decision_diff"] = _decision_diff(
                reference_decisions, decisions
            )
            report["set_gates"] = _set_gates(
                reference_frames,
                _detection_items(decisions),
                args.transition_budget,
            )
            report["margin_report"] = _margin_report(
                reference_arrays["logits"], arrays["logits"], args.threshold
            )
            report["speedup_vs_fp32_p50"] = reference_p50 / stats["p50_ms"]
            if mode_name == "fp32_repeat":
                report["bit_exact_vs_fp32"] = bool(
                    np.array_equal(reference_arrays["logits"], arrays["logits"])
                    and np.array_equal(
                        reference_arrays["pred_boxes"], arrays["pred_boxes"]
                    )
                )
        mode_reports[mode_name] = report

    candidates = (
        "selective_bf16_backbone",
        "selective_bf16_encoder",
        "selective_bf16_backbone_encoder",
    )
    candidate_summary = {}
    for name in candidates:
        report = mode_reports[name]
        candidate_summary[name] = {
            "raw_safety_pass": report["raw_safety_pass"],
            "positional_strict_pass": report["positional_decision_diff"][
                "strict_decision_equivalent_at_1e-3_and_half_px"
            ],
            "set_strict_pass": report["set_gates"]["strict_pass"],
            "set_diagnostic_pass": report["set_gates"]["diagnostic_pass"],
            "detection_counts": report["positional_decision_diff"][
                "candidate_detection_counts"
            ],
            "speedup_vs_fp32_p50": report["speedup_vs_fp32_p50"],
        }
    stability_ok = mode_reports["fp32_repeat"].get("bit_exact_vs_fp32", False)
    strict_winners = [
        name
        for name in candidates
        if candidate_summary[name]["raw_safety_pass"]
        and candidate_summary[name]["set_strict_pass"]
    ]
    diagnostic_winners = [
        name
        for name in candidates
        if candidate_summary[name]["raw_safety_pass"]
        and candidate_summary[name]["set_diagnostic_pass"]
    ]
    if not stability_ok:
        verdict = "NO_GO_FP32_UNSTABLE_IN_PROCESS"
    elif strict_winners:
        verdict = "SET_STRICT_PASS:" + ",".join(strict_winners)
    elif diagnostic_winners:
        verdict = "SET_DIAGNOSTIC_ONLY_PASS:" + ",".join(diagnostic_winners)
    else:
        verdict = "NO_GO_ALL_SELECTIVE_CANDIDATES_FAIL_SET_GATES"

    result = {
        "schema_version": "1.0",
        "scope": "SF1_DIAGNOSTIC_SELECTIVE_AUTOCAST_NOT_HERO_ACCEPTANCE",
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or baseline_manifest.get("code_commit", "unknown"),
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__,
        },
        "precision_policy": {
            "sealed": True,
            **SEALED_PRECISION_POLICY,
            "note": (
                "Sealed double-false policy differs from the legacy frozen oracle "
                "(cudnn TF32 true); frozen-oracle comparisons are diagnostics only."
            ),
        },
        "inputs": {
            "frozen_inputs": {
                "path": str(inputs_path),
                "sha256": _sha256(inputs_path),
            },
            "baseline_manifest": {
                "path": str(baseline_manifest_path),
                "sha256": _sha256(baseline_manifest_path),
            },
            "baseline_outputs": {
                "path": str(baseline_outputs_path),
                "sha256": _sha256(baseline_outputs_path),
            },
            "model_dir": args.model_dir,
        },
        "frozen_workload": {
            "batch_size": int(baseline_manifest["batch_size"]),
            "prompt": baseline_manifest.get("prompt"),
            "target_sizes": target_sizes,
            "box_threshold": args.threshold,
            "text_threshold": args.text_threshold,
        },
        "region_validation": region_validation,
        "modes": mode_reports,
        "candidate_summary": candidate_summary,
        "fp32_repeat_bit_exact": stability_ok,
        "verdict": verdict,
        "acceptance_boundary": (
            "Two frozen images and one prompt; diagnostic evidence only. A "
            "candidate passing here still requires a larger untouched frozen "
            "detection set and downstream hero metrics before mainline adoption."
        ),
        "exit_code_semantics": "0 = at least one selective candidate passes the strict set gate; 2 otherwise",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with outputs_npz_path.open("xb") as handle:
        np.savez_compressed(handle, **saved_arrays)
    with output_path.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"verdict": verdict, "candidates": candidate_summary}, ensure_ascii=False, indent=2))
    return 0 if strict_winners and stability_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

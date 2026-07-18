#!/usr/bin/env python
"""Measure the current HF Grounding DINO core and export it to ONNX.

This is an SF1-L2 feasibility probe, not a main-chain runtime switch.  It uses
the same processor semantics as ``GroundingDinoDetector._detect_view_batch``:
real RGB frames, one four-phrase prompt batch, padding=True, and model outputs
``logits`` plus ``pred_boxes``.  The manifest and tensors let a later TensorRT
step benchmark exactly the same inputs and audit numerical drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt",
        default="luggage. mini fridge. water bottle. desk.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--exporter", choices=("dynamo", "legacy"), default="dynamo")
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.runs < 1:
        parser.error("batch-size/runs must be positive and warmup non-negative")

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Spark TensorRT baseline")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    image_paths = [Path(path) for path in args.image]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    selected_paths = [
        image_paths[index % len(image_paths)] for index in range(args.batch_size)
    ]
    images = [Image.open(path).convert("RGB") for path in selected_paths]
    target_sizes = [image.size[::-1] for image in images]

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).cuda().eval()
    batch = processor(
        images=images,
        text=[args.prompt] * args.batch_size,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    if "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

    input_names = [
        "pixel_values",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "pixel_mask",
    ]
    missing = [name for name in input_names if name not in batch]
    if missing:
        raise RuntimeError(f"processor did not provide required inputs: {missing}")

    class ExportWrapper(torch.nn.Module):
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

    wrapper = ExportWrapper(model).cuda().eval()
    positional_inputs = tuple(batch[name] for name in input_names)

    with torch.inference_mode():
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
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        process_peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    assert outputs is not None

    input_np = {name: batch[name].detach().cpu().numpy() for name in input_names}
    np.savez_compressed(output_dir / "sample_inputs.npz", **input_np)
    np.savez_compressed(
        output_dir / "torch_outputs.npz",
        logits=outputs[0].detach().float().cpu().numpy(),
        pred_boxes=outputs[1].detach().float().cpu().numpy(),
    )

    project = Path(__file__).resolve().parent.parent
    baseline_manifest = {
        "schema_version": "1.0",
        "scope": "SF1-L2_TENSORRT_FEASIBILITY_ONLY",
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or _git_commit(project),
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "model_dir": str(model_dir),
        "images": [str(path) for path in selected_paths],
        "prompt": args.prompt,
        "batch_size": args.batch_size,
        "inputs": {
            name: {"shape": list(batch[name].shape), "dtype": str(batch[name].dtype)}
            for name in input_names
        },
        "outputs": {
            "logits": list(outputs[0].shape),
            "pred_boxes": list(outputs[1].shape),
        },
        "target_sizes": target_sizes,
        "pytorch_core": {
            "warmup": args.warmup,
            "runs": args.runs,
            "mean_ms": statistics.fmean(timings_ms),
            "p50_ms": _percentile(timings_ms, 0.50),
            "p95_ms": _percentile(timings_ms, 0.95),
            "throughput_images_per_second": args.batch_size
            / (statistics.fmean(timings_ms) / 1000.0),
            "peak_memory_bytes": peak_memory_bytes,
            "process_peak_rss_bytes": process_peak_rss_bytes,
            "samples_ms": timings_ms,
        },
    }
    (output_dir / "baseline_manifest.json").write_text(
        json.dumps(baseline_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    onnx_path = output_dir / "grounding_dino.onnx"
    dynamic_axes = {
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "input_ids": {0: "batch", 1: "text_length"},
        "token_type_ids": {0: "batch", 1: "text_length"},
        "attention_mask": {0: "batch", 1: "text_length"},
        "pixel_mask": {0: "batch", 1: "height", 2: "width"},
        "logits": {0: "batch", 2: "text_length"},
        "pred_boxes": {0: "batch"},
    }
    export_started = time.perf_counter()
    try:
        export_options = {
            "input_names": input_names,
            "output_names": ["logits", "pred_boxes"],
            "opset_version": args.opset,
            "do_constant_folding": True,
            "external_data": False,
            "dynamo": args.exporter == "dynamo",
        }
        if args.exporter == "legacy":
            export_options["dynamic_axes"] = dynamic_axes
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                positional_inputs,
                onnx_path,
                **export_options,
            )
    except Exception as exc:
        failure = {
            "schema_version": "1.0",
            "scope": "SF1-L2_TENSORRT_FEASIBILITY_ONLY",
            "exporter": args.exporter,
            "opset": args.opset,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - export_started,
        }
        (output_dir / "export_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        raise
    export_seconds = time.perf_counter() - export_started

    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(graph)

    manifest = {
        **baseline_manifest,
        "onnx": {
            "path": str(onnx_path),
            "sha256": _sha256(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "opset": args.opset,
            "exporter": args.exporter,
            "dynamic_shapes": args.exporter == "legacy",
            "export_seconds": export_seconds,
            "checker_pass": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

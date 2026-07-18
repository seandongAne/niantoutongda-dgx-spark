#!/usr/bin/env python
"""Expanded-set regression for the encoder-only BF16 candidate.

Adoption precondition #2 of docs/运行时等价判定口径_2026-07-19.md: the revised
IoU>=0.98 runtime-equivalence tier must hold on >=24 hero-domain frames that
never participated in tuning.  Reference is same-process sealed-policy eager
FP32; candidate is the encoder-region BF16 autocast from
gdino_selective_autocast_bench.  Frames are sampled deterministically
(sorted paths, even spacing) — no RNG, no hand-picking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

try:
    from gdino_selective_autocast_bench import (
        SEALED_PRECISION_POLICY,
        REQUIRED_TRANSFORMERS_VERSION,
        _RegionAutocast,
    )
    from gdino_capture_decision_compare import _match_image_decisions
except ModuleNotFoundError:
    from scripts.gdino_selective_autocast_bench import (
        SEALED_PRECISION_POLICY,
        REQUIRED_TRANSFORMERS_VERSION,
        _RegionAutocast,
    )
    from scripts.gdino_capture_decision_compare import _match_image_decisions

REVISED_IOU_MIN = 0.98
REVISED_SCORE_DELTA_MAX = 1e-2
TRANSITION_BUDGET = 1_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_evenly(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[i] for i in sorted(set(int(i) for i in indices))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--code-commit")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames_root = Path(args.frames_root)
    all_frames = sorted(frames_root.rglob("*.jpg"))
    if len(all_frames) < args.max_frames:
        raise SystemExit(
            f"only {len(all_frames)} frames under {frames_root}; need {args.max_frames}"
        )
    frames = _sample_evenly(all_frames, args.max_frames)

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers drifted: {transformers.__version__} != {REQUIRED_TRANSFORMERS_VERSION}"
        )
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = SEALED_PRECISION_POLICY["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = SEALED_PRECISION_POLICY["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(SEALED_PRECISION_POLICY["float32_matmul_precision"])

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(args.model_dir)
        .to("cuda")
        .eval()
    )
    encoder_region = {"encoder": model.model.encoder}

    def detections(pixel_inputs, target_sizes):
        with torch.inference_mode():
            outputs = model(**pixel_inputs)
        processed = processor.post_process_grounded_object_detection(
            outputs,
            pixel_inputs["input_ids"].detach().cpu(),
            threshold=args.threshold,
            text_threshold=args.text_threshold,
            target_sizes=target_sizes,
        )
        frames_out = []
        for row in processed:
            labels = row["text_labels"] if "text_labels" in row else row["labels"]
            frames_out.append(
                [
                    {
                        "label": str(label),
                        "score": float(score),
                        "box_xyxy_px": [float(v) for v in box],
                    }
                    for label, score, box in zip(labels, row["scores"], row["boxes"])
                ]
            )
        return frames_out[0]

    rows = []
    remaining_budget = TRANSITION_BUDGET
    for frame_path in frames:
        image = Image.open(frame_path).convert("RGB")
        target_sizes = [(image.height, image.width)]
        encoded = processor(images=image, text=args.prompt, return_tensors="pt")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        reference = detections(encoded, target_sizes)
        with _RegionAutocast(torch, encoder_region, torch.bfloat16) as active:
            candidate = detections(encoded, target_sizes)
        if active.invocations["encoder"] != 1:
            raise RuntimeError("encoder region did not run exactly once under autocast")
        match = _match_image_decisions(
            reference, candidate, transition_budget=remaining_budget
        )
        remaining_budget -= int(match["matching"]["estimated_transition_upper_bound"])
        complete = not match["unmatched_reference_indices"] and not match[
            "unmatched_candidate_indices"
        ]
        summary = match["delta_summary"]
        min_iou = summary["min_iou"]
        max_score = summary["max_score_abs_delta"]
        revised_pass = bool(
            complete
            and (min_iou is None or min_iou >= REVISED_IOU_MIN)
            and (max_score is None or max_score <= REVISED_SCORE_DELTA_MAX)
        )
        rows.append(
            {
                "frame": str(frame_path),
                "sha256": _sha256(frame_path),
                "image_hw": [image.height, image.width],
                "reference_count": len(reference),
                "candidate_count": len(candidate),
                "complete_one_to_one": complete,
                "min_iou": min_iou,
                "max_score_abs_delta": max_score,
                "max_box_abs_delta_px": summary["max_box_abs_delta_px"],
                "strict_pass": bool(match["gates"]["strict"]["pass"]),
                "diagnostic_pass": bool(match["gates"]["diagnostic"]["pass"]),
                "revised_098_pass": revised_pass,
            }
        )

    failed = [row for row in rows if not row["revised_098_pass"]]
    finite_ious = [row["min_iou"] for row in rows if row["min_iou"] is not None]
    finite_scores = [
        row["max_score_abs_delta"] for row in rows if row["max_score_abs_delta"] is not None
    ]
    verdict = (
        "REVISED_098_PASS_ALL_FRAMES"
        if not failed
        else f"REVISED_098_FAIL_{len(failed)}_OF_{len(rows)}"
    )
    result = {
        "schema_version": "1.0",
        "scope": "SF1_DIAGNOSTIC_ENCODER_BF16_EXPANDED_REGRESSION",
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or "unknown",
        "platform": {
            "machine": platform.machine(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "precision_policy": {"sealed": True, **SEALED_PRECISION_POLICY},
        "candidate": "selective_bf16_encoder_only",
        "gate": {
            "tier": "revised-0.98",
            "contract": "docs/运行时等价判定口径_2026-07-19.md",
            "iou_min_inclusive": REVISED_IOU_MIN,
            "score_abs_delta_max_inclusive": REVISED_SCORE_DELTA_MAX,
            "complete_one_to_one_label_match": True,
        },
        "frames_root": str(frames_root),
        "sampling": "sorted rglob *.jpg, numpy.linspace even indices, no RNG",
        "frame_count": len(rows),
        "zero_detection_frames": sum(1 for row in rows if row["reference_count"] == 0),
        "worst_min_iou": min(finite_ious) if finite_ious else None,
        "worst_max_score_abs_delta": max(finite_scores) if finite_scores else None,
        "pass_counts": {
            "strict": sum(1 for row in rows if row["strict_pass"]),
            "diagnostic": sum(1 for row in rows if row["diagnostic_pass"]),
            "revised_098": len(rows) - len(failed),
        },
        "per_frame": rows,
        "verdict": verdict,
        "acceptance_boundary": (
            "Runtime numerical equivalence regression only (reference = sealed "
            "FP32 in-process); says nothing about detection quality vs GT and "
            "does not by itself switch the production config."
        ),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "pass_counts": result["pass_counts"],
                      "worst_min_iou": result["worst_min_iou"],
                      "worst_max_score_abs_delta": result["worst_max_score_abs_delta"]},
                     ensure_ascii=False, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

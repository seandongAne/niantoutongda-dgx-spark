#!/usr/bin/env python
"""Small Spark-only probe for GDINO frame batching and stationary tiling.

The probe reuses already-sampled v5 keyframes, so it does not mutate an ingest
dataset.  It compares sequential and batched full-frame results, then runs one
tiled frame to verify coordinate remapping/NMS before the one-shot v6 run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))


def _signature(detections):
    return [
        {
            "label": item.label,
            "canonical_id": item.canonical_id,
            "category_id": item.category_id,
            "score": round(item.score, 7),
            "box": [round(value, 3) for value in item.box],
        }
        for item in detections
    ]


def _numeric_diff(sequential, batched):
    structure_equal = len(sequential) == len(batched)
    max_score_delta = 0.0
    max_box_delta = 0.0
    for left_frame, right_frame in zip(sequential, batched):
        structure_equal = structure_equal and len(left_frame) == len(right_frame)
        for left, right in zip(left_frame, right_frame):
            structure_equal = structure_equal and (
                left.label,
                left.canonical_id,
                left.category_id,
                left.raw_label,
            ) == (
                right.label,
                right.canonical_id,
                right.category_id,
                right.raw_label,
            )
            max_score_delta = max(max_score_delta, abs(left.score - right.score))
            max_box_delta = max(
                max_box_delta,
                *(abs(a - b) for a, b in zip(left.box, right.box)),
            )
    return structure_equal, max_score_delta, max_box_delta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--frame-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--tile-box-threshold", type=float, default=0.22)
    args = parser.parse_args()
    if args.frame_count < 1 or args.batch_size < 1:
        parser.error("frame-count and batch-size must be positive")

    from backend.pipeline.detect import GroundingDinoDetector
    from backend.pipeline.vocab import load_vocabulary

    paths = [
        str(path)
        for path in sorted(Path(args.ingest_root).glob("*/keyframes/*.jpg"))[: args.frame_count]
    ]
    if len(paths) < args.frame_count:
        raise FileNotFoundError(
            f"requested {args.frame_count} frames, found {len(paths)} under {args.ingest_root}"
        )
    prompts = load_vocabulary(args.vocab).compile()
    detector = GroundingDinoDetector(
        str(Path.home() / "models" / "IDEA-Research__grounding-dino-base"),
        box_threshold=args.box_threshold,
        tile_box_threshold=args.tile_box_threshold,
        image_batch_size=args.batch_size,
    )

    started = time.perf_counter()
    sequential = [detector.detect(path, prompts) for path in paths]
    sequential_s = time.perf_counter() - started
    started = time.perf_counter()
    batched = detector.detect_many(paths, prompts)
    batched_s = time.perf_counter() - started
    started = time.perf_counter()
    tiled = detector.detect_many([paths[0]], prompts, tiled_image_paths={paths[0]})[0]
    tiled_s = time.perf_counter() - started

    sequential_signature = [_signature(items) for items in sequential]
    batched_signature = [_signature(items) for items in batched]
    structure_equal, max_score_delta, max_box_delta = _numeric_diff(sequential, batched)
    decision_equivalent = structure_equal and max_score_delta <= 1e-3 and max_box_delta <= 0.5
    payload = {
        "frame_count": len(paths),
        "prompt_count": len(prompts),
        "prompt_batch_count": len(prompts.batches),
        "image_batch_size": args.batch_size,
        "sequential_s": round(sequential_s, 3),
        "batched_s": round(batched_s, 3),
        "speedup": round(sequential_s / max(batched_s, 1e-9), 3),
        "exact_dataclass_equal": batched == sequential,
        "rounded_signature_equal": batched_signature == sequential_signature,
        "structure_equal": structure_equal,
        "max_score_delta": max_score_delta,
        "max_box_delta_px": max_box_delta,
        "decision_equivalent_at_1e-3_and_half_px": decision_equivalent,
        "sequential_detection_counts": [len(items) for items in sequential],
        "batched_detection_counts": [len(items) for items in batched],
        "tiled_detection_count": len(tiled),
        "tiled_s": round(tiled_s, 3),
        "paths": paths,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if decision_equivalent else 2


if __name__ == "__main__":
    raise SystemExit(main())

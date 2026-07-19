#!/usr/bin/env python
"""Set-based decision gate between two raw-output NPZ files.

Postprocesses reference and candidate {logits, pred_boxes} through the frozen
AutoProcessor path, then adjudicates the three contract tiers of
docs/运行时等价判定口径_2026-07-19.md (strict / diagnostic / revised-0.98)
with the label-partitioned maximum-total-IoU matcher.  Unlike the positional
gate in gdino_trt_runtime_bench, set matching is invariant to score-order
permutations; FP16-class candidates are expected to fail strict and be judged
by the revised tier.
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
    from gdino_capture_decision_compare import _match_image_decisions
except ModuleNotFoundError:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True, help="frozen sample_inputs.npz")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--reference-npz", required=True)
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--code-commit")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoProcessor

    manifest = json.loads(Path(args.baseline_manifest).read_text())
    frozen = np.load(args.inputs)
    input_ids = torch.from_numpy(frozen["input_ids"])
    processor = AutoProcessor.from_pretrained(args.model_dir)

    def decisions(npz_path):
        arrays = np.load(npz_path)
        class Outputs:
            pass

        outputs = Outputs()
        outputs.logits = torch.from_numpy(np.asarray(arrays["logits"], dtype=np.float32))
        outputs.pred_boxes = torch.from_numpy(
            np.asarray(arrays["pred_boxes"], dtype=np.float32)
        )
        processed = processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=args.threshold,
            text_threshold=args.text_threshold,
            target_sizes=manifest["target_sizes"],
        )
        frames = []
        for row in processed:
            labels = row["text_labels"] if "text_labels" in row else row["labels"]
            frames.append(
                [
                    {
                        "label": str(label),
                        "score": float(score),
                        "box_xyxy_px": [float(v) for v in box],
                    }
                    for label, score, box in zip(labels, row["scores"], row["boxes"])
                ]
            )
        return frames

    reference_frames = decisions(args.reference_npz)
    candidate_frames = decisions(args.candidate_npz)
    if len(reference_frames) != len(candidate_frames):
        raise RuntimeError("frame count mismatch between reference and candidate")

    per_image = []
    remaining_budget = TRANSITION_BUDGET
    for index, (reference, candidate) in enumerate(
        zip(reference_frames, candidate_frames)
    ):
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
        per_image.append(
            {
                "image_index": index,
                "reference_count": len(reference),
                "candidate_count": len(candidate),
                "complete_one_to_one": complete,
                "min_iou": min_iou,
                "max_score_abs_delta": max_score,
                "max_box_abs_delta_px": summary["max_box_abs_delta_px"],
                "strict_pass": bool(match["gates"]["strict"]["pass"]),
                "diagnostic_pass": bool(match["gates"]["diagnostic"]["pass"]),
                "revised_098_pass": bool(
                    complete
                    and (min_iou is None or min_iou >= REVISED_IOU_MIN)
                    and (max_score is None or max_score <= REVISED_SCORE_DELTA_MAX)
                ),
            }
        )

    tiers = {
        tier: all(row[f"{tier}_pass"] for row in per_image)
        for tier in ("strict", "diagnostic", "revised_098")
    }
    verdict = (
        "SET_STRICT_PASS"
        if tiers["strict"]
        else "SET_DIAGNOSTIC_PASS"
        if tiers["diagnostic"]
        else "SET_REVISED_098_PASS"
        if tiers["revised_098"]
        else "SET_ALL_TIERS_FAIL"
    )
    result = {
        "schema_version": "1.0",
        "scope": "SF1_DIAGNOSTIC_NPZ_DECISION_SET_GATE",
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or "unknown",
        "platform": {"machine": platform.machine()},
        "gate_contract": "docs/运行时等价判定口径_2026-07-19.md",
        "reference_npz": {"path": args.reference_npz, "sha256": _sha256(Path(args.reference_npz))},
        "candidate_npz": {"path": args.candidate_npz, "sha256": _sha256(Path(args.candidate_npz))},
        "thresholds": {"box": args.threshold, "text": args.text_threshold},
        "per_image": per_image,
        "tiers_all_images": tiers,
        "verdict": verdict,
        "acceptance_boundary": (
            "Frozen two-image workload only; a revised-0.98 pass here still "
            "requires the contract's expanded-regression and integration "
            "preconditions before any production adoption."
        ),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "tiers": tiers, "per_image": [
        {k: row[k] for k in ("reference_count", "candidate_count", "min_iou", "max_score_abs_delta")}
        for row in per_image
    ]}, ensure_ascii=False, indent=2))
    return 0 if verdict != "SET_ALL_TIERS_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())

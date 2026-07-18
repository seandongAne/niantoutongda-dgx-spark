#!/usr/bin/env python
"""Check a frozen Grounding DINO ONNX graph against PyTorch outputs on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _compare(reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float):
    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    jointly_finite = reference_finite & candidate_finite
    delta = np.abs(reference[jointly_finite] - candidate[jointly_finite])
    nonfinite_pattern_equal = bool(
        np.array_equal(np.isnan(reference), np.isnan(candidate))
        and np.array_equal(np.isposinf(reference), np.isposinf(candidate))
        and np.array_equal(np.isneginf(reference), np.isneginf(candidate))
    )
    finite_values_close = bool(
        np.allclose(
            reference[jointly_finite],
            candidate[jointly_finite],
            atol=atol,
            rtol=rtol,
        )
    )
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "nonfinite_pattern_equal": nonfinite_pattern_equal,
        "jointly_finite_count": int(jointly_finite.sum()),
        "max_abs_on_jointly_finite": float(delta.max()) if delta.size else None,
        "mean_abs_on_jointly_finite": float(delta.mean()) if delta.size else None,
        "finite_values_close": finite_values_close,
        "equivalent": (
            reference.shape == candidate.shape
            and nonfinite_pattern_equal
            and finite_values_close
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--torch-outputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("runs must be positive and warmup non-negative")

    import onnxruntime as ort

    onnx_path = Path(args.onnx)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_started = time.perf_counter()
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    session_seconds = time.perf_counter() - session_started

    frozen_inputs = np.load(args.inputs)
    feed = {item.name: frozen_inputs[item.name] for item in session.get_inputs()}
    output_names = [item.name for item in session.get_outputs()]
    for _ in range(args.warmup):
        session.run(output_names, feed)

    timings_ms = []
    values = None
    for _ in range(args.runs):
        started = time.perf_counter()
        values = session.run(output_names, feed)
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    assert values is not None

    candidate = dict(zip(output_names, values))
    reference = np.load(args.torch_outputs)
    comparisons = {
        name: _compare(reference[name], candidate[name], args.atol, args.rtol)
        for name in ("logits", "pred_boxes")
    }
    result = {
        "schema_version": "1.0",
        "scope": "SF1-L2_ONNX_EXPORT_SEMANTICS_ONLY",
        "onnx": {
            "path": str(onnx_path),
            "sha256": _sha256(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
        },
        "onnxruntime": {
            "version": ort.__version__,
            "providers": session.get_providers(),
            "session_creation_seconds": session_seconds,
            "warmup": args.warmup,
            "runs": args.runs,
            "mean_ms": statistics.fmean(timings_ms),
            "samples_ms": timings_ms,
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            * 1024,
        },
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "comparisons": comparisons,
        "equivalent": all(item["equivalent"] for item in comparisons.values()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["equivalent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

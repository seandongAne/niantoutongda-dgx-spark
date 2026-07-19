#!/usr/bin/env python
"""Mark fusion-splitting outputs on the GDINO ONNX to build a repaired engine.

DAY-07 bisect v3/v4 established that the TRT FP32 no-TF32 divergence in the
contrastive-head region is fusion-dependent: every op is numerically clean
whenever its value is materialized as a graph output, and the dirt always
moves into whatever region stays fused.  This tool therefore marks the union
of the v3/v4 observation points (typed, dtype/shape backfilled from the bisect
result.json records — TRT's parser rejects UNKNOWN-dtype outputs) so the
production engine is forced to split those fusions while keeping the original
`logits`/`pred_boxes` contract intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DTYPE_TO_ONNX = {
    "float32": 1,   # onnx.TensorProto.FLOAT
    "float16": 10,
    "bool": 9,
    "int32": 6,
    "int64": 7,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument(
        "--from-results",
        nargs="+",
        required=True,
        help="bisect result.json files whose ort_vs_trt records supply names+dtypes+shapes",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=["bitwise_not_32"],
        help="marker names to exclude (default: bool mask)",
    )
    parser.add_argument("--output-onnx", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--code-commit")
    args = parser.parse_args()

    output_onnx = Path(args.output_onnx)
    manifest_path = Path(args.output_manifest)
    if output_onnx.exists():
        raise SystemExit(f"refusing to overwrite {output_onnx}")
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {manifest_path}")
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    markers: dict[str, dict] = {}
    for results_file in args.from_results:
        record = json.loads(Path(results_file).read_text())
        for name, rec in record.get("ort_vs_trt", {}).items():
            if name in ("logits", "pred_boxes") or name in args.skip:
                continue
            if not isinstance(rec, dict):
                continue
            dtype = rec.get("candidate_dtype")
            shape = rec.get("candidate_shape")
            if dtype not in DTYPE_TO_ONNX or not shape:
                continue
            markers[name] = {"dtype": dtype, "shape": shape, "source": results_file}
    if not markers:
        raise SystemExit("no typed markers recovered from the given result files")

    import onnx

    model = onnx.load(args.onnx)
    graph = model.graph
    existing_outputs = {value.name for value in graph.output}
    produced = {out for node in graph.node for out in node.output}
    added = []
    for name in sorted(markers):
        if name in existing_outputs:
            continue
        if name not in produced:
            raise SystemExit(f"marker {name!r} is not produced by any node in this graph")
        info = markers[name]
        graph.output.append(
            onnx.helper.make_tensor_value_info(
                name, DTYPE_TO_ONNX[info["dtype"]], info["shape"]
            )
        )
        added.append(name)
    if not added:
        raise SystemExit("all markers were already graph outputs; nothing to do")

    onnx.save_model(model, str(output_onnx), save_as_external_data=False)
    manifest = {
        "schema_version": "1.0",
        "scope": "SF1_DIAGNOSTIC_FUSION_SPLIT_REPAIR_PREP",
        "code_commit": args.code_commit or "unknown",
        "source_onnx": {"path": args.onnx, "sha256": _sha256(Path(args.onnx))},
        "marked_outputs": {name: markers[name] for name in added},
        "skipped": list(args.skip),
        "output_onnx": {"path": str(output_onnx), "sha256": _sha256(output_onnx)},
        "note": (
            "Extra outputs exist only to force TRT fusion splits; consumers must "
            "keep reading logits/pred_boxes and ignore the markers."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"added_markers": added}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

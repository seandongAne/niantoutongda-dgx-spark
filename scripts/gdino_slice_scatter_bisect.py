#!/usr/bin/env python
"""Final-layer bisect for the TensorRT class-logits divergence.

DAY-07 sentinel localization proved TRT FP32 no-TF32 drifts only on the class
logits path: proposal coords, encoder GridSamples, and proposal-ID-aligned
final boxes all match ORT, while the contrastive-head output (`slice_scatter`,
[2,20906,256]) is off by up to 4.8.  Two suspects remain: the contrastive
matmul feeding the scatter, or the slice_scatter assembly itself.

This probe marks the inputs of the node that produces `slice_scatter` as graph
outputs, proves instrumentation equivalence in ORT, builds a diagnostic TRT
engine, and compares each marked tensor ORT-vs-TRT.  Clean node inputs + dirty
node output convicts the scatter; a dirty input moves guilt upstream.

Diagnostic only; instrumented engines carry no timing claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    from gdino_compare_onnx_pytorch_boundaries import _finite_array_diff
    from gdino_trt_runtime_bench import _torch_dtype
except ModuleNotFoundError:
    from scripts.gdino_compare_onnx_pytorch_boundaries import _finite_array_diff
    from scripts.gdino_trt_runtime_bench import _torch_dtype

TARGET_VALUE = "slice_scatter"
CLEAN_MAX_ABS = 1e-3
DIRTY_MAX_ABS = 1e-1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="original (non-instrumented) ONNX")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trtexec", required=True)
    parser.add_argument("--workspace-mib", type=int, default=16384)
    parser.add_argument(
        "--target-value",
        default=TARGET_VALUE,
        help="graph value whose real producer gets bisected (default: slice_scatter)",
    )
    parser.add_argument("--code-commit")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensors_dir = output_path.parent / "tensors"
    tensors_dir.mkdir(exist_ok=True)

    import onnx
    import onnxruntime

    model = onnx.load(args.onnx)
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    producer_of = {}
    for node in graph.node:
        for out in node.output:
            producer_of[out] = node

    PASSTHROUGH_OPS = {
        "Transpose",
        "Reshape",
        "Identity",
        "Cast",
        "Squeeze",
        "Unsqueeze",
        "Flatten",
    }

    def real_producer(value_name):
        """Walk upstream through movement-only ops to the first computing node."""
        chain = []
        current = value_name
        node = producer_of.get(current)
        while node is not None and node.op_type in PASSTHROUGH_OPS:
            chain.append({"op_type": node.op_type, "output": current})
            current = node.input[0]
            if current in initializer_names:
                return None, current, chain
            node = producer_of.get(current)
        return node, current, chain

    target_value = args.target_value
    if target_value not in producer_of:
        raise RuntimeError(f"no node produces value {target_value!r}")

    level1_node, level1_output, level1_chain = real_producer(target_value)
    if level1_node is None:
        raise RuntimeError("walked into an initializer before any computing node")

    marked = [target_value]
    skipped_initializers = []
    node_map = {}

    def mark_node_inputs(node, depth):
        entries = []
        for input_name in node.input:
            if not input_name:
                continue
            if input_name in initializer_names:
                skipped_initializers.append(input_name)
                continue
            if input_name not in marked:
                marked.append(input_name)
            entries.append(input_name)
        node_map[node.name or node.output[0]] = {
            "op_type": node.op_type,
            "depth": depth,
            "output": node.output[0],
            "traced_inputs": entries,
        }
        return entries

    if level1_output not in marked:
        marked.append(level1_output)
    level1_inputs = mark_node_inputs(level1_node, 1)
    for input_name in list(level1_inputs):
        level2_node, level2_output, _ = real_producer(input_name)
        if level2_node is None:
            continue
        if level2_output not in marked:
            marked.append(level2_output)
        mark_node_inputs(level2_node, 2)
    if len(marked) > 16:
        marked = marked[:16]

    target_node = level1_node

    existing_outputs = [value.name for value in graph.output]
    for name in marked:
        if name not in existing_outputs:
            graph.output.append(onnx.helper.make_empty_tensor_value_info(name))

    instrumented_path = output_path.parent / "scatter_bisect_instrumented.onnx"
    if instrumented_path.exists():
        raise SystemExit(f"refusing to overwrite {instrumented_path}")
    onnx.save_model(model, str(instrumented_path), save_as_external_data=False)

    result: dict = {
        "schema_version": "1.0",
        "scope": "SF1_DIAGNOSTIC_SCATTER_BISECT_NOT_ACCEPTANCE",
        "created_at_unix": int(time.time()),
        "code_commit": args.code_commit or "unknown",
        "onnx": {"path": args.onnx, "sha256": _sha256(Path(args.onnx))},
        "target_node": {
            "target_value": target_value,
            "op_type": target_node.op_type,
            "name": target_node.name,
            "inputs": list(target_node.input),
            "outputs": list(target_node.output),
            "passthrough_chain_from_target_value": level1_chain,
            "traced_nodes": node_map,
            "marked_values": marked,
            "skipped_initializer_inputs": skipped_initializers,
        },
        "thresholds": {"clean_max_abs": CLEAN_MAX_ABS, "dirty_max_abs": DIRTY_MAX_ABS},
    }

    frozen = np.load(args.inputs)
    feed = {key: np.asarray(frozen[key]) for key in frozen.files}
    session_options = onnxruntime.SessionOptions()
    request = ["logits", "pred_boxes"] + marked

    original_session = onnxruntime.InferenceSession(
        args.onnx, sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    original_finals = original_session.run(["logits", "pred_boxes"], feed)
    del original_session
    session = onnxruntime.InferenceSession(
        str(instrumented_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_values = dict(zip(request, session.run(request, feed)))
    del session
    equivalence = {
        name: bool(np.array_equal(reference, ort_values[name]))
        for name, reference in zip(("logits", "pred_boxes"), original_finals)
    }
    result["instrumentation_equivalence"] = {
        "bit_exact": equivalence,
        "equivalent": all(equivalence.values()),
    }
    if not all(equivalence.values()):
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        raise RuntimeError("instrumented graph is not bit-exact to the original in ORT")

    # ORT tolerates untyped graph outputs; the TensorRT parser does not
    # ("Unsupported ONNX data type: <UNKNOWN> (0)").  Backfill dtype/shape on the
    # appended outputs from the ORT-produced arrays, then re-save for trtexec.
    for value in graph.output:
        if value.name in marked and value.type.tensor_type.elem_type == 0:
            array = ort_values[value.name]
            typed = onnx.helper.make_tensor_value_info(
                value.name,
                onnx.helper.np_dtype_to_tensor_dtype(array.dtype),
                list(array.shape),
            )
            value.CopyFrom(typed)
    onnx.save_model(model, str(instrumented_path), save_as_external_data=False)
    result["instrumented_onnx"] = {
        "path": str(instrumented_path),
        "sha256": _sha256(instrumented_path),
        "typed_outputs_backfilled": True,
    }

    engine_path = Path(args.engine)
    if engine_path.exists():
        raise SystemExit(f"refusing to overwrite existing engine: {engine_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = engine_path.with_suffix(".trtexec.log")
    command = [
        args.trtexec,
        f"--onnx={instrumented_path}",
        f"--saveEngine={engine_path}",
        "--noTF32",
        "--skipInference",
        f"--memPoolSize=workspace:{args.workspace_mib}M",
    ]
    started = time.time()
    with log_path.open("w") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    result["engine_build"] = {
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": time.time() - started,
        "log_tail": log_path.read_text(errors="replace").splitlines()[-25:]
        if completed.returncode != 0
        else [],
    }
    if completed.returncode != 0 or not engine_path.exists():
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"trtexec build failed rc={completed.returncode}; see {log_path}")

    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result["platform"] = {
        "machine": platform.machine(),
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    logger = trt.Logger(trt.Logger.WARNING)
    with engine_path.open("rb") as handle:
        engine = trt.Runtime(logger).deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {engine_path}")
    context = engine.create_execution_context()

    tensors = {}
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if name not in feed:
                raise KeyError(f"engine input missing from frozen NPZ: {name}")
            dtype = _torch_dtype(trt, torch, engine.get_tensor_dtype(name))
            tensor = torch.as_tensor(feed[name], dtype=dtype, device="cuda").contiguous()
            if not context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"rejected shape for {name}")
            tensors[name] = tensor
        else:
            output_names.append(name)
    missing = [name for name in marked if name not in output_names]
    if missing:
        result["engine_output_names"] = sorted(output_names)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"marked values missing from engine outputs: {missing}")
    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"unresolved shapes: {unresolved}")
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        dtype = _torch_dtype(trt, torch, engine.get_tensor_dtype(name))
        tensors[name] = torch.empty(shape, dtype=dtype, device="cuda")
    for name, tensor in tensors.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"failed to bind {name}")
    stream = torch.cuda.Stream()
    for _ in range(2):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        stream.synchronize()

    comparisons = {}
    for name in marked + ["logits", "pred_boxes"]:
        trt_array = tensors[name].cpu().numpy()
        reference = ort_values[name]
        np.save(tensors_dir / f"{name.replace('/', '_')}-trt.npy", trt_array)
        if np.issubdtype(reference.dtype, np.floating):
            comparisons[name] = _finite_array_diff(reference, trt_array.astype(reference.dtype, copy=False))
        else:
            comparisons[name] = {
                "integer_equal": bool(
                    np.array_equal(reference.astype(np.int64), trt_array.astype(np.int64))
                ),
                "reference_dtype": str(reference.dtype),
                "candidate_dtype": str(trt_array.dtype),
            }
    result["ort_vs_trt"] = comparisons

    def value_max(name):
        return comparisons.get(name, {}).get("max_abs_on_jointly_finite")

    integer_mismatch = [
        name
        for name in marked
        if "integer_equal" in comparisons.get(name, {})
        and not comparisons[name]["integer_equal"]
    ]
    guilty_nodes = []
    for label, info in node_map.items():
        output_max = value_max(info["output"])
        input_maxes = [value_max(n) for n in info["traced_inputs"]]
        float_in = [m for m in input_maxes if m is not None]
        if output_max is None or not float_in:
            continue
        if output_max > DIRTY_MAX_ABS and all(m <= CLEAN_MAX_ABS for m in float_in):
            guilty_nodes.append(
                {
                    "node": label,
                    "op_type": info["op_type"],
                    "depth": info["depth"],
                    "output_max_abs": output_max,
                }
            )
    dirty_values = sorted(
        (name for name in marked if (value_max(name) or 0.0) > DIRTY_MAX_ABS),
        key=lambda name: -(value_max(name) or 0.0),
    )
    result["guilty_nodes"] = guilty_nodes
    if guilty_nodes:
        deepest = sorted(guilty_nodes, key=lambda g: -g["depth"])[0]
        verdict = (
            f"GUILTY_OP:{deepest['op_type']}@{deepest['node']}"
            f"(depth={deepest['depth']},output_max_abs={deepest['output_max_abs']:.6g})"
        )
    elif integer_mismatch:
        verdict = "INDEX_PATH_MISMATCH:" + ",".join(integer_mismatch)
    elif dirty_values:
        verdict = "DIRTY_BEYOND_DEPTH2:" + ",".join(dirty_values)
    else:
        verdict = "NO_DIVERGENCE_REPRODUCED_AT_TRACED_NODES"
    result["verdict"] = verdict
    result["acceptance_boundary"] = (
        "Frozen two-image workload; instrumented engine blocks fusions, so no "
        "timing claim. A guilty verdict here names the first divergent op-site, "
        "not yet a reproduced upstream TensorRT bug report."
    )

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": verdict,
        "level1_op": target_node.op_type,
        "marked_max_abs": {name: value_max(name) for name in marked},
        "guilty_nodes": guilty_nodes,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

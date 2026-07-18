#!/usr/bin/env python
"""Instrument a frozen Grounding DINO ONNX graph at decisive boundaries.

The probe deliberately does *not* expose every ONNX value.  A Grounding DINO
graph is close to one gigabyte and marking every value as an output can make
ONNX Runtime consume tens of gigabytes.  Instead this script only proceeds when
it can prove all of the following from graph topology and exporter metadata:

* exactly one ``TopK`` selects the final query count and its indices feed
  proposal ``GatherElements`` operations on axis 1;
* every standard ``GridSample`` can be mapped to an explicit Grounding DINO
  encoder/decoder layer; and
* every selected value has a static type and shape that fit configured tensor,
  byte, disk, and memory limits.  Large GridSample outputs are exposed only
  through a deterministic, evenly spaced query-axis ``Gather`` sample.

On success, the script writes an instrumented ONNX model and exact ``.npy``
outputs for the frozen input batch.  If any mapping is ambiguous it writes an
``inspection.json`` with verdict ``NO_GO`` and does not create or run an
instrumented model.  This is a graph-semantics diagnostic, not a deployment
runtime or performance benchmark.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


SCOPE = "SF1-L2_ONNX_INTERMEDIATE_SEMANTICS_ONLY"
SCHEMA_VERSION = "1.0"
STANDARD_ONNX_DOMAINS = {"", "ai.onnx"}
INDEX_PASSTHROUGH_OPS = {
    "Cast",
    "Expand",
    "Identity",
    "Reshape",
    "Squeeze",
    "Tile",
    "Unsqueeze",
}
DEFAULT_MAX_TENSORS = 20
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_SINGLE_OUTPUT_BYTES = 96 * 1024 * 1024
DEFAULT_GRID_SAMPLE_QUERY_SAMPLES = 256
DEFAULT_MIN_AVAILABLE_MEMORY_BYTES = 12 * 1024 * 1024 * 1024
DEFAULT_MIN_DISK_RESERVE_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_INSTRUMENTATION_ATOL = 1e-7
DEFAULT_INSTRUMENTATION_RTOL = 1e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed intermediate probe for a frozen Grounding DINO ONNX graph."
    )
    parser.add_argument("--onnx", required=True, help="Existing static ONNX model.")
    parser.add_argument("--inputs", required=True, help="Frozen sample_inputs.npz.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-num-queries",
        type=int,
        help="Optional assertion; the graph's logits output must independently agree.",
    )
    parser.add_argument(
        "--max-grid-sample-probes",
        type=int,
        default=4,
        help="Maximum mapped GridSample sentinels. Four covers both ends of encoder and decoder.",
    )
    parser.add_argument(
        "--grid-sample-query-samples",
        type=int,
        default=DEFAULT_GRID_SAMPLE_QUERY_SAMPLES,
        help=(
            "Evenly spaced query-axis values captured from each GridSample sentinel; "
            "the full output is never exposed."
        ),
    )
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_MAX_TENSORS)
    parser.add_argument(
        "--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES
    )
    parser.add_argument(
        "--max-single-output-bytes",
        type=int,
        default=DEFAULT_MAX_SINGLE_OUTPUT_BYTES,
    )
    parser.add_argument(
        "--min-available-memory-bytes",
        type=int,
        default=DEFAULT_MIN_AVAILABLE_MEMORY_BYTES,
    )
    parser.add_argument(
        "--min-disk-reserve-bytes",
        type=int,
        default=DEFAULT_MIN_DISK_RESERVE_BYTES,
    )
    parser.add_argument(
        "--provider", default="CPUExecutionProvider", help="ONNX Runtime provider."
    )
    parser.add_argument(
        "--ort-optimization-level",
        choices=("disable", "basic", "extended", "all"),
        default="all",
    )
    parser.add_argument(
        "--instrumentation-atol",
        type=float,
        default=DEFAULT_INSTRUMENTATION_ATOL,
        help="Absolute tolerance for original-vs-instrumented final outputs.",
    )
    parser.add_argument(
        "--instrumentation-rtol",
        type=float,
        default=DEFAULT_INSTRUMENTATION_RTOL,
        help="Relative tolerance for original-vs-instrumented final outputs.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Write the mapping plan without saving an ONNX copy or running ORT.",
    )
    args = parser.parse_args()
    positive = {
        "max-grid-sample-probes": args.max_grid_sample_probes,
        "grid-sample-query-samples": args.grid_sample_query_samples,
        "max-tensors": args.max_tensors,
        "max-output-bytes": args.max_output_bytes,
        "max-single-output-bytes": args.max_single_output_bytes,
        "min-available-memory-bytes": args.min_available_memory_bytes,
        "min-disk-reserve-bytes": args.min_disk_reserve_bytes,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        parser.error(f"these values must be positive: {', '.join(invalid)}")
    if args.max_grid_sample_probes < 4:
        parser.error("max-grid-sample-probes must be at least 4")
    if args.expected_num_queries is not None and args.expected_num_queries < 1:
        parser.error("expected-num-queries must be positive")
    if args.instrumentation_atol < 0 or args.instrumentation_rtol < 0:
        parser.error("instrumentation tolerances must be non-negative")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        temporary.chmod(0o644)
        temporary.replace(path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_status(output_dir: Path, state: str, incomplete: bool, **details: Any) -> None:
    _write_json(
        output_dir / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "scope": SCOPE,
            "state": state,
            "incomplete": incomplete,
            "updated_at_unix": int(time.time()),
            **details,
        },
    )


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    reserved = {
        "failure.json",
        "inspection.json",
        "instrumented.onnx",
        "result.json",
        "status.json",
        "tensors",
    }
    conflicts = sorted(item.name for item in path.iterdir() if item.name in reserved)
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite existing probe artifacts: " + ", ".join(conflicts)
        )


def _available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
        return None
    return None


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB while macOS reports bytes.
    return peak if platform.system() == "Darwin" else peak * 1024


def _metadata(node: Any) -> dict[str, str]:
    values = {
        item.key: item.value
        for item in getattr(node, "metadata_props", ())
        if item.key
    }
    if getattr(node, "doc_string", ""):
        values["onnx.doc_string"] = node.doc_string
    return values


def _bounded_metadata(node: Any, limit: int = 800) -> dict[str, str]:
    return {key: value[:limit] for key, value in _metadata(node).items()}


def _node_summary(index: int, node: Any) -> dict[str, Any]:
    return {
        "index": index,
        "name": node.name or None,
        "domain": node.domain or "ai.onnx",
        "op_type": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "metadata": _bounded_metadata(node),
    }


def _value_info_map(model: Any, onnx: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            result[value.name] = value
    for initializer in model.graph.initializer:
        if initializer.name not in result:
            result[initializer.name] = onnx.helper.make_tensor_value_info(
                initializer.name, initializer.data_type, list(initializer.dims)
            )
    return result


def _shape_and_type(value: Any) -> tuple[list[int | str | None], int] | None:
    if value is None or not value.type.HasField("tensor_type"):
        return None
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type == 0 or not tensor_type.HasField("shape"):
        return None
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape, int(tensor_type.elem_type)


def _static_shape(value: Any) -> list[int] | None:
    description = _shape_and_type(value)
    if description is None:
        return None
    shape, _ = description
    if not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape):
        return None
    return [int(dimension) for dimension in shape]


def _dtype_name(elem_type: int, onnx: Any) -> str:
    try:
        return onnx.TensorProto.DataType.Name(elem_type).lower()
    except (AttributeError, ValueError):
        return f"onnx_dtype_{elem_type}"


def _dtype_bytes(elem_type: int, onnx: Any) -> int | None:
    bit_widths = {
        onnx.TensorProto.BOOL: 8,
        onnx.TensorProto.UINT8: 8,
        onnx.TensorProto.INT8: 8,
        onnx.TensorProto.UINT16: 16,
        onnx.TensorProto.INT16: 16,
        onnx.TensorProto.FLOAT16: 16,
        onnx.TensorProto.BFLOAT16: 16,
        onnx.TensorProto.INT32: 32,
        onnx.TensorProto.UINT32: 32,
        onnx.TensorProto.FLOAT: 32,
        onnx.TensorProto.INT64: 64,
        onnx.TensorProto.UINT64: 64,
        onnx.TensorProto.DOUBLE: 64,
        onnx.TensorProto.COMPLEX64: 64,
        onnx.TensorProto.COMPLEX128: 128,
    }
    bits = bit_widths.get(elem_type)
    return None if bits is None else bits // 8


def _tensor_description(name: str, values: dict[str, Any], onnx: Any) -> dict[str, Any]:
    value = values.get(name)
    description = _shape_and_type(value)
    if description is None:
        return {"name": name, "shape": None, "dtype": None, "estimated_bytes": None}
    shape, elem_type = description
    item_bytes = _dtype_bytes(elem_type, onnx)
    estimated_bytes = None
    if item_bytes is not None and all(isinstance(item, int) and item >= 0 for item in shape):
        estimated_bytes = int(math.prod(shape)) * item_bytes
    return {
        "name": name,
        "shape": shape,
        "dtype": _dtype_name(elem_type, onnx),
        "elem_type": elem_type,
        "estimated_bytes": estimated_bytes,
    }


def _attribute_int(node: Any, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _attribute_string(node: Any, name: str, default: str) -> str:
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute.s.decode("utf-8", errors="strict")
    return default


def _constant_array(
    tensor_name: str,
    initializers: dict[str, Any],
    producers: dict[str, tuple[int, Any]],
    numpy_helper: Any,
    np: Any,
    depth: int = 0,
) -> Any | None:
    if depth > 8:
        return None
    if tensor_name in initializers:
        return numpy_helper.to_array(initializers[tensor_name])
    producer_entry = producers.get(tensor_name)
    if producer_entry is None:
        return None
    _, producer = producer_entry
    if producer.op_type == "Constant":
        for attribute in producer.attribute:
            if attribute.name == "value" and attribute.HasField("t"):
                return numpy_helper.to_array(attribute.t)
            if attribute.name == "value_int":
                return np.asarray(attribute.i, dtype=np.int64)
            if attribute.name == "value_ints":
                return np.asarray(attribute.ints, dtype=np.int64)
        return None
    if producer.op_type not in INDEX_PASSTHROUGH_OPS or not producer.input:
        return None
    return _constant_array(
        producer.input[0], initializers, producers, numpy_helper, np, depth + 1
    )


def _constant_ints(
    tensor_name: str,
    initializers: dict[str, Any],
    producers: dict[str, tuple[int, Any]],
    numpy_helper: Any,
    np: Any,
) -> list[int] | None:
    value = _constant_array(
        tensor_name, initializers, producers, numpy_helper, np
    )
    if value is None:
        return None
    flattened = np.asarray(value).reshape(-1)
    if flattened.size == 0 or not np.issubdtype(flattened.dtype, np.integer):
        return None
    return [int(item) for item in flattened.tolist()]


def _reduction_axes(
    node: Any,
    initializers: dict[str, Any],
    producers: dict[str, tuple[int, Any]],
    numpy_helper: Any,
    np: Any,
) -> list[int] | None:
    if len(node.input) >= 2 and node.input[1]:
        return _constant_ints(
            node.input[1], initializers, producers, numpy_helper, np
        )
    for attribute in node.attribute:
        if attribute.name == "axes":
            return [int(item) for item in attribute.ints]
    return None


def _normalize_axis(axis: int, rank: int) -> int | None:
    normalized = axis + rank if axis < 0 else axis
    return normalized if 0 <= normalized < rank else None


def _trace_topk_gathers(
    indices_name: str,
    nodes: list[Any],
    consumers: dict[str, list[tuple[int, int]]],
) -> tuple[list[int], dict[str, Any]]:
    queue: deque[tuple[str, int]] = deque([(indices_name, 0)])
    visited = {indices_name}
    gather_indices: set[int] = set()
    traversed: list[dict[str, Any]] = []
    while queue:
        tensor_name, depth = queue.popleft()
        if depth > 8 or len(visited) > 128:
            return [], {
                "status": "AMBIGUOUS",
                "reason": "index lineage exceeded depth or tensor bound",
                "visited_tensor_count": len(visited),
            }
        for node_index, input_index in consumers.get(tensor_name, []):
            node = nodes[node_index]
            traversed.append(
                {
                    "tensor": tensor_name,
                    "consumer_index": node_index,
                    "consumer_op": node.op_type,
                    "consumer_input_index": input_index,
                }
            )
            if node.op_type == "GatherElements" and input_index == 1:
                gather_indices.add(node_index)
                continue
            if node.op_type in INDEX_PASSTHROUGH_OPS and input_index == 0:
                for output_name in node.output:
                    if output_name and output_name not in visited:
                        visited.add(output_name)
                        queue.append((output_name, depth + 1))
    return sorted(gather_indices), {
        "status": "MAPPED" if gather_indices else "UNMAPPED",
        "visited_tensor_count": len(visited),
        "traversed": traversed[:128],
    }


def _infer_query_count(
    model: Any, values: dict[str, Any], expected_num_queries: int | None
) -> tuple[int | None, list[str]]:
    reasons: list[str] = []
    logits_outputs = [item for item in model.graph.output if item.name == "logits"]
    if len(logits_outputs) != 1:
        reasons.append("graph must have exactly one output named logits")
        return None, reasons
    shape = _static_shape(values.get("logits"))
    if shape is None or len(shape) != 3 or shape[1] < 1:
        reasons.append("logits output must have a static rank-3 shape")
        return None, reasons
    inferred = shape[1]
    if expected_num_queries is not None and expected_num_queries != inferred:
        reasons.append(
            f"expected-num-queries={expected_num_queries} disagrees with "
            f"logits dimension {inferred}"
        )
    return inferred, reasons


def _inspect_topk(
    model: Any,
    values: dict[str, Any],
    expected_num_queries: int | None,
    onnx: Any,
    np: Any,
) -> dict[str, Any]:
    nodes = list(model.graph.node)
    producers = {
        output: (index, node)
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    consumers: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for input_index, input_name in enumerate(node.input):
            if input_name:
                consumers[input_name].append((index, input_index))
    initializers = {item.name: item for item in model.graph.initializer}
    query_count, query_reasons = _infer_query_count(
        model, values, expected_num_queries
    )
    candidates: list[dict[str, Any]] = []
    matching: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        if node.op_type != "TopK":
            continue
        reasons: list[str] = []
        if node.domain not in STANDARD_ONNX_DOMAINS:
            reasons.append(f"non-standard TopK domain {node.domain!r}")
        if len(node.input) < 2 or len(node.output) < 2:
            reasons.append("TopK must have two inputs and two outputs")
            k_values = None
        else:
            k_values = _constant_ints(
                node.input[1], initializers, producers, onnx.numpy_helper, np
            )
        k = k_values[0] if k_values is not None and len(k_values) == 1 else None
        if k is None:
            reasons.append("TopK K input is not a resolvable scalar integer")
        elif query_count is not None and k != query_count:
            reasons.append(f"TopK K={k} does not match query count {query_count}")

        input_description = (
            _tensor_description(node.input[0], values, onnx)
            if node.input
            else {"shape": None}
        )
        value_description = (
            _tensor_description(node.output[0], values, onnx)
            if node.output
            else {"shape": None}
        )
        indices_description = (
            _tensor_description(node.output[1], values, onnx)
            if len(node.output) >= 2
            else {"shape": None}
        )
        input_shape = input_description.get("shape")
        values_shape = value_description.get("shape")
        indices_shape = indices_description.get("shape")
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            reasons.append("TopK score input must have known rank 2")
            normalized_axis = None
        else:
            normalized_axis = _normalize_axis(
                _attribute_int(node, "axis", -1), len(input_shape)
            )
            if normalized_axis != 1:
                reasons.append(f"TopK axis must normalize to 1, got {normalized_axis}")
        if values_shape != indices_shape:
            reasons.append("TopK values and indices shapes differ")
        if (
            isinstance(indices_shape, list)
            and query_count is not None
            and (len(indices_shape) != 2 or indices_shape[1] != query_count)
        ):
            reasons.append("TopK indices shape does not end in the query count")
        if indices_description.get("elem_type") != onnx.TensorProto.INT64:
            reasons.append("TopK indices must be INT64")
        largest = _attribute_int(node, "largest", 1)
        sorted_output = _attribute_int(node, "sorted", 1)
        if largest != 1:
            reasons.append("TopK must select largest values")
        if sorted_output != 1:
            reasons.append("TopK must return sorted values and indices")

        gathers, lineage = (
            _trace_topk_gathers(node.output[1], nodes, consumers)
            if len(node.output) >= 2
            else ([], {"status": "UNMAPPED"})
        )
        valid_gathers: list[int] = []
        for gather_index in gathers:
            gather = nodes[gather_index]
            data_shape = _static_shape(values.get(gather.input[0]))
            output_shape = _static_shape(values.get(gather.output[0]))
            axis = _normalize_axis(
                _attribute_int(gather, "axis", 0), len(data_shape or [])
            )
            if (
                data_shape is not None
                and output_shape is not None
                and len(data_shape) == 3
                and len(output_shape) == 3
                and axis == 1
                and query_count is not None
                and output_shape[1] == query_count
                and data_shape[0] == output_shape[0]
                and data_shape[2] == output_shape[2]
            ):
                valid_gathers.append(gather_index)
        if not valid_gathers:
            reasons.append("TopK indices do not map to a valid axis-1 proposal gather")

        candidate = {
            "node": _node_summary(index, node),
            "k": k,
            "axis": normalized_axis,
            "attributes": {
                "axis_raw": _attribute_int(node, "axis", -1),
                "axis_normalized": normalized_axis,
                "largest": largest,
                "sorted": sorted_output,
            },
            "score_input": input_description,
            "values_output": value_description,
            "indices_output": indices_description,
            "index_lineage": lineage,
            "valid_gather_node_indices": valid_gathers,
            "rejection_reasons": reasons,
            "matches_contract": not reasons,
        }
        candidates.append(candidate)
        if not reasons:
            matching.append(candidate)

    reasons = list(query_reasons)
    if not candidates:
        reasons.append("no TopK node was found in the main graph")
    if len(matching) != 1:
        reasons.append(
            f"expected exactly one query-selection TopK, found {len(matching)}"
        )
    selected = matching[0] if len(matching) == 1 else None
    return {
        "status": "MAPPED" if selected is not None and not reasons else "NO_GO",
        "query_count": query_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "reasons": reasons,
        "producer_map": producers,
        "consumer_map": consumers,
        "initializer_map": initializers,
    }


def _metadata_strings(node: Any) -> list[str]:
    strings: list[str] = []
    for value in _metadata(node).values():
        strings.append(value)
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            strings.extend(str(item) for item in parsed)
        elif isinstance(parsed, dict):
            strings.extend(f"{key}={item}" for key, item in parsed.items())
    return strings


def _grid_sample_scope(node: Any) -> tuple[str, int] | None:
    text = "\n".join(_metadata_strings(node))
    lowered = text.lower()
    if "multiscaledeformableattention" not in lowered and "deformable" not in lowered:
        return None
    matches: set[tuple[str, int]] = set()
    separators = r"(?:\.|/|::|:|_)"
    for stage in ("encoder", "decoder"):
        patterns = (
            rf"\b{stage}{separators}+layers?{separators}+(\d+)\b",
            rf"groundingdino{stage}layer[^\n]{{0,160}}?layers?{separators}+(\d+)\b",
            rf"layers?{separators}+(\d+)\b[^\n]{{0,160}}?groundingdino{stage}layer",
        )
        for pattern in patterns:
            matches.update(
                (stage, int(item))
                for item in re.findall(pattern, lowered, flags=re.IGNORECASE)
            )
    return next(iter(matches)) if len(matches) == 1 else None


def _nested_grid_sample_locations(model: Any, onnx: Any) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []

    def visit_graph(graph: Any, path: str) -> None:
        for index, node in enumerate(graph.node):
            if node.op_type == "GridSample" or (
                "grid" in node.op_type.lower() and "sample" in node.op_type.lower()
            ):
                locations.append(
                    {
                        "location": path,
                        "index": index,
                        "domain": node.domain or "ai.onnx",
                        "op_type": node.op_type,
                        "name": node.name or None,
                    }
                )
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attribute.g, f"{path}/{index}:{attribute.name}")
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph_index, nested in enumerate(attribute.graphs):
                        visit_graph(
                            nested, f"{path}/{index}:{attribute.name}[{graph_index}]"
                        )

    # Do not include the main graph here; callers inspect it with full value info.
    for index, node in enumerate(model.graph.node):
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                visit_graph(attribute.g, f"main/{index}:{attribute.name}")
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph_index, nested in enumerate(attribute.graphs):
                    visit_graph(
                        nested, f"main/{index}:{attribute.name}[{graph_index}]"
                    )
    for function in model.functions:
        for index, node in enumerate(function.node):
            if node.op_type == "GridSample" or (
                "grid" in node.op_type.lower() and "sample" in node.op_type.lower()
            ):
                locations.append(
                    {
                        "location": f"function:{function.domain}::{function.name}",
                        "index": index,
                        "domain": node.domain or "ai.onnx",
                        "op_type": node.op_type,
                        "name": node.name or None,
                    }
                )
    return locations


def _inspect_grid_samples(
    model: Any, values: dict[str, Any], max_probes: int, onnx: Any
) -> dict[str, Any]:
    nodes = list(model.graph.node)
    main_candidates: list[dict[str, Any]] = []
    mapped_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    reasons: list[str] = []
    custom_like: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        looks_like_grid_sample = (
            node.op_type == "GridSample"
            or "grid" in node.op_type.lower()
            and "sample" in node.op_type.lower()
        )
        if not looks_like_grid_sample:
            continue
        if node.op_type != "GridSample" or node.domain not in STANDARD_ONNX_DOMAINS:
            custom_like.append(_node_summary(index, node))
            continue
        scope = _grid_sample_scope(node)
        output_description = (
            _tensor_description(node.output[0], values, onnx)
            if node.output
            else {"shape": None, "estimated_bytes": None}
        )
        candidate = {
            "node": _node_summary(index, node),
            "attributes": {
                "mode": _attribute_string(node, "mode", "linear"),
                "padding_mode": _attribute_string(node, "padding_mode", "zeros"),
                "align_corners": _attribute_int(node, "align_corners", 0),
            },
            "scope": (
                {"stage": scope[0], "layer": scope[1]} if scope is not None else None
            ),
            "value_input": (
                _tensor_description(node.input[0], values, onnx)
                if node.input
                else None
            ),
            "grid_input": (
                _tensor_description(node.input[1], values, onnx)
                if len(node.input) >= 2
                else None
            ),
            "output": output_description,
        }
        main_candidates.append(candidate)
        if scope is None:
            reasons.append(
                f"GridSample node {index} lacks an unambiguous Grounding DINO layer scope"
            )
        else:
            mapped_groups[scope].append(candidate)

    nested = _nested_grid_sample_locations(model, onnx)
    if nested:
        reasons.append("GridSample exists inside a subgraph or local function")
    if custom_like:
        reasons.append("custom or non-standard GridSample-like operators were found")
    if not main_candidates:
        reasons.append("no standard GridSample node was found in the main graph")

    stages: dict[str, dict[int, list[dict[str, Any]]]] = {
        "encoder": defaultdict(list),
        "decoder": defaultdict(list),
    }
    for (stage, layer), candidates in mapped_groups.items():
        stages[stage][layer].extend(candidates)
    for stage in ("encoder", "decoder"):
        layers = sorted(stages[stage])
        if not layers:
            reasons.append(f"no mapped {stage} GridSample layers")
            continue
        if layers != list(range(layers[-1] + 1)):
            reasons.append(f"{stage} GridSample layer indices are not contiguous from zero")
        counts = {len(stages[stage][layer]) for layer in layers}
        if len(counts) != 1:
            reasons.append(f"{stage} GridSample call counts differ between layers")

    selected: list[dict[str, Any]] = []
    if not reasons:
        for stage in ("encoder", "decoder"):
            layers = sorted(stages[stage])
            first_layer = layers[0]
            last_layer = layers[-1]
            first = stages[stage][first_layer][0]
            last = stages[stage][last_layer][-1]
            selected.extend([first, last])
        deduplicated: dict[int, dict[str, Any]] = {
            item["node"]["index"]: item for item in selected
        }
        selected = [deduplicated[index] for index in sorted(deduplicated)]
        if len(selected) > max_probes:
            reasons.append(
                f"{len(selected)} required boundary probes exceed "
                f"max-grid-sample-probes={max_probes}"
            )
            selected = []
        else:
            for item in selected:
                stage = item["scope"]["stage"]
                layer = item["scope"]["layer"]
                same_scope = stages[stage][layer]
                item["scope"]["call_ordinal"] = same_scope.index(item)

    stage_summary = {
        stage: {
            str(layer): len(candidates)
            for layer, candidates in sorted(layers.items())
        }
        for stage, layers in stages.items()
    }
    return {
        "status": "MAPPED" if selected and not reasons else "NO_GO",
        "standard_main_graph_count": len(main_candidates),
        "mapped_layer_call_counts": stage_summary,
        "candidates": main_candidates,
        "custom_or_nonstandard_candidates": custom_like,
        "nested_or_function_candidates": nested,
        "selected": selected,
        "interpretation_boundary": (
            "Selected GridSample values are graph-order sentinels only. A mismatch may "
            "localize the earliest observed divergence but does not establish operator causality."
        ),
        "reasons": reasons,
    }


def _has_external_data(model: Any, onnx: Any) -> bool:
    return any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        or bool(initializer.external_data)
        for initializer in model.graph.initializer
    )


def _add_probe(
    probes_by_name: dict[str, dict[str, Any]],
    tensor_name: str,
    role: str,
    source: dict[str, Any],
    values: dict[str, Any],
    onnx: Any,
) -> None:
    description = _tensor_description(tensor_name, values, onnx)
    existing = probes_by_name.get(tensor_name)
    if existing is not None:
        existing["roles"].append(role)
        existing["sources"].append(source)
        return
    probes_by_name[tensor_name] = {
        **description,
        "roles": [role],
        "sources": [source],
    }


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("sample length and count must be positive")
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    # Integer arithmetic makes the sample definition identical in every runtime.
    indices = [index * (length - 1) // (count - 1) for index in range(count)]
    if len(set(indices)) != len(indices):
        raise RuntimeError("evenly spaced query indices unexpectedly contain duplicates")
    return indices


def _add_grid_sample_probe(
    probes_by_name: dict[str, dict[str, Any]],
    tensor_name: str,
    role: str,
    source: dict[str, Any],
    values: dict[str, Any],
    onnx: Any,
    sample_count: int,
) -> str | None:
    description = _tensor_description(tensor_name, values, onnx)
    shape = description.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or not all(isinstance(item, int) and item >= 0 for item in shape)
        or shape[2] < 1
    ):
        return f"GridSample tensor {tensor_name!r} is not a static non-empty rank-4 output"
    indices = _evenly_spaced_indices(int(shape[2]), sample_count)
    derived_name = (
        f"__gdino_probe_grid_{source['stage']}_{source['layer']}_"
        f"{source['call_ordinal']}_{source['node_index']}"
    )
    sampled_shape = list(shape)
    sampled_shape[2] = len(indices)
    elem_type = description.get("elem_type")
    item_bytes = _dtype_bytes(elem_type, onnx) if elem_type is not None else None
    estimated_bytes = (
        int(math.prod(sampled_shape)) * item_bytes if item_bytes is not None else None
    )
    probes_by_name[derived_name] = {
        "name": derived_name,
        "shape": sampled_shape,
        "dtype": description.get("dtype"),
        "elem_type": elem_type,
        "estimated_bytes": estimated_bytes,
        "roles": [role],
        "sources": [source],
        "derivation": {
            "op": "Gather",
            "source_tensor": tensor_name,
            "axis": 2,
            "indices": indices,
            "source_shape": shape,
            "contract": (
                "evenly spaced query-axis sample from standard ONNX GridSample output"
            ),
        },
    }
    return None


def _build_probe_plan(
    model: Any,
    values: dict[str, Any],
    topk: dict[str, Any],
    grid_samples: dict[str, Any],
    args: argparse.Namespace,
    onnx: Any,
    np: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    probes_by_name: dict[str, dict[str, Any]] = {}
    nodes = list(model.graph.node)
    selected_topk = topk.get("selected")
    if selected_topk is None:
        reasons.append("query-selection TopK is not uniquely mapped")
    else:
        node_index = selected_topk["node"]["index"]
        node = nodes[node_index]
        source = {"kind": "TopK", "node_index": node_index}
        _add_probe(
            probes_by_name, node.input[0], "topk_input_scores", source, values, onnx
        )
        _add_probe(
            probes_by_name, node.output[0], "topk_values", source, values, onnx
        )
        _add_probe(
            probes_by_name, node.output[1], "topk_indices", source, values, onnx
        )

        producers = topk["producer_map"]
        initializers = topk["initializer_map"]
        score_producer_entry = producers.get(node.input[0])
        if score_producer_entry is not None:
            reduce_index, reduce_node = score_producer_entry
            if reduce_node.op_type == "ReduceMax" and reduce_node.input:
                input_shape = _static_shape(values.get(reduce_node.input[0]))
                axes = _reduction_axes(
                    reduce_node,
                    initializers,
                    producers,
                    onnx.numpy_helper,
                    np,
                )
                normalized_axes = (
                    [_normalize_axis(axis, len(input_shape)) for axis in axes]
                    if input_shape is not None and axes is not None
                    else None
                )
                if (
                    input_shape is not None
                    and len(input_shape) == 3
                    and normalized_axes == [2]
                    and _attribute_int(reduce_node, "keepdims", 1) == 0
                ):
                    _add_probe(
                        probes_by_name,
                        reduce_node.input[0],
                        "encoder_class_logits_before_topk_reduce",
                        {"kind": "ReduceMax", "node_index": reduce_index},
                        values,
                        onnx,
                    )
                else:
                    reasons.append(
                        "TopK score producer ReduceMax is not a proven last-axis class reduction"
                    )
            else:
                reasons.append("TopK score input is not produced by ReduceMax")
        else:
            reasons.append("TopK score input producer is unavailable")

        for ordinal, gather_index in enumerate(selected_topk["valid_gather_node_indices"]):
            gather = nodes[gather_index]
            gather_source = {
                "kind": "GatherElements",
                "node_index": gather_index,
                "ordinal": ordinal,
            }
            _add_probe(
                probes_by_name,
                gather.input[0],
                f"topk_gather_{ordinal}_data_before_selection",
                gather_source,
                values,
                onnx,
            )
            _add_probe(
                probes_by_name,
                gather.output[0],
                f"topk_gather_{ordinal}_output_after_selection",
                gather_source,
                values,
                onnx,
            )

    for selected in grid_samples.get("selected", []):
        node_index = selected["node"]["index"]
        node = nodes[node_index]
        scope = selected["scope"]
        role = (
            f"grid_sample_{scope['stage']}_layer_{scope['layer']}_"
            f"call_{scope['call_ordinal']}_output"
        )
        grid_reason = _add_grid_sample_probe(
            probes_by_name,
            node.output[0],
            role,
            {
                "kind": "GridSample",
                "node_index": node_index,
                "attributes": selected["attributes"],
                "interpretation": "sentinel_only_not_causal_attribution",
                **scope,
            },
            values,
            onnx,
            args.grid_sample_query_samples,
        )
        if grid_reason is not None:
            reasons.append(grid_reason)

    graph_outputs = {item.name for item in model.graph.output}
    for output_name in ("logits", "pred_boxes"):
        if output_name not in graph_outputs or output_name not in values:
            reasons.append(f"required final graph output {output_name!r} is unavailable")
            continue
        _add_probe(
            probes_by_name,
            output_name,
            f"final_{output_name}",
            {"kind": "graph_output"},
            values,
            onnx,
        )

    probes = list(probes_by_name.values())
    if len(probes) > args.max_tensors:
        reasons.append(
            f"selected tensor count {len(probes)} exceeds max-tensors={args.max_tensors}"
        )
    total_bytes = 0
    for probe in probes:
        estimated = probe.get("estimated_bytes")
        if estimated is None:
            reasons.append(f"tensor {probe['name']!r} lacks a static byte estimate")
            continue
        if estimated > args.max_single_output_bytes:
            reasons.append(
                f"tensor {probe['name']!r} estimate {estimated} exceeds "
                f"max-single-output-bytes={args.max_single_output_bytes}"
            )
        total_bytes += estimated
        if probe["name"] not in values and "derivation" not in probe:
            reasons.append(f"tensor {probe['name']!r} lacks ONNX ValueInfo")
    if total_bytes > args.max_output_bytes:
        reasons.append(
            f"selected outputs estimate {total_bytes} exceeds "
            f"max-output-bytes={args.max_output_bytes}"
        )
    return probes, reasons


def _sanitize_filename(name: str, role: str, ordinal: int) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", role).strip("._")
    readable = readable[:80] or "tensor"
    short_hash = hashlib.sha256(name.encode()).hexdigest()[:12]
    return f"{ordinal:02d}-{readable}-{short_hash}.npy"


def _array_summary(array: Any, np: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "element_count": int(array.size),
    }
    if np.issubdtype(array.dtype, np.floating):
        finite = np.isfinite(array)
        values = array[finite]
        result.update(
            {
                "finite_count": int(finite.sum()),
                "nan_count": int(np.isnan(array).sum()),
                "positive_infinity_count": int(np.isposinf(array).sum()),
                "negative_infinity_count": int(np.isneginf(array).sum()),
                "finite_min": float(values.min()) if values.size else None,
                "finite_max": float(values.max()) if values.size else None,
                "finite_mean": float(values.mean()) if values.size else None,
            }
        )
    elif np.issubdtype(array.dtype, np.integer):
        result.update(
            {
                "min": int(array.min()) if array.size else None,
                "max": int(array.max()) if array.size else None,
            }
        )
    return result


def _save_array(path: Path, array: Any, np: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.chmod(0o644)
        temporary.replace(path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ort_optimization_level(ort: Any, name: str) -> Any:
    return {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[name]


def _create_ort_session(
    model_path: Path, args: argparse.Namespace, ort: Any
) -> tuple[Any, float]:
    options = ort.SessionOptions()
    options.graph_optimization_level = _ort_optimization_level(
        ort, args.ort_optimization_level
    )
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=[args.provider]
    )
    return session, time.perf_counter() - started


def _load_frozen_feed(inputs_path: Path, input_names: list[str], np: Any) -> dict[str, Any]:
    with np.load(inputs_path, allow_pickle=False) as frozen_inputs:
        missing = [name for name in input_names if name not in frozen_inputs]
        if missing:
            raise RuntimeError(f"frozen inputs are missing model inputs: {missing}")
        return {name: frozen_inputs[name] for name in input_names}


def _compare_instrumentation_output(
    reference: Any,
    candidate: Any,
    atol: float,
    rtol: float,
    np: Any,
) -> dict[str, Any]:
    shape_equal = reference.shape == candidate.shape
    dtype_equal = reference.dtype == candidate.dtype
    comparison: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "nonfinite_pattern_equal": False,
        "finite_values_allclose": False,
        "bit_exact": False,
        "max_abs_on_jointly_finite": None,
        "mean_abs_on_jointly_finite": None,
        "equivalent": False,
    }
    if not shape_equal:
        return comparison
    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    jointly_finite = reference_finite & candidate_finite
    nonfinite_pattern_equal = bool(
        np.array_equal(np.isnan(reference), np.isnan(candidate))
        and np.array_equal(np.isposinf(reference), np.isposinf(candidate))
        and np.array_equal(np.isneginf(reference), np.isneginf(candidate))
    )
    finite_close = bool(
        np.allclose(
            reference[jointly_finite],
            candidate[jointly_finite],
            atol=atol,
            rtol=rtol,
        )
    )
    delta = np.abs(
        reference[jointly_finite].astype(np.float64)
        - candidate[jointly_finite].astype(np.float64)
    )
    bit_exact = bool(
        dtype_equal
        and reference.flags.c_contiguous
        and candidate.flags.c_contiguous
        and reference.tobytes(order="C") == candidate.tobytes(order="C")
    )
    comparison.update(
        {
            "nonfinite_pattern_equal": nonfinite_pattern_equal,
            "jointly_finite_count": int(jointly_finite.sum()),
            "finite_values_allclose": finite_close,
            "bit_exact": bit_exact,
            "max_abs_on_jointly_finite": float(delta.max()) if delta.size else None,
            "mean_abs_on_jointly_finite": float(delta.mean()) if delta.size else None,
            "equivalent": bool(
                dtype_equal and nonfinite_pattern_equal and finite_close
            ),
        }
    )
    return comparison


def _instrument_and_run(
    model: Any,
    original_onnx: Path,
    inputs_path: Path,
    output_dir: Path,
    probes: list[dict[str, Any]],
    values: dict[str, Any],
    args: argparse.Namespace,
    onnx: Any,
    np: Any,
) -> dict[str, Any]:
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    if args.provider not in available_providers:
        raise RuntimeError(
            f"requested provider {args.provider!r} is unavailable; got {available_providers}"
        )

    final_names = ["logits", "pred_boxes"]
    _write_status(output_dir, "RUNNING_ORIGINAL_FINALS", True)
    original_session, original_creation_seconds = _create_ort_session(
        original_onnx, args, ort
    )
    original_actual_providers = original_session.get_providers()
    original_inputs = [item.name for item in original_session.get_inputs()]
    original_outputs = {item.name for item in original_session.get_outputs()}
    missing_finals = [name for name in final_names if name not in original_outputs]
    if missing_finals:
        raise RuntimeError(f"original ONNX lacks final outputs: {missing_finals}")
    feed = _load_frozen_feed(inputs_path, original_inputs, np)
    original_run_started = time.perf_counter()
    original_final_values = original_session.run(final_names, feed)
    original_run_seconds = time.perf_counter() - original_run_started
    del original_session
    gc.collect()
    _write_status(output_dir, "ORIGINAL_FINALS_CAPTURED", True)

    existing_outputs = {item.name for item in model.graph.output}
    for probe in probes:
        derivation = probe.get("derivation")
        if derivation is not None:
            indices_name = f"{probe['name']}__indices"
            indices = np.asarray(derivation["indices"], dtype=np.int64)
            model.graph.initializer.append(
                onnx.numpy_helper.from_array(indices, name=indices_name)
            )
            model.graph.node.append(
                onnx.helper.make_node(
                    "Gather",
                    [derivation["source_tensor"], indices_name],
                    [probe["name"]],
                    axis=int(derivation["axis"]),
                    name=f"{probe['name']}__gather",
                )
            )
            derived_value = onnx.helper.make_tensor_value_info(
                probe["name"], int(probe["elem_type"]), list(probe["shape"])
            )
            values[probe["name"]] = derived_value
        if probe["name"] not in existing_outputs:
            model.graph.output.append(copy.deepcopy(values[probe["name"]]))
            existing_outputs.add(probe["name"])
    onnx.checker.check_model(model, full_check=False)
    instrumented_path = output_dir / "instrumented.onnx"
    temporary = instrumented_path.with_name(
        f".{instrumented_path.name}.tmp-{os.getpid()}"
    )
    try:
        onnx.save_model(model, str(temporary), save_as_external_data=False)
        temporary.chmod(0o644)
        temporary.replace(instrumented_path)
        instrumented_path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()
    onnx.checker.check_model(str(instrumented_path), full_check=False)
    _write_status(
        output_dir,
        "INSTRUMENTED_MODEL_WRITTEN",
        True,
        instrumented_path=str(instrumented_path),
    )

    instrumented_session, instrumented_creation_seconds = _create_ort_session(
        instrumented_path, args, ort
    )
    instrumented_actual_providers = instrumented_session.get_providers()
    instrumented_inputs = [item.name for item in instrumented_session.get_inputs()]
    if instrumented_inputs != original_inputs:
        raise RuntimeError(
            "instrumentation changed model inputs: "
            f"original={original_inputs}, instrumented={instrumented_inputs}"
        )
    output_names = [probe["name"] for probe in probes]
    instrumented_run_started = time.perf_counter()
    arrays = instrumented_session.run(output_names, feed)
    instrumented_run_seconds = time.perf_counter() - instrumented_run_started
    del instrumented_session
    gc.collect()

    if len(arrays) != len(output_names):
        raise RuntimeError(
            f"ORT returned {len(arrays)} arrays for {len(output_names)} requested outputs"
        )
    if len(arrays) > args.max_tensors:
        raise RuntimeError("ORT returned more arrays than the configured tensor cap")
    total_actual_bytes = sum(int(array.nbytes) for array in arrays)
    if total_actual_bytes > args.max_output_bytes:
        raise RuntimeError(
            f"ORT output bytes {total_actual_bytes} exceed cap {args.max_output_bytes}"
        )
    candidates = dict(zip(output_names, arrays))
    comparisons = {
        name: _compare_instrumentation_output(
            reference,
            candidates[name],
            args.instrumentation_atol,
            args.instrumentation_rtol,
            np,
        )
        for name, reference in zip(final_names, original_final_values)
    }
    same_provider_stack = original_actual_providers == instrumented_actual_providers
    requested_provider_active = bool(
        original_actual_providers
        and instrumented_actual_providers
        and original_actual_providers[0] == args.provider
        and instrumented_actual_providers[0] == args.provider
    )
    instrumentation_equivalent = (
        all(item["equivalent"] for item in comparisons.values())
        and same_provider_stack
        and requested_provider_active
    )
    common = {
        "onnx": {
            "original_path": str(original_onnx),
            "original_sha256": _sha256(original_onnx),
            "instrumented_path": str(instrumented_path),
            "instrumented_sha256": _sha256(instrumented_path),
            "instrumented_size_bytes": instrumented_path.stat().st_size,
        },
        "onnxruntime": {
            "version": ort.__version__,
            "provider": args.provider,
            "available_providers": available_providers,
            "original_actual_providers": original_actual_providers,
            "instrumented_actual_providers": instrumented_actual_providers,
            "optimization_level": args.ort_optimization_level,
            "original_session_creation_seconds": original_creation_seconds,
            "original_run_seconds": original_run_seconds,
            "instrumented_session_creation_seconds": instrumented_creation_seconds,
            "instrumented_run_seconds": instrumented_run_seconds,
            "peak_process_rss_bytes": _peak_rss_bytes(),
        },
        "instrumentation_equivalence": {
            "atol": args.instrumentation_atol,
            "rtol": args.instrumentation_rtol,
            "same_provider_stack": same_provider_stack,
            "requested_provider_active": requested_provider_active,
            "same_optimization_level": True,
            "same_frozen_inputs": True,
            "comparisons": comparisons,
            "equivalent": instrumentation_equivalent,
        },
    }
    if not instrumentation_equivalent:
        _write_status(
            output_dir,
            "NO_GO_INSTRUMENTATION_PERTURBED",
            False,
            intermediate_outputs_valid=False,
        )
        return {
            **common,
            "intermediate_outputs_valid": False,
            "outputs": {
                "status": "NOT_SAVED_INSTRUMENTATION_PERTURBED",
                "count": 0,
                "total_array_bytes": 0,
                "tensors": [],
            },
        }

    _write_status(output_dir, "INSTRUMENTATION_EQUIVALENCE_PASSED", True)
    tensors_dir = output_dir / "tensors"
    tensors_dir.mkdir()
    tensors_dir.chmod(0o755)
    tensor_results: list[dict[str, Any]] = []
    for ordinal, (probe, array) in enumerate(zip(probes, arrays)):
        if array.dtype.hasobject:
            raise RuntimeError(f"refusing object output for tensor {probe['name']!r}")
        if array.nbytes > args.max_single_output_bytes:
            raise RuntimeError(
                f"actual output {probe['name']!r} exceeds single-output cap"
            )
        if list(array.shape) != probe["shape"]:
            raise RuntimeError(
                f"actual shape for {probe['name']!r} differs from the static plan"
            )
        filename = _sanitize_filename(probe["name"], probe["roles"][0], ordinal)
        tensor_path = tensors_dir / filename
        _save_array(tensor_path, array, np)
        tensor_results.append(
            {
                **probe,
                "file": str(tensor_path),
                "file_sha256": _sha256(tensor_path),
                "file_size_bytes": tensor_path.stat().st_size,
                "array": _array_summary(array, np),
            }
        )

    return {
        **common,
        "intermediate_outputs_valid": True,
        "outputs": {
            "status": "SAVED_AFTER_INSTRUMENTATION_EQUIVALENCE_GATE",
            "count": len(tensor_results),
            "total_array_bytes": total_actual_bytes,
            "tensors": tensor_results,
        },
    }


def _compact_topk(topk: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in topk.items()
        if key not in {"producer_map", "consumer_map", "initializer_map"}
    }


def _pre_load_resource_gate(
    args: argparse.Namespace,
    output_dir: Path,
    onnx_path: Path,
    inputs_path: Path,
) -> tuple[int | None, Any, list[str]]:
    memory_available = _available_memory_bytes()
    disk = shutil.disk_usage(output_dir)
    reasons: list[str] = []
    if memory_available is None:
        reasons.append("available memory could not be measured on this host")
    elif memory_available < args.min_available_memory_bytes:
        reasons.append(
            f"available memory {memory_available} is below required "
            f"{args.min_available_memory_bytes}"
        )
    disk_required = args.min_disk_reserve_bytes
    if not args.inspect_only:
        disk_required += onnx_path.stat().st_size + args.max_output_bytes
    if disk.free < disk_required:
        reasons.append(
            f"free disk {disk.free} is below pre-load requirement {disk_required}"
        )
    if reasons:
        inspection = {
            "schema_version": SCHEMA_VERSION,
            "scope": SCOPE,
            "verdict": "NO_GO",
            "reason": "Pre-load resource gate failed; ONNX was not loaded.",
            "created_at_unix": int(time.time()),
            "stage": "PRE_ONNX_LOAD_RESOURCE_GATE",
            "model": {
                "path": str(onnx_path),
                "size_bytes": onnx_path.stat().st_size,
                "sha256": None,
                "loaded": False,
            },
            "inputs": {"path": str(inputs_path), "sha256": None},
            "limits": {
                "min_available_memory_bytes": args.min_available_memory_bytes,
                "min_disk_reserve_bytes": args.min_disk_reserve_bytes,
                "max_output_bytes": args.max_output_bytes,
            },
            "resources": {
                "available_memory_bytes": memory_available,
                "free_disk_bytes": disk.free,
                "pre_load_disk_required_bytes": disk_required,
            },
            "reasons": reasons,
        }
        _write_json(output_dir / "inspection.json", inspection)
        _write_status(
            output_dir,
            "NO_GO_PRE_ONNX_LOAD_RESOURCE_GATE",
            False,
            onnx_loaded=False,
        )
        print(
            json.dumps(inspection, ensure_ascii=False, indent=2, sort_keys=True),
            flush=True,
        )
    return memory_available, disk, reasons


def _run(args: argparse.Namespace, output_dir: Path) -> int:
    onnx_path = Path(args.onnx)
    inputs_path = Path(args.inputs)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)

    # This gate intentionally runs before importing ONNX and before deserializing
    # the approximately one-gigabyte protobuf. Inspect-only uses the same gate.
    memory_available, disk, pre_load_reasons = _pre_load_resource_gate(
        args, output_dir, onnx_path, inputs_path
    )
    if pre_load_reasons:
        return 2

    import numpy as np
    import onnx

    load_started = time.perf_counter()
    model = onnx.load(str(onnx_path), load_external_data=False)
    load_seconds = time.perf_counter() - load_started
    has_external_data = _has_external_data(model, onnx)
    if not has_external_data:
        onnx.checker.check_model(model, full_check=False)
    values = _value_info_map(model, onnx)
    reasons: list[str] = []
    if has_external_data:
        reasons.append(
            "external-data ONNX models are unsupported because relocating references is unsafe"
        )

    topk = _inspect_topk(
        model, values, args.expected_num_queries, onnx=onnx, np=np
    )
    grid_samples = _inspect_grid_samples(
        model, values, args.max_grid_sample_probes, onnx
    )
    if topk["status"] != "MAPPED":
        reasons.extend(f"TopK: {item}" for item in topk["reasons"])
    if grid_samples["status"] != "MAPPED":
        reasons.extend(f"GridSample: {item}" for item in grid_samples["reasons"])

    probes, plan_reasons = _build_probe_plan(
        model, values, topk, grid_samples, args, onnx, np
    )
    reasons.extend(f"probe plan: {item}" for item in plan_reasons)
    estimated_output_bytes = sum(
        int(probe.get("estimated_bytes") or 0) for probe in probes
    )
    disk_required = args.min_disk_reserve_bytes
    if not args.inspect_only:
        disk_required += onnx_path.stat().st_size + estimated_output_bytes
    if disk.free < disk_required:
        reasons.append(
            f"free disk {disk.free} is below model+outputs+reserve {disk_required}"
        )

    verdict = "NO_GO" if reasons else ("INSPECT_ONLY" if args.inspect_only else "GO")
    inspection = {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "verdict": verdict,
        "reason": (
            "Ambiguous or unsafe graph instrumentation; ORT was not run."
            if reasons
            else (
                "Mapping passed; inspect-only requested, so no ONNX copy or ORT run was made."
                if args.inspect_only
                else "Mapping and resource gates passed; selected outputs may be instrumented."
            )
        ),
        "created_at_unix": int(time.time()),
        "model": {
            "path": str(onnx_path),
            "sha256": _sha256(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "load_seconds": load_seconds,
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
            "has_external_data": has_external_data,
            "opset_imports": {
                item.domain or "ai.onnx": item.version for item in model.opset_import
            },
        },
        "inputs": {"path": str(inputs_path), "sha256": _sha256(inputs_path)},
        "mapping_contract": {
            "mark_all": False,
            "topk": (
                "unique standard TopK; K equals logits query dimension; rank-2 axis-1 "
                "scores; largest=1; sorted=1; INT64 indices; indices reach valid "
                "axis-1 GatherElements"
            ),
            "grid_sample": (
                "all standard main-graph GridSample nodes carry explicit deformable-attention "
                "encoder/decoder layer metadata; layers are contiguous and call counts stable; "
                "selected values are sentinels and do not establish operator causality"
            ),
            "selection": (
                "TopK pre/post tensors, verified proposal gathers, final outputs, and bounded "
                "query-axis samples from first/last mapped GridSample sentinels for encoder "
                "and decoder; never mark-all"
            ),
            "instrumentation_equivalence": (
                "same frozen inputs, provider, and optimization level on original and "
                "instrumented ONNX; shape, dtype, nonfinite patterns, and finite allclose "
                "must pass before intermediate tensors are saved"
            ),
        },
        "discoveries": {
            "topk": _compact_topk(topk),
            "grid_sample": grid_samples,
        },
        "probe_plan": {
            "count": len(probes),
            "estimated_output_bytes": estimated_output_bytes,
            "tensors": probes,
        },
        "limits": {
            "max_grid_sample_probes": args.max_grid_sample_probes,
            "grid_sample_query_samples": args.grid_sample_query_samples,
            "max_tensors": args.max_tensors,
            "max_output_bytes": args.max_output_bytes,
            "max_single_output_bytes": args.max_single_output_bytes,
            "min_available_memory_bytes": args.min_available_memory_bytes,
            "min_disk_reserve_bytes": args.min_disk_reserve_bytes,
            "instrumentation_atol": args.instrumentation_atol,
            "instrumentation_rtol": args.instrumentation_rtol,
        },
        "resources": {
            "available_memory_bytes": memory_available,
            "free_disk_bytes": disk.free,
            "minimum_disk_required_bytes": disk_required,
        },
        "reasons": reasons,
    }
    _write_json(output_dir / "inspection.json", inspection)
    print(json.dumps(inspection, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if reasons:
        _write_status(output_dir, "NO_GO_GRAPH_INSPECTION", False)
        return 2
    if args.inspect_only:
        _write_status(output_dir, "INSPECT_ONLY_COMPLETE", False)
        return 0

    _write_status(output_dir, "READY_FOR_INSTRUMENTATION_GATE", True)
    run = _instrument_and_run(
        model,
        onnx_path,
        inputs_path,
        output_dir,
        probes,
        values,
        args,
        onnx,
        np,
    )
    valid = bool(run["intermediate_outputs_valid"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "verdict": (
            "INTERMEDIATES_CAPTURED_NOT_YET_COMPARED_TO_PYTORCH"
            if valid
            else "NO_GO_INSTRUMENTATION_PERTURBED"
        ),
        "interpretation_boundary": (
            "GridSample captures are sentinels only; divergence does not by itself prove "
            "GridSample is the causal operator."
        ),
        "created_at_unix": int(time.time()),
        "inspection": str(output_dir / "inspection.json"),
        **run,
    }
    _write_json(output_dir / "result.json", result)
    _write_status(
        output_dir,
        (
            "COMPLETE_INTERMEDIATES_CAPTURED"
            if valid
            else "COMPLETE_NO_GO_INSTRUMENTATION_PERTURBED"
        ),
        False,
        result_path=str(output_dir / "result.json"),
        intermediate_outputs_valid=valid,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if valid else 2


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    prepared = False
    try:
        _prepare_output_dir(output_dir)
        prepared = True
        return _run(args, output_dir)
    except Exception as exc:
        previous_status = None
        status_path = output_dir / "status.json"
        if prepared and status_path.is_file():
            try:
                previous_status = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                previous_status = {"state": "UNREADABLE_STATUS"}
        failure = {
            "schema_version": SCHEMA_VERSION,
            "scope": SCOPE,
            "verdict": "NO_GO_INCOMPLETE",
            "incomplete": True,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "instrumented_model_exists": (output_dir / "instrumented.onnx").is_file(),
            "previous_status": previous_status,
        }
        if prepared:
            try:
                _write_json(output_dir / "failure.json", failure)
                _write_status(
                    output_dir,
                    "FAILED_INCOMPLETE",
                    True,
                    failure_path=str(output_dir / "failure.json"),
                    instrumented_model_exists=(
                        output_dir / "instrumented.onnx"
                    ).is_file(),
                    previous_state=(previous_status or {}).get("state"),
                )
            except Exception:
                pass
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

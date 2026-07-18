#!/usr/bin/env python
"""Dump Grounding DINO boundaries around the two-stage proposal ``topk``.

The probe is intentionally eager-FP32-only.  It is designed to answer whether
an exported backend first diverges in the visual backbone, in the encoder, or
when small score drift changes the 900 proposals selected for the decoder.

The implementation is pinned to the Hugging Face Transformers 5.13.1 source
contract.  In that implementation ``GroundingDinoModel.forward`` computes
``enc_outputs_class.max(-1)[0]`` and passes it directly to ``torch.topk`` with
``dim=1`` and ``config.num_queries``.  The exact call is intercepted rather
than inferred from a downstream tensor or an assumed module name.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


INPUT_NAMES = (
    "pixel_values",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "pixel_mask",
)
EXPECTED_TRANSFORMERS_VERSION = "5.13.1"
EXPECTED_TOPK = 900
DEFAULT_MAX_NPZ_BYTES = 49_000_000
MODEL_ARTIFACT_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".safetensors",
    ".txt",
    ".vocab",
}
TOPK_SOURCE_FRAGMENTS = (
    "topk_logits = enc_outputs_class.max(-1)[0]",
    "topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256_bytes(memoryview(contiguous).cast("B"))


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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.size),
        "sha256": _sha256_array(value),
    }


def _tensor_summary(tensor) -> dict[str, Any]:
    """Summarize a tensor without copying the full tensor to host memory."""

    import torch

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite].float()
    return {
        "shape": [int(item) for item in detached.shape],
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "posinf_count": int(torch.isposinf(detached).sum().item()),
        "neginf_count": int(torch.isneginf(detached).sum().item()),
        "finite_min": (
            _json_float(finite_values.min().item()) if finite_values.numel() else None
        ),
        "finite_max": (
            _json_float(finite_values.max().item()) if finite_values.numel() else None
        ),
        "finite_mean": (
            _json_float(finite_values.mean().item()) if finite_values.numel() else None
        ),
        "finite_std": (
            _json_float(finite_values.std(unbiased=False).item())
            if finite_values.numel()
            else None
        ),
    }


def _sample_tensor(tensor, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic flat indices and FP32 values for cross-runtime use."""

    import torch

    flat = tensor.detach().reshape(-1)
    count = int(flat.numel())
    if not count:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    selected = min(count, sample_size)
    indices = np.linspace(0, count - 1, num=selected, dtype=np.int64)
    torch_indices = torch.from_numpy(indices).to(device=flat.device)
    values = flat.index_select(0, torch_indices).float().cpu().numpy()
    return indices, values


def _store_sample(
    arrays: dict[str, np.ndarray],
    summaries: dict[str, dict[str, Any]],
    key: str,
    tensor,
    sample_size: int,
) -> None:
    indices, values = _sample_tensor(tensor, sample_size)
    arrays[f"{key}_flat_indices"] = indices
    arrays[f"{key}_sample"] = values
    summaries[key] = {
        **_tensor_summary(tensor),
        "sample_size": int(values.size),
        "sample_values_sha256": _sha256_array(values),
        "sample_flat_indices_sha256": _sha256_array(indices),
    }


def _find_exactly_one_module(model, expected_type, label: str):
    matches = [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(module, expected_type)
    ]
    if len(matches) != 1:
        found = [name for name, _ in matches]
        raise RuntimeError(f"expected exactly one {label}, found {len(matches)}: {found}")
    return matches[0]


def _topk_source_contract(modeling_module) -> dict[str, Any]:
    forward = modeling_module.GroundingDinoModel.forward
    source_lines, first_line = inspect.getsourcelines(forward)
    source = "".join(source_lines)
    missing = [fragment for fragment in TOPK_SOURCE_FRAGMENTS if fragment not in source]
    if missing:
        raise RuntimeError(
            "installed GroundingDinoModel.forward does not match the audited "
            f"Transformers 5.13.1 topk contract; missing {missing}"
        )
    locations = {}
    for fragment in TOPK_SOURCE_FRAGMENTS:
        offset = next(
            index for index, line in enumerate(source_lines) if fragment in line
        )
        locations[fragment] = first_line + offset
    source_file = Path(inspect.getsourcefile(modeling_module) or "")
    if not source_file.is_file():
        raise RuntimeError(f"cannot resolve installed Grounding DINO source: {source_file}")
    return {
        "module_file": str(source_file),
        "module_file_sha256": _sha256_file(source_file),
        "forward_source_sha256": _sha256_bytes(source.encode("utf-8")),
        "forward_first_line": int(first_line),
        "fragment_lines": locations,
    }


@contextlib.contextmanager
def _capture_exact_topk(torch_module, *, expected_k: int) -> Iterator[list[dict[str, Any]]]:
    """Intercept the audited two-stage ``torch.topk`` call and preserve its result."""

    original_topk = torch_module.topk
    records: list[dict[str, Any]] = []

    def audited_topk(input_tensor, k, *args, **kwargs):
        result = original_topk(input_tensor, k, *args, **kwargs)
        dim = kwargs.get("dim", args[0] if args else None)
        normalized_dim = (
            int(dim) % input_tensor.ndim if dim is not None and input_tensor.ndim else None
        )
        if (
            int(k) == expected_k
            and input_tensor.ndim == 2
            and normalized_dim == 1
            and input_tensor.is_floating_point()
        ):
            records.append(
                {
                    "input": input_tensor.detach().float().cpu().numpy(),
                    "values": result.values.detach().float().cpu().numpy(),
                    "indices": result.indices.detach().cpu().numpy().astype(np.int64),
                    "input_dtype": str(input_tensor.dtype),
                    "sorted": bool(kwargs.get("sorted", args[2] if len(args) > 2 else True)),
                    "largest": bool(kwargs.get("largest", args[1] if len(args) > 1 else True)),
                }
            )
        return result

    torch_module.topk = audited_topk
    try:
        yield records
    finally:
        torch_module.topk = original_topk


def _boundary_diagnostics(
    scores,
    selected_indices,
    selected_values,
    *,
    boundary_window: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Measure how much score perturbation is required to change the top-k set."""

    import torch

    batch_size, proposal_count = scores.shape
    k = selected_indices.shape[1]
    extended_k = min(proposal_count, k + boundary_window)
    extended_values, extended_indices = torch.topk(
        scores, extended_k, dim=1, largest=True, sorted=True
    )
    arrays = {
        "topk_boundary_rank_indices": extended_indices[:, max(0, k - boundary_window) :]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64),
        "topk_boundary_rank_scores": extended_values[:, max(0, k - boundary_window) :]
        .detach()
        .float()
        .cpu()
        .numpy(),
    }
    rows = []
    epsilons = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
    for batch_index in range(batch_size):
        batch_scores = scores[batch_index]
        batch_selected_indices = selected_indices[batch_index]
        batch_selected_values = selected_values[batch_index]
        selected_mask = torch.zeros(
            proposal_count, dtype=torch.bool, device=batch_scores.device
        )
        selected_mask.scatter_(0, batch_selected_indices, True)
        unselected = batch_scores.masked_fill(selected_mask, float("-inf"))
        lowest_selected = batch_selected_values.min()
        highest_rejected = unselected.max()
        gap = lowest_selected - highest_rejected
        finite_scores = batch_scores[torch.isfinite(batch_scores)]
        ordered_indices = batch_selected_indices.detach().cpu().numpy().astype(np.int64)
        set_indices = np.sort(ordered_indices)
        row = {
            "batch_index": batch_index,
            "proposal_count": proposal_count,
            "selected_count": k,
            "lowest_selected_score": _json_float(lowest_selected.item()),
            "highest_rejected_score": _json_float(highest_rejected.item()),
            "selected_rejected_gap": _json_float(gap.item()),
            "guaranteed_linf_perturbation_radius": (
                _json_float(max(0.0, gap.item() / 2.0))
                if torch.isfinite(gap)
                else None
            ),
            "selected_order_sha256": _sha256_array(ordered_indices),
            "selected_set_sha256": _sha256_array(set_indices),
            "boundary_exact_tie_count": int(
                (finite_scores == lowest_selected).sum().item()
            ),
            "near_boundary_counts": {
                f"abs_le_{epsilon:g}": int(
                    (torch.abs(finite_scores - lowest_selected) <= epsilon).sum().item()
                )
                for epsilon in epsilons
            },
            "topk_set_guaranteed_stable_for_linf": {
                f"epsilon_{epsilon:g}": bool(
                    torch.isfinite(gap).item() and gap.item() > 2.0 * epsilon
                )
                for epsilon in epsilons
            },
        }
        rows.append(row)
    return arrays, rows


def _model_artifact_hashes(model_dir: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_ARTIFACT_SUFFIXES
    )
    if not files:
        raise RuntimeError(f"no model artifacts found under {model_dir}")
    records = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(model_dir).as_posix()
        digest = _sha256_file(path)
        size = path.stat().st_size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        records.append({"path": relative, "bytes": size, "sha256": digest})
    return {"combined_sha256": aggregate.hexdigest(), "files": records}


def _input_manifest(frozen: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: _array_descriptor(frozen[name]) for name in INPUT_NAMES}


def _finite_array_diff(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    reference_finite = np.isfinite(reference)
    candidate_finite = np.isfinite(candidate)
    jointly_finite = reference_finite & candidate_finite
    delta = np.abs(reference[jointly_finite] - candidate[jointly_finite])
    return {
        "shape_equal": True,
        "dtype_equal": reference.dtype == candidate.dtype,
        "bit_exact": bool(np.array_equal(reference, candidate)),
        "nonfinite_pattern_equal": bool(
            np.array_equal(np.isnan(reference), np.isnan(candidate))
            and np.array_equal(np.isposinf(reference), np.isposinf(candidate))
            and np.array_equal(np.isneginf(reference), np.isneginf(candidate))
        ),
        "jointly_finite_count": int(jointly_finite.sum()),
        "max_abs_on_jointly_finite": (
            _json_float(delta.max()) if delta.size else None
        ),
        "mean_abs_on_jointly_finite": (
            _json_float(delta.mean()) if delta.size else None
        ),
    }


def _numpy_boundary_gap(
    scores: np.ndarray, indices: np.ndarray, values: np.ndarray, batch_index: int
) -> float | None:
    batch_scores = scores[batch_index]
    selected = indices[batch_index]
    selected_mask = np.zeros(batch_scores.shape[0], dtype=bool)
    selected_mask[selected] = True
    finite_rejected = batch_scores[(~selected_mask) & np.isfinite(batch_scores)]
    finite_selected = values[batch_index][np.isfinite(values[batch_index])]
    if not finite_rejected.size or not finite_selected.size:
        return None
    return _json_float(finite_selected.min() - finite_rejected.max())


def _resolve_reference_capture(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / "stage-boundaries.npz", path / "summary.json"
    if path.suffix == ".npz":
        return path, path.with_name("summary.json")
    raise ValueError("compare-to must be a capture directory or stage-boundaries.npz")


def _compare_capture(
    reference_path: Path,
    candidate: Mapping[str, np.ndarray],
    *,
    input_sha256: str,
    model_combined_sha256: str | None,
) -> dict[str, Any]:
    npz_path, summary_path = _resolve_reference_capture(reference_path)
    if not npz_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"reference capture requires {npz_path} and {summary_path}"
        )
    reference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded_npz_sha256 = reference_summary.get("artifacts", {}).get("npz_sha256")
    actual_npz_sha256 = _sha256_file(npz_path)
    if recorded_npz_sha256 != actual_npz_sha256:
        raise RuntimeError(
            "reference capture NPZ hash does not match its summary: "
            f"recorded={recorded_npz_sha256}, actual={actual_npz_sha256}"
        )
    reference_input_sha256 = reference_summary.get("inputs", {}).get("file_sha256")
    if reference_input_sha256 != input_sha256:
        raise RuntimeError(
            "reference and candidate do not use the same frozen input file: "
            f"{reference_input_sha256} != {input_sha256}"
        )
    reference_model_sha256 = (
        reference_summary.get("model", {}).get("artifact_hashes") or {}
    ).get("combined_sha256")
    if (
        reference_model_sha256 is not None
        and model_combined_sha256 is not None
        and reference_model_sha256 != model_combined_sha256
    ):
        raise RuntimeError(
            "reference and candidate model artifact hashes differ: "
            f"{reference_model_sha256} != {model_combined_sha256}"
        )

    with np.load(npz_path, allow_pickle=False) as loaded:
        reference = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    required = {
        "encoder_proposal_scores",
        "encoder_proposal_coord_logits",
        "topk_indices",
        "topk_values",
        "final_logits",
        "final_pred_boxes",
    }
    missing_reference = sorted(required - reference.keys())
    missing_candidate = sorted(required - candidate.keys())
    if missing_reference or missing_candidate:
        raise KeyError(
            f"capture comparison missing arrays; reference={missing_reference}, "
            f"candidate={missing_candidate}"
        )

    reference_indices = reference["topk_indices"]
    candidate_indices = candidate["topk_indices"]
    if reference_indices.shape != candidate_indices.shape:
        raise RuntimeError(
            f"topk index shapes differ: {reference_indices.shape} != {candidate_indices.shape}"
        )
    per_batch = []
    for batch_index in range(reference_indices.shape[0]):
        left = reference_indices[batch_index]
        right = candidate_indices[batch_index]
        left_set = set(int(item) for item in left)
        right_set = set(int(item) for item in right)
        intersection = left_set & right_set
        union = left_set | right_set
        reference_gap = _numpy_boundary_gap(
            reference["encoder_proposal_scores"],
            reference_indices,
            reference["topk_values"],
            batch_index,
        )
        candidate_gap = _numpy_boundary_gap(
            candidate["encoder_proposal_scores"],
            candidate_indices,
            candidate["topk_values"],
            batch_index,
        )
        reference_only = sorted(left_set - right_set)
        candidate_only = sorted(right_set - left_set)
        per_batch.append(
            {
                "batch_index": batch_index,
                "ordered_indices_equal": bool(np.array_equal(left, right)),
                "same_rank_count": int((left == right).sum()),
                "same_rank_fraction": float((left == right).mean()),
                "set_equal": left_set == right_set,
                "set_overlap_count": len(intersection),
                "set_overlap_fraction_of_k": len(intersection) / len(left_set),
                "set_jaccard": len(intersection) / len(union) if union else 1.0,
                "reference_only_count": len(reference_only),
                "candidate_only_count": len(candidate_only),
                "reference_only_first_32": reference_only[:32],
                "candidate_only_first_32": candidate_only[:32],
                "reference_boundary_gap": reference_gap,
                "candidate_boundary_gap": candidate_gap,
                "boundary_gap_delta": (
                    _json_float(candidate_gap - reference_gap)
                    if reference_gap is not None and candidate_gap is not None
                    else None
                ),
            }
        )

    direct_keys = sorted(
        key
        for key in required - {"topk_indices"}
        if key in reference and key in candidate
    )
    sample_keys = sorted(
        key
        for key in reference
        if key.endswith("_sample") and key in candidate
    )
    sample_comparisons = {}
    for key in sample_keys:
        index_key = key.removesuffix("_sample") + "_flat_indices"
        indices_equal = bool(
            index_key in reference
            and index_key in candidate
            and np.array_equal(reference[index_key], candidate[index_key])
        )
        sample_comparisons[key] = {
            "sample_flat_indices_equal": indices_equal,
            **(
                _finite_array_diff(reference[key], candidate[key])
                if indices_equal
                else {
                    "shape_equal": reference[key].shape == candidate[key].shape,
                    "comparison_valid": False,
                }
            ),
        }
    return {
        "reference": {
            "npz": str(npz_path),
            "npz_sha256": actual_npz_sha256,
            "summary": str(summary_path),
        },
        "same_frozen_input_sha256": True,
        "same_model_artifact_sha256": (
            reference_model_sha256 == model_combined_sha256
            if reference_model_sha256 is not None and model_combined_sha256 is not None
            else None
        ),
        "topk": {
            "ordered_indices_equal_all_batches": all(
                row["ordered_indices_equal"] for row in per_batch
            ),
            "sets_equal_all_batches": all(row["set_equal"] for row in per_batch),
            "per_batch": per_batch,
        },
        "full_tensor_diffs": {
            key: _finite_array_diff(reference[key], candidate[key])
            for key in direct_keys
        },
        "sample_tensor_diffs": sample_comparisons,
    }


def _write_artifacts(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    *,
    max_npz_bytes: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    npz_path = output_dir / "stage-boundaries.npz"
    json_path = output_dir / "summary.json"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", dir=output_dir, delete=False
        ) as handle:
            npz_tmp = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        npz_bytes = npz_tmp.stat().st_size
        if npz_bytes >= max_npz_bytes:
            raise RuntimeError(
                f"compressed capture is {npz_bytes} bytes; refusing artifact at or above "
                f"the {max_npz_bytes}-byte pull limit"
            )
        os.replace(npz_tmp, npz_path)
        npz_path.chmod(0o644)
        report["artifacts"] = {
            "npz": npz_path.name,
            "npz_bytes": npz_bytes,
            "npz_size_limit_bytes_exclusive": max_npz_bytes,
            "npz_sha256": _sha256_file(npz_path),
            "npz_arrays": {key: _array_descriptor(value) for key, value in arrays.items()},
            "json": json_path.name,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=output_dir, delete=False, encoding="utf-8"
        ) as handle:
            json_tmp = Path(handle.name)
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(json_tmp, json_path)
        json_path.chmod(0o644)
    except BaseException:
        for candidate in (locals().get("npz_tmp"), locals().get("json_tmp")):
            if isinstance(candidate, Path):
                candidate.unlink(missing_ok=True)
        if output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-manifest")
    parser.add_argument(
        "--compare-to",
        help=(
            "Optional prior capture directory (or stage-boundaries.npz) to compare "
            "against in the emitted summary."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--boundary-window", type=int, default=32)
    parser.add_argument("--max-npz-bytes", type=int, default=DEFAULT_MAX_NPZ_BYTES)
    parser.add_argument("--expected-topk", type=int, default=EXPECTED_TOPK)
    parser.add_argument(
        "--require-transformers",
        default=EXPECTED_TRANSFORMERS_VERSION,
        help="Fail closed unless the installed Transformers version is exactly this value.",
    )
    parser.add_argument(
        "--tf32-policy",
        choices=("disabled", "pytorch-default", "enabled"),
        default="disabled",
        help=(
            "Explicit TF32 state: disabled=false/false, pytorch-default="
            "matmul-false/cudnn-true, enabled=true/true."
        ),
    )
    parser.add_argument(
        "--skip-model-artifact-hash",
        action="store_true",
        help="Skip hashing model weights; source, config, and frozen input hashes remain recorded.",
    )
    parser.add_argument("--code-commit")
    args = parser.parse_args(argv)
    if (
        args.sample_size < 1
        or args.boundary_window < 1
        or args.expected_topk < 1
        or args.max_npz_bytes < 1
    ):
        parser.error(
            "sample-size, boundary-window, expected-topk, and max-npz-bytes "
            "must be positive"
        )

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    inputs_path = Path(args.inputs)
    model_dir = Path(args.model_dir)
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)

    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection
    from transformers.models.grounding_dino import modeling_grounding_dino

    if transformers.__version__ != args.require_transformers:
        raise RuntimeError(
            f"Transformers {args.require_transformers} is required; found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.expected_topk != EXPECTED_TOPK:
        raise RuntimeError(
            f"this experiment is frozen at {EXPECTED_TOPK} queries; got {args.expected_topk}"
        )

    source_contract = _topk_source_contract(modeling_grounding_dino)
    with np.load(inputs_path, allow_pickle=False) as loaded:
        missing = [name for name in INPUT_NAMES if name not in loaded]
        if missing:
            raise KeyError(f"frozen inputs are missing: {missing}")
        frozen = {name: np.array(loaded[name], copy=True) for name in INPUT_NAMES}

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tf32_states = {
        "disabled": (False, False, "highest"),
        "pytorch-default": (False, True, "highest"),
        "enabled": (True, True, "high"),
    }
    matmul_tf32, cudnn_tf32, matmul_precision = tf32_states[args.tf32_policy]
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)

    positional_inputs = tuple(
        torch.from_numpy(frozen[name]).to(device="cuda") for name in INPUT_NAMES
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).cuda().eval()
    core = model.model
    if not bool(core.config.two_stage):
        raise RuntimeError("the loaded Grounding DINO model is not configured for two-stage proposals")
    if int(core.config.num_queries) != args.expected_topk:
        raise RuntimeError(
            f"expected config.num_queries={args.expected_topk}, found {core.config.num_queries}"
        )

    backbone_name, backbone = _find_exactly_one_module(
        model, modeling_grounding_dino.GroundingDinoConvModel, "GroundingDinoConvModel"
    )
    encoder_name, encoder = _find_exactly_one_module(
        model, modeling_grounding_dino.GroundingDinoEncoder, "GroundingDinoEncoder"
    )
    captures: dict[str, Any] = {"backbone": None, "projected": {}, "encoder": None}
    handles = []

    def capture_backbone(_module, _inputs, output):
        if captures["backbone"] is not None:
            raise RuntimeError("backbone executed more than once")
        # GroundingDinoModel.forward appends the synthetic fourth-level position
        # embedding to the list returned by the backbone.  Freeze both list
        # containers at hook time so the captured native backbone levels keep
        # matching after the caller mutates ``position_embeddings_list``.
        backbone_levels, backbone_positions = output
        captures["backbone"] = (
            tuple(backbone_levels),
            tuple(backbone_positions),
        )

    def capture_encoder(_module, _inputs, output):
        if captures["encoder"] is not None:
            raise RuntimeError("encoder executed more than once")
        captures["encoder"] = output

    def capture_projection(index: int):
        def hook(_module, _inputs, output):
            key = str(index)
            if key in captures["projected"]:
                raise RuntimeError(f"input_proj_vision[{index}] executed more than once")
            captures["projected"][key] = output

        return hook

    handles.append(backbone.register_forward_hook(capture_backbone))
    handles.append(encoder.register_forward_hook(capture_encoder))
    for index, projection in enumerate(core.input_proj_vision):
        handles.append(projection.register_forward_hook(capture_projection(index)))

    try:
        with _capture_exact_topk(
            torch, expected_k=args.expected_topk
        ) as topk_records, torch.inference_mode():
            outputs = model(
                pixel_values=positional_inputs[0],
                input_ids=positional_inputs[1],
                token_type_ids=positional_inputs[2],
                attention_mask=positional_inputs[3],
                pixel_mask=positional_inputs[4],
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    if len(topk_records) != 1:
        raise RuntimeError(
            "expected exactly one audited two-stage torch.topk call, "
            f"captured {len(topk_records)}"
        )
    if captures["backbone"] is None or captures["encoder"] is None:
        raise RuntimeError("required backbone or encoder hook did not execute")
    if outputs.enc_outputs_class is None or outputs.enc_outputs_coord_logits is None:
        raise RuntimeError("two-stage proposal outputs were not returned")

    topk_record = topk_records[0]
    proposal_scores = outputs.enc_outputs_class.max(-1).values
    recomputed_values, recomputed_indices = torch.topk(
        proposal_scores, args.expected_topk, dim=1, largest=True, sorted=True
    )
    recomputed_scores_np = proposal_scores.detach().float().cpu().numpy()
    recomputed_values_np = recomputed_values.detach().float().cpu().numpy()
    recomputed_indices_np = recomputed_indices.detach().cpu().numpy().astype(np.int64)
    if not np.array_equal(topk_record["input"], recomputed_scores_np):
        raise RuntimeError("captured topk input does not exactly match enc_outputs_class.max(-1)")
    if not np.array_equal(topk_record["values"], recomputed_values_np):
        raise RuntimeError("captured and recomputed topk values differ")
    if not np.array_equal(topk_record["indices"], recomputed_indices_np):
        raise RuntimeError("captured and recomputed topk indices differ")

    arrays: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    backbone_out, backbone_positions = captures["backbone"]
    for index, ((feature, mask), position) in enumerate(
        zip(backbone_out, backbone_positions, strict=True)
    ):
        _store_sample(
            arrays, summaries, f"backbone_level_{index}_feature", feature, args.sample_size
        )
        _store_sample(
            arrays, summaries, f"backbone_level_{index}_mask", mask, args.sample_size
        )
        _store_sample(
            arrays,
            summaries,
            f"backbone_level_{index}_position",
            position,
            args.sample_size,
        )
    for index, projected in sorted(captures["projected"].items(), key=lambda row: int(row[0])):
        _store_sample(
            arrays,
            summaries,
            f"projected_level_{index}_feature",
            projected,
            args.sample_size,
        )

    encoder_output = captures["encoder"]
    encoder_vision = encoder_output.last_hidden_state_vision
    encoder_text = encoder_output.last_hidden_state_text
    _store_sample(
        arrays,
        summaries,
        "encoder_last_hidden_state_vision",
        encoder_vision,
        args.sample_size,
    )
    _store_sample(
        arrays,
        summaries,
        "encoder_last_hidden_state_text",
        encoder_text,
        args.sample_size,
    )

    _store_sample(
        arrays,
        summaries,
        "encoder_proposal_class_logits",
        outputs.enc_outputs_class,
        args.sample_size,
    )
    arrays["encoder_proposal_coord_logits"] = (
        outputs.enc_outputs_coord_logits.detach().float().cpu().numpy()
    )
    arrays["encoder_proposal_boxes"] = (
        outputs.enc_outputs_coord_logits.sigmoid().detach().float().cpu().numpy()
    )
    arrays["encoder_proposal_scores"] = topk_record["input"]
    arrays["topk_values"] = topk_record["values"]
    arrays["topk_indices"] = topk_record["indices"]
    arrays["topk_selected_class_logits"] = (
        torch.gather(
            outputs.enc_outputs_class,
            1,
            recomputed_indices.unsqueeze(-1).expand(
                -1, -1, outputs.enc_outputs_class.shape[-1]
            ),
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    selected_coord_logits = torch.gather(
        outputs.enc_outputs_coord_logits,
        1,
        recomputed_indices.unsqueeze(-1).expand(-1, -1, 4),
    )
    arrays["topk_selected_coord_logits"] = (
        selected_coord_logits.detach().float().cpu().numpy()
    )
    arrays["topk_selected_boxes"] = (
        selected_coord_logits.sigmoid().detach().float().cpu().numpy()
    )
    arrays["final_logits"] = outputs.logits.detach().float().cpu().numpy()
    arrays["final_pred_boxes"] = outputs.pred_boxes.detach().float().cpu().numpy()
    summaries["encoder_proposal_coord_logits"] = _tensor_summary(
        outputs.enc_outputs_coord_logits
    )
    summaries["encoder_proposal_scores"] = _tensor_summary(proposal_scores)
    summaries["final_logits"] = _tensor_summary(outputs.logits)
    summaries["final_pred_boxes"] = _tensor_summary(outputs.pred_boxes)

    boundary_arrays, boundary_rows = _boundary_diagnostics(
        proposal_scores,
        recomputed_indices,
        recomputed_values,
        boundary_window=args.boundary_window,
    )
    arrays.update(boundary_arrays)

    project = Path(__file__).resolve().parent.parent
    baseline_manifest = Path(args.baseline_manifest) if args.baseline_manifest else None
    if baseline_manifest is not None and not baseline_manifest.is_file():
        raise FileNotFoundError(baseline_manifest)
    input_file_sha256 = _sha256_file(inputs_path)
    model_artifact_hashes = (
        None if args.skip_model_artifact_hash else _model_artifact_hashes(model_dir)
    )
    device = torch.cuda.current_device()
    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "gdino_topk_stage_probe",
        "verdict_scope": "diagnostic_only_no_backend_comparison",
        "precision": "eager_fp32",
        "source_contract": source_contract,
        "model": {
            "directory": str(model_dir),
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "core_class": f"{type(core).__module__}.{type(core).__qualname__}",
            "config": {
                "two_stage": bool(core.config.two_stage),
                "num_queries": int(core.config.num_queries),
                "num_feature_levels": int(core.config.num_feature_levels),
                "d_model": int(core.config.d_model),
            },
            "artifact_hashes": model_artifact_hashes,
        },
        "inputs": {
            "path": str(inputs_path),
            "file_sha256": input_file_sha256,
            "arrays": _input_manifest(frozen),
            "baseline_manifest": (
                {
                    "path": str(baseline_manifest),
                    "sha256": _sha256_file(baseline_manifest),
                }
                if baseline_manifest is not None
                else None
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "torch_tensorrt": _package_version("torch-tensorrt"),
            "onnxruntime": _package_version("onnxruntime"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "tf32_policy": args.tf32_policy,
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
                if hasattr(torch, "get_float32_matmul_precision")
                else None
            ),
        },
        "code": {
            "commit": args.code_commit or _git_commit(project),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "hooks": {
            "backbone_module": backbone_name,
            "encoder_module": encoder_name,
            "projected_level_calls": sorted(captures["projected"], key=int),
        },
        "topk": {
            "captured_call_count": len(topk_records),
            "k": args.expected_topk,
            "dim": 1,
            "largest": topk_record["largest"],
            "sorted": topk_record["sorted"],
            "input_dtype": topk_record["input_dtype"],
            "capture_matches_recomputed_scores_bit_exact": True,
            "capture_matches_recomputed_values_bit_exact": True,
            "capture_matches_recomputed_indices_bit_exact": True,
            "boundary_window": args.boundary_window,
            "per_batch_boundary": boundary_rows,
        },
        "tensor_summaries": summaries,
    }
    if args.compare_to:
        report["comparison"] = _compare_capture(
            Path(args.compare_to),
            arrays,
            input_sha256=input_file_sha256,
            model_combined_sha256=(
                model_artifact_hashes["combined_sha256"]
                if model_artifact_hashes is not None
                else None
            ),
        )
        report["verdict_scope"] = "paired_capture_diagnostic"
    _write_artifacts(
        output_dir, arrays, report, max_npz_bytes=args.max_npz_bytes
    )
    print(json.dumps({"output_dir": str(output_dir), "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

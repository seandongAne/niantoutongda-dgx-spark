#!/usr/bin/env python
"""从 ingest 的 Top-K prototype crops 生成逐视角 DINO 嵌入产物。

脚本不读取人工 GT。NPZ 和 manifest 都先写同目录临时文件再原子替换；manifest
是完成标记，消费者应同时校验其中记录的 NPZ SHA-256。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from backend.tools.reid.multiview import ARTIFACT_FORMAT_VERSION

FORMAT_VERSION = ARTIFACT_FORMAT_VERSION


class BatchEmbedder(Protocol):
    model_version: str

    def embed_many(
        self, image_paths: list[str], *, batch_size: int = 32
    ) -> list[list[float]]: ...


@dataclass(frozen=True)
class PrototypeRecord:
    tracklet_id: str
    view_index: int
    crop_ref: str
    crop_path: Path


@dataclass(frozen=True)
class MissingPrototype:
    tracklet_id: str
    view_index: int
    crop_ref: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_crop(ref: str, ingest_root: Path) -> Path | None:
    path = Path(ref)
    if path.is_absolute():
        return path.resolve() if path.is_file() else None

    candidates = [Path.cwd() / path, ingest_root / path]
    candidates.extend(parent / path for parent in ingest_root.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_prototype_records(
    ingest_root: str | Path,
) -> tuple[list[PrototypeRecord], list[MissingPrototype], list[dict]]:
    """读取并稳定排序所有 tracklets.jsonl 中的 prototype_refs。"""

    root = Path(ingest_root).resolve()
    tracklet_paths = sorted(root.rglob("tracklets.jsonl"))
    if not tracklet_paths:
        raise FileNotFoundError(f"no tracklets.jsonl under {root}")

    records: list[PrototypeRecord] = []
    missing: list[MissingPrototype] = []
    sources: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for path in tracklet_paths:
        row_count = 0
        ref_count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row_count += 1
                row = json.loads(line)
                tracklet_id = row.get("tracklet_id")
                refs = row.get("prototype_refs") or []
                if not isinstance(tracklet_id, str) or not tracklet_id:
                    raise ValueError(f"{path}:{line_number}: invalid tracklet_id")
                if not isinstance(refs, list) or not all(
                    isinstance(ref, str) and ref for ref in refs
                ):
                    raise ValueError(f"{path}:{line_number}: invalid prototype_refs")
                for view_index, crop_ref in enumerate(refs):
                    key = (tracklet_id, view_index)
                    if key in seen:
                        raise ValueError(
                            f"duplicate tracklet/view key: {tracklet_id}[{view_index}]"
                        )
                    seen.add(key)
                    ref_count += 1
                    resolved = _resolve_crop(crop_ref, root)
                    if resolved is None:
                        missing.append(
                            MissingPrototype(tracklet_id, view_index, crop_ref)
                        )
                    else:
                        records.append(
                            PrototypeRecord(
                                tracklet_id=tracklet_id,
                                view_index=view_index,
                                crop_ref=crop_ref,
                                crop_path=resolved,
                            )
                        )
        try:
            display_path = path.relative_to(root).as_posix()
        except ValueError:
            display_path = path.name
        sources.append(
            {
                "path": display_path,
                "sha256": sha256_file(path),
                "tracklet_count": row_count,
                "prototype_ref_count": ref_count,
            }
        )

    records.sort(key=lambda item: (item.tracklet_id, item.view_index, item.crop_ref))
    missing.sort(key=lambda item: (item.tracklet_id, item.view_index, item.crop_ref))
    return records, missing, sources


def input_sha256(
    records: Sequence[PrototypeRecord], missing: Sequence[MissingPrototype]
) -> str:
    """聚合轨迹/视角身份及 crop 字节，避免路径变化影响输入指纹。"""

    digest = hashlib.sha256()
    for record in records:
        crop_hash = sha256_file(record.crop_path)
        payload = (
            f"present\0{record.tracklet_id}\0{record.view_index}\0"
            f"{record.crop_ref}\0{crop_hash}\n"
        )
        digest.update(payload.encode("utf-8"))
    for item in missing:
        payload = (
            f"missing\0{item.tracklet_id}\0{item.view_index}\0{item.crop_ref}\n"
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def embed_records(
    records: Sequence[PrototypeRecord],
    embedder: BatchEmbedder,
    *,
    batch_size: int,
) -> np.ndarray:
    """按稳定记录顺序批量嵌入，并再次强制 float32/L2 归一化。"""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not records:
        raise ValueError("no existing prototype crops to embed")

    batches: list[np.ndarray] = []
    dimension: int | None = None
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        raw = embedder.embed_many(
            [str(record.crop_path) for record in batch],
            batch_size=len(batch),
        )
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(batch):
            raise ValueError(
                f"embedder returned {tuple(values.shape)} for batch of {len(batch)}"
            )
        if dimension is None:
            dimension = int(values.shape[1])
            if dimension <= 0:
                raise ValueError("embedder returned zero-dimensional vectors")
        elif values.shape[1] != dimension:
            raise ValueError(
                f"embedding dimension changed from {dimension} to {values.shape[1]}"
            )
        if not np.isfinite(values).all():
            raise ValueError("embedder returned non-finite vectors")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise ValueError("embedder returned a zero vector")
        batches.append((values / norms).astype(np.float32, copy=False))
    return np.concatenate(batches, axis=0)


def _atomic_save_npz(
    target: Path,
    records: Sequence[PrototypeRecord],
    vectors: np.ndarray,
    *,
    model_version: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            max_id_length = max(1, *(len(record.tracklet_id) for record in records))
            np.savez_compressed(
                handle,
                format_version=np.asarray(FORMAT_VERSION),
                model_version=np.asarray(model_version),
                tracklet_ids=np.asarray(
                    [record.tracklet_id for record in records],
                    dtype=f"<U{max_id_length}",
                ),
                view_index=np.asarray(
                    [record.view_index for record in records], dtype=np.uint16
                ),
                vectors=np.asarray(vectors, dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json(target: Path, payload: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_artifact(
    output_path: str | Path,
    manifest_path: str | Path,
    records: Sequence[PrototypeRecord],
    vectors: np.ndarray,
    *,
    model_version: str,
    source_files: Sequence[dict],
    missing: Sequence[MissingPrototype],
    inputs_sha256: str,
) -> dict:
    output = Path(output_path)
    manifest = Path(manifest_path)
    if vectors.shape[0] != len(records):
        raise ValueError("record/vector count mismatch")
    _atomic_save_npz(output, records, vectors, model_version=model_version)
    output_hash = sha256_file(output)
    payload = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "artifact": {
            "path": output.name,
            "sha256": output_hash,
            "view_count": len(records),
            "tracklet_count": len({record.tracklet_id for record in records}),
            "embedding_dim": int(vectors.shape[1]),
            "vector_dtype": "float32",
            "view_index_base": 0,
        },
        "inputs_sha256": inputs_sha256,
        "source_files": list(source_files),
        "missing_prototype_count": len(missing),
        "missing_prototypes": [
            {
                "tracklet_id": item.tracklet_id,
                "view_index": item.view_index,
                "crop_ref": item.crop_ref,
            }
            for item in missing
        ],
    }
    _atomic_write_json(manifest, payload)
    return payload


def _valid_existing_artifact(
    output: Path,
    manifest: Path,
    *,
    model_version: str,
    inputs_sha256: str,
) -> dict | None:
    if not output.is_file() or not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != FORMAT_VERSION
            or payload.get("model_version") != model_version
            or payload.get("inputs_sha256") != inputs_sha256
            or payload.get("artifact", {}).get("sha256") != sha256_file(output)
        ):
            return None
        with np.load(output, allow_pickle=False) as data:
            if str(data["format_version"].item()) != FORMAT_VERSION:
                return None
        return payload
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True, help="输出 .npz 路径")
    parser.add_argument("--manifest", help="默认与 --out 同名的 .manifest.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="任一 prototype crop 缺失即失败；默认跳过并写入 manifest",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    output = Path(args.out)
    manifest = Path(args.manifest) if args.manifest else output.with_suffix(
        ".manifest.json"
    )
    records, missing, sources = load_prototype_records(args.ingest_root)
    if args.strict_missing and missing:
        raise FileNotFoundError(f"{len(missing)} prototype crops are missing")
    inputs_hash = input_sha256(records, missing)
    model_version = f"dinov2-base@{args.model_dir}"

    if not args.force:
        cached = _valid_existing_artifact(
            output,
            manifest,
            model_version=model_version,
            inputs_sha256=inputs_hash,
        )
        if cached is not None:
            print(json.dumps({"status": "reused", **cached["artifact"]}, sort_keys=True))
            return 0

    from backend.pipeline.embed import Dinov2Embedder

    embedder = Dinov2Embedder(args.model_dir, device=args.device)
    vectors = embed_records(records, embedder, batch_size=args.batch_size)
    payload = write_artifact(
        output,
        manifest,
        records,
        vectors,
        model_version=embedder.model_version,
        source_files=sources,
        missing=missing,
        inputs_sha256=inputs_hash,
    )
    print(json.dumps({"status": "written", **payload["artifact"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

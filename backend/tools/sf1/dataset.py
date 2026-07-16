"""从冻结的 anchor→tracklet 标注与 DINOv2 嵌入构建无泄漏 SF1 切分。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.schemas.core import Tracklet


@dataclass(frozen=True)
class SF1Sample:
    tracklet_id: str
    video_id: str
    identity_id: str
    vector: np.ndarray


@dataclass(frozen=True)
class SF1Split:
    train: tuple[SF1Sample, ...]
    validation: tuple[SF1Sample, ...]
    validation_video_by_identity: dict[str, str]
    policy: str = "leave_last_video_out_v1"

    def manifest(self) -> dict:
        def rows(samples: tuple[SF1Sample, ...]) -> list[dict[str, str]]:
            return [
                {
                    "tracklet_id": sample.tracklet_id,
                    "video_id": sample.video_id,
                    "identity_id": sample.identity_id,
                }
                for sample in samples
            ]

        train_ids = {sample.tracklet_id for sample in self.train}
        validation_ids = {sample.tracklet_id for sample in self.validation}
        return {
            "policy": self.policy,
            "leakage_check": {
                "tracklet_overlap": sorted(train_ids & validation_ids),
                "pass": not bool(train_ids & validation_ids),
            },
            "validation_video_by_identity": dict(
                sorted(self.validation_video_by_identity.items())
            ),
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "identities": len({sample.identity_id for sample in self.train}),
            },
            "train": rows(self.train),
            "validation": rows(self.validation),
        }


def _resolve_ref(ref: str, ingest_root: Path) -> Path:
    path = Path(ref)
    if path.is_absolute() and path.exists():
        return path
    for base in (Path.cwd(), ingest_root, *ingest_root.parents):
        candidate = base / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve artifact reference: {ref}")


def _load_vector(path: Path, input_dim: int) -> np.ndarray:
    raw = json.loads(path.read_text(encoding="utf-8"))
    vector = np.asarray(raw["vector"], dtype=np.float32)
    if vector.shape != (input_dim,):
        raise ValueError(f"{path}: expected {input_dim} dims, got {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"{path}: invalid embedding norm {norm}")
    return vector / norm


def _label_index(labels_path: Path) -> dict[str, tuple[str, str]]:
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    index: dict[str, tuple[str, str]] = {}
    for entity in raw.get("entities", []):
        identity_id = str(entity["anchor_id"])
        for video_id, tracklet_ids in entity.get(
            "confirmed_tracklet_ids_by_video", {}
        ).items():
            for tracklet_id in tracklet_ids:
                if tracklet_id in index:
                    raise ValueError(f"duplicate labeled tracklet: {tracklet_id}")
                index[str(tracklet_id)] = (identity_id, str(video_id))
    if not index:
        raise ValueError(f"{labels_path}: no confirmed tracklet labels")
    return index


def load_labeled_samples(
    ingest_root: str | Path,
    labels_path: str | Path,
    *,
    input_dim: int,
    require_all_labels: bool = True,
) -> list[SF1Sample]:
    root = Path(ingest_root)
    labels = _label_index(Path(labels_path))
    samples: list[SF1Sample] = []
    seen: set[str] = set()
    for tracklet_path in sorted(root.glob("*/tracklets.jsonl")):
        for line in tracklet_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            tracklet = Tracklet.model_validate_json(line)
            labeled = labels.get(tracklet.tracklet_id)
            if labeled is None:
                continue
            identity_id, expected_video = labeled
            if tracklet.video_id != expected_video:
                raise ValueError(
                    f"{tracklet.tracklet_id}: label video {expected_video} != data {tracklet.video_id}"
                )
            if not tracklet.embedding_ref:
                raise ValueError(f"{tracklet.tracklet_id}: missing embedding_ref")
            samples.append(
                SF1Sample(
                    tracklet_id=tracklet.tracklet_id,
                    video_id=tracklet.video_id,
                    identity_id=identity_id,
                    vector=_load_vector(
                        _resolve_ref(tracklet.embedding_ref, root), input_dim
                    ),
                )
            )
            seen.add(tracklet.tracklet_id)
    missing = sorted(set(labels) - seen)
    if require_all_labels and missing:
        raise ValueError(
            f"{len(missing)} labeled tracklets are absent from ingest root: {missing[:8]}"
        )
    samples.sort(key=lambda item: item.tracklet_id)
    if not samples:
        raise ValueError("no labeled embeddings found")
    return samples


def build_leave_last_video_out_split(samples: list[SF1Sample]) -> SF1Split:
    """每个身份按 video_id 排序，最后一个视频只作验证 query。"""

    by_identity: dict[str, list[SF1Sample]] = {}
    for sample in samples:
        by_identity.setdefault(sample.identity_id, []).append(sample)
    train: list[SF1Sample] = []
    validation: list[SF1Sample] = []
    validation_video_by_identity: dict[str, str] = {}
    for identity_id in sorted(by_identity):
        group = sorted(by_identity[identity_id], key=lambda item: item.tracklet_id)
        videos = sorted({item.video_id for item in group})
        if len(videos) < 2:
            raise ValueError(
                f"{identity_id}: leave-video-out requires at least two videos"
            )
        held_out = videos[-1]
        validation_video_by_identity[identity_id] = held_out
        train.extend(item for item in group if item.video_id != held_out)
        validation.extend(item for item in group if item.video_id == held_out)
    train.sort(key=lambda item: item.tracklet_id)
    validation.sort(key=lambda item: item.tracklet_id)
    train_counts: dict[str, int] = {}
    for item in train:
        train_counts[item.identity_id] = train_counts.get(item.identity_id, 0) + 1
    too_small = {identity: count for identity, count in train_counts.items() if count < 2}
    if too_small:
        raise ValueError(f"identities need >=2 training samples for SupCon: {too_small}")
    return SF1Split(
        train=tuple(train),
        validation=tuple(validation),
        validation_video_by_identity=validation_video_by_identity,
    )


def dataset_fingerprint(samples: list[SF1Sample]) -> str:
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda item: item.tracklet_id):
        digest.update(sample.tracklet_id.encode())
        digest.update(b"\0")
        digest.update(sample.video_id.encode())
        digest.update(b"\0")
        digest.update(sample.identity_id.encode())
        digest.update(b"\0")
        digest.update(np.asarray(sample.vector, dtype="<f4").tobytes())
    return digest.hexdigest()

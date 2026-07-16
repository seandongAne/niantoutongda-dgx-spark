"""SF1-L1 的无泄漏切分、权重校验与 ReID 推理接线。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.pipeline.vocab import Vocabulary, VocabularyEntry
from backend.schemas.core import Observation, Tracklet
from backend.tools.reid.model import ProjectionConfig, load_features
from backend.tools.sf1.dataset import (
    SF1Sample,
    build_leave_last_video_out_split,
    load_labeled_samples,
)
from backend.tools.sf1.metrics import retrieval_metrics
from backend.tools.sf1.projection import NumpyProjectionHead, sha256_file

PROJ = Path(__file__).resolve().parent.parent.parent


def _identity_head() -> NumpyProjectionHead:
    return NumpyProjectionHead(
        weight1=np.eye(2, dtype=np.float32),
        bias1=np.zeros(2, dtype=np.float32),
        weight2=np.eye(2, dtype=np.float32),
        bias2=np.zeros(2, dtype=np.float32),
    )


def test_projection_roundtrip_normalizes_and_checks_hash(tmp_path):
    path = tmp_path / "projection.npz"
    _identity_head().save(path)
    digest = sha256_file(path)
    loaded = NumpyProjectionHead.load(path, expected_sha256=digest)
    projected = loaded.apply(np.asarray([3.0, 4.0], dtype=np.float32))
    assert projected.tolist() == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError, match="sha256 mismatch"):
        NumpyProjectionHead.load(path, expected_sha256="0" * 64)


def test_zero_initialized_residual_head_is_identity_mapping(tmp_path):
    head = NumpyProjectionHead(
        weight1=np.eye(2, dtype=np.float32),
        bias1=np.zeros(2, dtype=np.float32),
        weight2=np.zeros((2, 2), dtype=np.float32),
        bias2=np.zeros(2, dtype=np.float32),
        mode="residual",
        residual_scale=0.25,
    )
    path = tmp_path / "residual.npz"
    head.save(path)
    loaded = NumpyProjectionHead.load(path)
    assert loaded.mode == "residual"
    assert loaded.residual_scale == pytest.approx(0.25)
    assert loaded.apply(np.asarray([0.6, 0.8], dtype=np.float32)).tolist() == pytest.approx(
        [0.6, 0.8]
    )


def test_concat_head_preserves_raw_branch_and_adds_learned_branch():
    head = NumpyProjectionHead(
        weight1=np.eye(2, dtype=np.float32),
        bias1=np.zeros(2, dtype=np.float32),
        weight2=np.eye(2, dtype=np.float32),
        bias2=np.zeros(2, dtype=np.float32),
        mode="concat",
        residual_scale=0.4,
    )
    output = head.apply(np.asarray([0.6, 0.8], dtype=np.float32))
    expected = np.asarray([0.6, 0.8, 0.24, 0.32], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert head.output_dim == 4
    assert output.tolist() == pytest.approx(expected.tolist())


def test_leave_video_out_split_has_no_tracklet_or_video_leakage():
    samples = []
    for identity, vector in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
        for tracklet_id, video_id in (
            (f"{identity}_1", "v1"),
            (f"{identity}_2", "v1"),
            (f"{identity}_3", "v2"),
        ):
            samples.append(
                SF1Sample(
                    tracklet_id=tracklet_id,
                    video_id=video_id,
                    identity_id=identity,
                    vector=np.asarray(vector, dtype=np.float32),
                )
            )
    split = build_leave_last_video_out_split(samples)
    assert {sample.video_id for sample in split.train} == {"v1"}
    assert {sample.video_id for sample in split.validation} == {"v2"}
    manifest = split.manifest()
    assert manifest["leakage_check"] == {"tracklet_overlap": [], "pass": True}
    metrics = retrieval_metrics(split.train, split.validation)
    assert metrics["recall_at_1"] == 1.0
    assert metrics["finite"] is True


def _write_ingest(root, artifact):
    video = root / "v1"
    evidence = video / "evidence"
    evidence.mkdir(parents=True)
    embedding = evidence / "t1_emb.json"
    embedding.write_text(json.dumps({"vector": [0.8, 0.6]}), encoding="utf-8")
    observation = Observation(
        observation_id="o1",
        video_id="v1",
        timestamp_ms=0,
        bbox=(0.0, 0.0, 1.0, 1.0),
        crop_ref="crop.jpg",
        quality=0.9,
        model_version="fake",
    )
    tracklet = Tracklet(
        tracklet_id="t1",
        video_id="v1",
        observation_ids=["o1"],
        embedding_ref=str(embedding),
        attributes={"label": "lamp"},
    )
    (video / "observations.jsonl").write_text(
        observation.model_dump_json() + "\n", encoding="utf-8"
    )
    (video / "tracklets.jsonl").write_text(
        tracklet.model_dump_json() + "\n", encoding="utf-8"
    )
    labels = {
        "entities": [
            {
                "anchor_id": "anchor_1",
                "confirmed_tracklet_ids_by_video": {"v1": ["t1"]},
            }
        ]
    }
    artifact.write_text(json.dumps(labels), encoding="utf-8")


def test_labeled_loader_and_reid_projection_hook(tmp_path):
    ingest = tmp_path / "ingest"
    labels = tmp_path / "labels.json"
    _write_ingest(ingest, labels)
    samples = load_labeled_samples(ingest, labels, input_dim=2)
    assert [(item.tracklet_id, item.identity_id) for item in samples] == [
        ("t1", "anchor_1")
    ]

    artifact = tmp_path / "projection.npz"
    _identity_head().save(artifact)
    vocab = Vocabulary(
        (
            VocabularyEntry(
                canonical_id="lamp",
                category_id="lamp",
                display_label_zh="台灯",
                detection_prompts=("lamp",),
            ),
        )
    )
    features = load_features(
        ingest,
        vocab=vocab,
        embedding_dim=2,
        projection=ProjectionConfig(
            enabled=True,
            artifact=str(artifact),
            sha256=sha256_file(artifact),
        ),
    )
    assert features[0].vector == pytest.approx((0.8, 0.6))


def test_wire_cli_derives_config_without_mutating_baseline(tmp_path):
    artifact = tmp_path / "projection.npz"
    _identity_head().save(artifact)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "slice_id": "SF1-L1",
                "projection": {
                    "path": str(artifact),
                    "sha256": sha256_file(artifact),
                },
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base.yaml"
    base_text = """version: test-v1
embedding_dim: 2
top_k: 2
weights: {instance: 1.0, semantic: 0.0, attribute: 0.0, context: 0.0, geometry: 0.0}
thresholds: {match: 0.9, new: 0.6, margin: 0.0, min_quality: 0.0}
"""
    base.write_text(base_text, encoding="utf-8")
    out = tmp_path / "derived.yaml"
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJ / "scripts/sf1_wire_reid.py"),
            "--base",
            str(base),
            "--manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=PROJ,
    )
    assert proc.returncode == 0, proc.stderr
    assert base.read_text(encoding="utf-8") == base_text
    derived = out.read_text(encoding="utf-8")
    assert "enabled: true" in derived
    assert sha256_file(artifact) in derived

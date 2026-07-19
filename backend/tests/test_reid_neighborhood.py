from __future__ import annotations

import json

import pytest

from backend.tools.reid.neighborhood import (
    CONTEXT_FORMAT_VERSION,
    FrameDetection,
    GeometryObservation,
    NeighborRelation,
    NeighborhoodSignature,
    TrackGeometry,
    build_frame_index,
    build_signatures,
    collapse_stitched_geometries,
    compare_signatures,
    select_anchor_categories,
)
from backend.tools.reid.model import (
    NeighborhoodConfig,
    NeighborhoodPairEvidence,
    ReIDConfig,
    ThresholdConfig,
    TrackFeature,
    WeightConfig,
    load_neighborhood_evidence,
)
from backend.schemas.core import Tracklet
from backend.tools.reid.matcher import (
    IdentityConstraints,
    PairScore,
    _pairwise_assignments,
    _rerank_neighborhood_scores,
)
from backend.tools.sf1.projection import sha256_file


def _geometry(tracklet_id, video, category, rows):
    return TrackGeometry(
        tracklet_id=tracklet_id,
        video_id=video,
        category_id=category,
        observations=tuple(
            GeometryObservation(timestamp_ms=timestamp, bbox=bbox, quality=quality)
            for timestamp, bbox, quality in rows
        ),
    )


def test_anchor_selection_rejects_multi_instance_category():
    index = {}
    for video in ("v1", "v2", "v3"):
        for timestamp in (0, 1):
            index[(video, timestamp)] = (
                FrameDetection("lamp", "lamp", (0, 0, 10, 10), 0.9),
                FrameDetection("book-a", "book", (20, 0, 30, 10), 0.9),
                FrameDetection("book-b", "book", (50, 0, 60, 10), 0.8),
            )

    anchors, diagnostics = select_anchor_categories(
        index,
        videos=("v1", "v2", "v3"),
        min_visible_frames=2,
        min_single_fraction=1.0,
    )

    assert anchors == frozenset({"lamp"})
    assert diagnostics["book"]["v1_single_fraction"] == 0.0


def test_signature_excludes_overlapping_false_anchor_and_keeps_neighbors():
    geometries = {
        "cup": _geometry(
            "cup", "v1", "mug", [(0, (40, 40, 60, 60), 1.0), (1, (40, 40, 60, 60), 1.0)]
        ),
        "fake": _geometry(
            "fake", "v1", "lamp", [(0, (40, 40, 60, 60), 0.9), (1, (40, 40, 60, 60), 0.9)]
        ),
        "real": _geometry(
            "real", "v1", "book", [(0, (70, 40, 90, 60), 0.9), (1, (70, 40, 90, 60), 0.9)]
        ),
    }
    index = build_frame_index(geometries)

    signatures = build_signatures(
        geometries,
        index,
        anchor_categories=frozenset({"lamp", "book"}),
        frame_sizes={"v1": (100, 100)},
        max_neighbors=3,
        min_common_frames=2,
    )

    assert [row.anchor_category for row in signatures["cup"].relations] == ["book"]


def test_same_topology_scores_above_changed_topology():
    def signature(tracklet_id, video, anchors):
        return NeighborhoodSignature(
            tracklet_id=tracklet_id,
            video_id=video,
            relations=tuple(
                NeighborRelation(anchor, visibility, distance, rank, 3)
                for anchor, visibility, distance, rank in anchors
            ),
        )

    left = signature(
        "a", "v1", (("lamp", 1.0, 0.10, 0.0), ("book", 0.8, 0.20, 1.0))
    )
    same = signature(
        "b", "v2", (("lamp", 0.9, 0.12, 0.0), ("book", 0.7, 0.21, 1.0))
    )
    changed = signature(
        "c", "v3", (("lamp", 0.2, 0.45, 1.0), ("book", 0.2, 0.50, 0.0))
    )

    same_score = compare_signatures(left, same, min_shared_anchors=2)
    changed_score = compare_signatures(left, changed, min_shared_anchors=2)

    assert same_score is not None and changed_score is not None
    assert same_score.score > changed_score.score
    assert compare_signatures(left, same, min_shared_anchors=3) is None


def test_stitch_geometry_is_collapsed_without_removing_originals():
    geometries = {
        "v1_t1": _geometry("v1_t1", "v1", "mug", [(0, (0, 0, 1, 1), 0.8)]),
        "v1_t2": _geometry("v1_t2", "v1", "mug", [(1, (0, 0, 1, 1), 0.9)]),
    }

    collapsed = collapse_stitched_geometries(geometries, {"v1_t1": ("v1_t1", "v1_t2")})

    assert set(collapsed) == {"v1_t1", "v1_t2"}
    assert [row.timestamp_ms for row in collapsed["v1_t1"].observations] == [0, 1]


def _reid_config(neighborhood):
    return ReIDConfig(
        version="test-neighborhood-v1",
        embedding_dim=2,
        top_k=1,
        weights=WeightConfig(1.0, 0.0, 0.0, 0.0, 0.0),
        thresholds=ThresholdConfig(0.8, 0.6, 0.0, 0.0),
        neighborhood=neighborhood,
    )


def test_neighborhood_sidecar_loader_validates_hash_and_schema(tmp_path):
    artifact = tmp_path / "context.jsonl"
    rows = [
        {
            "schema_version": CONTEXT_FORMAT_VERSION,
            "tracklet_a": "v1_t1",
            "tracklet_b": "v2_t1",
            "score": 0.8,
            "shared_anchors": ["book", "lamp"],
            "overlap": 0.7,
            "relation_agreement": 0.9,
        },
        {
            "schema_version": CONTEXT_FORMAT_VERSION,
            "tracklet_a": "v1_t1",
            "tracklet_b": "v2_t2",
            "score": None,
            "shared_anchors": [],
            "overlap": None,
            "relation_agreement": None,
        },
    ]
    artifact.write_text("".join(json.dumps(row) + "\n" for row in rows))

    loaded = load_neighborhood_evidence(
        artifact.name, tmp_path, expected_sha256=sha256_file(artifact)
    )

    assert loaded[("v1_t1", "v2_t1")].shared_anchors == ("book", "lamp")
    assert loaded[("v1_t1", "v2_t2")].score is None
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_neighborhood_evidence(
            artifact.name, tmp_path, expected_sha256="0" * 64
        )


def test_neighborhood_config_and_reranker_fail_closed():
    with pytest.raises(ValueError, match="requires artifact"):
        _reid_config(NeighborhoodConfig(enabled=True)).validate()
    config = _reid_config(
        NeighborhoodConfig(
            enabled=True,
            artifact="context.jsonl",
            sha256="hash",
            blend=1.0,
        )
    )
    config.validate()
    scores = {
        ("v1_t1", "v2_t1"): PairScore(
            "v1_t1", "v2_t1", 0.71, 1.0, None, 0.5, 1.0, 0.71
        ),
        ("v1_t1", "v2_t2"): PairScore(
            "v1_t1", "v2_t2", 0.79, 1.0, None, 0.5, 1.0, 0.79
        ),
        ("v1_t1", "v2_t3"): PairScore(
            "v1_t1", "v2_t3", 0.75, 1.0, None, 0.5, 1.0, 0.75
        ),
    }
    evidence = {
        ("v1_t1", "v2_t1"): NeighborhoodPairEvidence(
            0.9, ("book", "lamp"), 0.8, 1.0
        ),
        ("v1_t1", "v2_t2"): NeighborhoodPairEvidence(
            0.1, ("book", "lamp"), 0.2, 0.0
        ),
        ("v1_t1", "v2_t3"): NeighborhoodPairEvidence(score=None),
    }

    reranked = _rerank_neighborhood_scores(scores, evidence, config)

    assert reranked[("v1_t1", "v2_t1")].total == 0.79
    assert reranked[("v1_t1", "v2_t2")].total == 0.71
    assert reranked[("v1_t1", "v2_t3")].total == 0.75
    assert reranked[("v1_t1", "v2_t3")].context_calibrated is None
    with pytest.raises(ValueError, match="misses 2 recalled pairs"):
        _rerank_neighborhood_scores(scores, {next(iter(evidence)): next(iter(evidence.values()))}, config)


def test_neighborhood_locks_baseline_high_confidence_hungarian_edge(monkeypatch):
    def feature(tracklet_id, video):
        return TrackFeature(
            tracklet=Tracklet(
                tracklet_id=tracklet_id,
                video_id=video,
                observation_ids=[],
                attributes={"label": "item"},
            ),
            vector=(1.0, 0.0),
            raw_label="item",
            canonical_id="item",
            category_id="item",
            quality=1.0,
            aspect_ratio=1.0,
            area=1.0,
        )

    features = [feature("v1_a", "v1"), feature("v1_b", "v1"), feature("v2_a", "v2"), feature("v2_b", "v2")]
    totals = {
        ("v1_a", "v2_a"): 0.83,
        ("v1_a", "v2_b"): 0.81,
        ("v1_b", "v2_a"): 0.70,
        ("v1_b", "v2_b"): 0.80,
    }

    def fake_score_pair(a, b, _config, _constraints):
        total = totals[(a.tracklet_id, b.tracklet_id)]
        return PairScore(
            a.tracklet_id, b.tracklet_id, total, 1.0, None, 0.5, 1.0, total
        )

    monkeypatch.setattr("backend.tools.reid.matcher.score_pair", fake_score_pair)
    config = ReIDConfig(
        version="test-neighborhood-lock-v1",
        embedding_dim=2,
        top_k=2,
        weights=WeightConfig(1.0, 0.0, 0.0, 0.0, 0.0),
        thresholds=ThresholdConfig(0.82, 0.60, 0.0, 0.0),
        neighborhood=NeighborhoodConfig(
            enabled=True,
            artifact="unused",
            sha256="unused",
            blend=1.0,
        ),
    )
    evidence = {
        ("v1_a", "v2_a"): NeighborhoodPairEvidence(0.5, ("x", "y", "z")),
        ("v1_a", "v2_b"): NeighborhoodPairEvidence(0.9, ("x", "y", "z")),
        ("v1_b", "v2_a"): NeighborhoodPairEvidence(0.8, ("x", "y", "z")),
        ("v1_b", "v2_b"): NeighborhoodPairEvidence(0.1, ("x", "y", "z")),
    }

    accepted, _, candidates = _pairwise_assignments(
        features, config, IdentityConstraints(), evidence
    )

    assert [(score.a, score.b) for score in accepted] == [("v1_a", "v2_a")]
    assert accepted[0].baseline_high_confidence_locked is True
    locked = next(
        row for row in candidates if row["tracklet_a"] == "v1_a" and row["tracklet_b"] == "v2_a"
    )
    assert locked["assigned"] is True
    assert locked["baseline_high_confidence_locked"] is True

"""S3 跨视频匹配的纯本地合成测试。"""

import json
from pathlib import Path

from backend.schemas.core import IdentityState, Observation, Tracklet
from backend.pipeline.vocab import Vocabulary, VocabularyEntry
from backend.tools.reid.assignment import maximise_assignment
from backend.tools.reid.matcher import (
    IdentityConstraints,
    PairScore,
    _UnionFind,
    _cycle_complete_components,
    _pairwise_assignments,
    run_reid,
    score_pair,
)
from backend.tools.reid.model import (
    CycleConfig,
    ReIDConfig,
    ThresholdConfig,
    TrackFeature,
    WeightConfig,
)


def _config() -> ReIDConfig:
    return ReIDConfig(
        version="test-reid-v1",
        embedding_dim=2,
        top_k=2,
        weights=WeightConfig(instance=0.85, semantic=0.15, attribute=0, context=0, geometry=0),
        thresholds=ThresholdConfig(match=0.95, new=0.60, margin=0.10, min_quality=0.2),
    )


def _cycle_config() -> ReIDConfig:
    return ReIDConfig(
        version="test-cycle-v1",
        embedding_dim=2,
        top_k=2,
        weights=WeightConfig(instance=1, semantic=0, attribute=0, context=0, geometry=0),
        thresholds=ThresholdConfig(match=0.90, new=0.60, margin=0, min_quality=0),
        cycle=CycleConfig(enabled=True),
    )


def _feature(tracklet_id, video_id, label, vector, *, category, canonical=None, quality=0.9):
    tracklet = Tracklet(
        tracklet_id=tracklet_id,
        video_id=video_id,
        observation_ids=[f"{tracklet_id}_o1"],
        prototype_refs=[f"{tracklet_id}.jpg"],
        attributes={"label": label},
    )
    return TrackFeature(
        tracklet=tracklet,
        vector=tuple(vector),
        raw_label=label,
        canonical_id=canonical or label,
        category_id=category,
        quality=quality,
        aspect_ratio=1.0,
        area=100.0,
    )


def _candidate(a: str, b: str, score: float, *, assigned: bool) -> dict:
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "score": score,
        "assigned": assigned,
    }


def _cycle_run(features, candidates, seed_pairs, constraints=None):
    feature_by_id = {feature.tracklet_id: feature for feature in features}
    union_find = _UnionFind(sorted(feature_by_id))
    for a, b in seed_pairs:
        union_find.union(a, b)
    records = _cycle_complete_components(
        union_find,
        feature_by_id,
        constraints or IdentityConstraints(),
        candidates,
        _cycle_config(),
    )
    clusters = sorted(
        tuple(sorted(union_find.members(tracklet_id)))
        for tracklet_id in feature_by_id
        if union_find.find(tracklet_id) == tracklet_id
    )
    return records, clusters


def test_hungarian_finds_global_optimum_and_allows_unmatched():
    assignment = maximise_assignment([[0.90, 0.80], [0.85, 0.10]], unmatched_weight=0.0)
    assert assignment == [1, 0]  # 0.80 + 0.85 > 0.90 + 0.10
    assert maximise_assignment([[0.2]], unmatched_weight=0.5) == [None]


def test_candidates_are_deduplicated_symmetric_top_k_union(monkeypatch):
    """candidates.jsonl must mirror the symmetric recall set used by assignment."""

    config = ReIDConfig(
        version="test-symmetric-candidates-v1",
        embedding_dim=2,
        top_k=1,
        weights=WeightConfig(instance=1, semantic=0, attribute=0, context=0, geometry=0),
        thresholds=ThresholdConfig(match=0.90, new=0.60, margin=0, min_quality=0),
    )
    features = [
        _feature("v1_a", "v1", "item", [1, 0], category="item"),
        _feature("v1_b", "v1", "item", [0, 1], category="item"),
        _feature("v2_a", "v2", "item", [1, 0], category="item"),
        _feature("v2_b", "v2", "item", [0, 1], category="item"),
    ]
    totals = {
        ("v1_a", "v2_a"): 0.95,  # mutual Top-1 and assigned
        ("v1_a", "v2_b"): 0.61,  # right-only Top-1, left endpoint prefers v2_a
        ("v1_b", "v2_a"): 0.59,  # left-only Top-1, below the unmatched weight
        ("v1_b", "v2_b"): 0.10,
    }

    def fake_score_pair(a, b, _config, _constraints):
        total = totals[(a.tracklet_id, b.tracklet_id)]
        return PairScore(
            a=a.tracklet_id,
            b=b.tracklet_id,
            instance=total,
            semantic=0,
            attribute=None,
            context=0,
            geometry=0,
            total=total,
        )

    monkeypatch.setattr("backend.tools.reid.matcher.score_pair", fake_score_pair)

    _, _, candidates = _pairwise_assignments(features, config, IdentityConstraints())
    _, _, reordered = _pairwise_assignments(
        list(reversed(features)), config, IdentityConstraints()
    )

    assert candidates == reordered
    assert [
        (row["tracklet_a"], row["tracklet_b"], row["assigned"])
        for row in candidates
    ] == [
        ("v1_a", "v2_a", True),
        ("v1_a", "v2_b", False),
        ("v1_b", "v2_a", False),
    ]
    assert len(candidates) == len(
        {(row["tracklet_a"], row["tracklet_b"]) for row in candidates}
    )


def test_cycle_closes_when_both_spokes_are_assigned_above_new_threshold():
    features = [
        _feature("v1_a", "v1", "item", [1, 0], category="item"),
        _feature("v2_a", "v2", "item", [1, 0], category="item"),
        _feature("v3_x", "v3", "item", [1, 0], category="item"),
    ]
    candidates = [
        _candidate("v1_a", "v3_x", 0.61, assigned=True),
        _candidate("v2_a", "v3_x", 0.62, assigned=True),
    ]

    records, clusters = _cycle_run(features, candidates, [("v1_a", "v2_a")])

    assert clusters == [("v1_a", "v2_a", "v3_x")]
    assert [record["evidence_mode"] for record in records] == ["two_assigned"]
    assert records[0]["bottleneck"] == 0.61


def test_cycle_closes_assigned_plus_bidirectional_top1_strong_spoke():
    features = [
        _feature("v1_a", "v1", "item", [1, 0], category="item"),
        _feature("v2_a", "v2", "item", [1, 0], category="item"),
        _feature("v3_x", "v3", "item", [1, 0], category="item"),
    ]
    candidates = [
        _candidate("v1_a", "v3_x", 0.72, assigned=True),
        _candidate("v2_a", "v3_x", 0.91, assigned=False),
    ]

    records, clusters = _cycle_run(features, candidates, [("v1_a", "v2_a")])

    assert clusters == [("v1_a", "v2_a", "v3_x")]
    assert [record["evidence_mode"] for record in records] == [
        "assigned_plus_mutual_top1"
    ]
    strong = max(records[0]["support_pairs"], key=lambda item: item["score"])
    assert strong["assigned"] is False
    assert strong["bidirectional_top1"] is True


def test_cycle_rejects_strong_spoke_that_is_not_bidirectional_top1():
    features = [
        _feature("v1_a", "v1", "item", [1, 0], category="item"),
        _feature("v2_a", "v2", "item", [1, 0], category="item"),
        _feature("v3_x", "v3", "item", [1, 0], category="item"),
        _feature("v3_rival", "v3", "item", [1, 0], category="item"),
    ]
    candidates = [
        _candidate("v1_a", "v3_x", 0.72, assigned=True),
        _candidate("v2_a", "v3_x", 0.91, assigned=False),
        # v2_a prefers this rival, so v2_a <-> v3_x is not reciprocal Top-1.
        _candidate("v2_a", "v3_rival", 0.92, assigned=False),
    ]

    records, clusters = _cycle_run(features, candidates, [("v1_a", "v2_a")])

    assert records == []
    assert clusters == [("v1_a", "v2_a"), ("v3_rival",), ("v3_x",)]


def test_cycle_requires_mutual_component_best_and_is_order_deterministic():
    features = [
        _feature("v1_a", "v1", "item", [1, 0], category="item"),
        _feature("v2_a", "v2", "item", [1, 0], category="item"),
        _feature("v1_b", "v1", "item", [1, 0], category="item"),
        _feature("v2_b", "v2", "item", [1, 0], category="item"),
        _feature("v3_x", "v3", "item", [1, 0], category="item"),
        _feature("v3_y", "v3", "item", [1, 0], category="item"),
    ]
    candidates = [
        # x prefers seed A by bottleneck 0.76 vs seed B at 0.75.
        _candidate("v1_a", "v3_x", 0.76, assigned=True),
        _candidate("v2_a", "v3_x", 0.95, assigned=False),
        _candidate("v1_b", "v3_x", 0.94, assigned=False),
        _candidate("v2_b", "v3_x", 0.75, assigned=True),
        # Seed A prefers y by bottleneck 0.85, leaving x unmerged.
        _candidate("v1_a", "v3_y", 0.96, assigned=False),
        _candidate("v2_a", "v3_y", 0.85, assigned=True),
    ]
    seeds = [("v1_a", "v2_a"), ("v1_b", "v2_b")]

    first_records, first_clusters = _cycle_run(features, candidates, seeds)
    second_records, second_clusters = _cycle_run(
        list(reversed(features)), list(reversed(candidates)), list(reversed(seeds))
    )

    assert first_records == second_records
    assert first_clusters == second_clusters == [
        ("v1_a", "v2_a", "v3_y"),
        ("v1_b", "v2_b"),
        ("v3_x",),
    ]


def test_cycle_rejects_canonical_conflict_and_same_video_candidate():
    features = [
        _feature(
            "v1_a", "v1", "toy cabinet", [1, 0],
            category="cabinet", canonical="toy_cabinet",
        ),
        _feature(
            "v2_a", "v2", "toy cabinet", [1, 0],
            category="cabinet", canonical="toy_cabinet",
        ),
        _feature(
            "v3_other", "v3", "wardrobe", [1, 0],
            category="cabinet", canonical="wardrobe",
        ),
        _feature(
            "v1_fragment", "v1", "toy cabinet", [1, 0],
            category="cabinet", canonical="toy_cabinet",
        ),
    ]
    candidates = [
        _candidate("v1_a", "v3_other", 0.95, assigned=True),
        _candidate("v2_a", "v3_other", 0.95, assigned=True),
        _candidate("v1_a", "v1_fragment", 0.95, assigned=True),
        _candidate("v2_a", "v1_fragment", 0.95, assigned=True),
    ]

    records, clusters = _cycle_run(features, candidates, [("v1_a", "v2_a")])

    assert records == []
    assert clusters == [
        ("v1_a", "v2_a"),
        ("v1_fragment",),
        ("v3_other",),
    ]


def test_category_layer_keeps_cabinet_hard_negative_in_scope():
    config = _config()
    toy = _feature(
        "v1_toy", "v1", "toy storage organizer", [1.0, 0.0], category="cabinet", canonical="toy_cabinet"
    )
    wardrobe = _feature(
        "v2_wardrobe", "v2", "wardrobe", [0.9, 0.1], category="cabinet", canonical="wardrobe"
    )
    score = score_pair(toy, wardrobe, config, IdentityConstraints())
    assert "CATEGORY_CONFLICT" not in score.gate_reasons


def test_attribute_score_ignores_pipeline_metadata_keys():
    config = _config()
    a = _feature("v1_a", "v1", "bookshelf", [1, 0], category="bookshelf")
    b = _feature("v2_b", "v2", "bookshelf", [1, 0], category="bookshelf")
    for feature in (a, b):
        feature.tracklet.attributes.update(
            {
                "hero_ref": f"{feature.tracklet_id}.jpg",
                "hero_score": "0.50000000",
                "hero_scoring_version": "area-sharpness-completeness-v1",
            }
        )
    score = score_pair(a, b, config, IdentityConstraints())
    # 仅有元数据键时属性证据缺失:白名单语义下不产生属性分(None,权重让渡),
    # 更不得因常量版本串恒等于 1.0。
    assert score.attribute is None


def test_hard_gates_same_video_category_and_user_negative():
    config = _config()
    a = _feature("v1_a", "v1", "bookshelf", [1, 0], category="bookshelf")
    same_video = _feature("v1_b", "v1", "bookshelf", [1, 0], category="bookshelf")
    other_category = _feature("v2_c", "v2", "desk", [1, 0], category="desk")
    negative = _feature("v2_d", "v2", "bookshelf", [1, 0], category="bookshelf")
    assert "SAME_VIDEO_MUTEX" in score_pair(a, same_video, config, IdentityConstraints()).gate_reasons
    assert "CATEGORY_CONFLICT" in score_pair(a, other_category, config, IdentityConstraints()).gate_reasons
    constraints = IdentityConstraints(different=frozenset({("v1_a", "v2_d")}))
    assert "USER_CANNOT_LINK" in score_pair(a, negative, config, constraints).gate_reasons


def _write_ingest(root: Path) -> None:
    rows = {
        "v1": [("v1_a", "bookshelf", [1.0, 0.0]), ("v1_b", "bookshelf", [0.0, 1.0])],
        "v2": [("v2_a", "bookshelf", [1.0, 0.0]), ("v2_b", "bookshelf", [0.0, 1.0])],
        "v3": [("v3_a", "bookshelf", [1.0, 0.0]), ("v3_b", "bookshelf", [0.0, 1.0])],
    }
    for video_id, items in rows.items():
        video_dir = root / video_id
        evidence = video_dir / "evidence"
        evidence.mkdir(parents=True)
        observations = []
        tracklets = []
        for index, (tracklet_id, label, vector) in enumerate(items):
            observation_id = f"{tracklet_id}_o1"
            embedding = evidence / f"{tracklet_id}_emb.json"
            embedding.write_text(json.dumps({"model": "fake", "vector": vector}))
            observations.append(
                Observation(
                    observation_id=observation_id,
                    video_id=video_id,
                    timestamp_ms=index * 100,
                    bbox=(10.0, 10.0, 110.0, 110.0),
                    crop_ref=str(evidence / f"{tracklet_id}.jpg"),
                    quality=0.9,
                    model_version="fake",
                )
            )
            tracklets.append(
                Tracklet(
                    tracklet_id=tracklet_id,
                    video_id=video_id,
                    observation_ids=[observation_id],
                    prototype_refs=[str(evidence / f"{tracklet_id}.jpg")],
                    embedding_ref=str(embedding),
                    attributes={"label": label},
                )
            )
        (video_dir / "observations.jsonl").write_text(
            "".join(item.model_dump_json() + "\n" for item in observations)
        )
        (video_dir / "tracklets.jsonl").write_text(
            "".join(item.model_dump_json() + "\n" for item in tracklets)
        )


def test_full_reid_is_deterministic_and_preserves_two_instances(tmp_path):
    ingest = tmp_path / "ingest"
    _write_ingest(ingest)
    vocab = Vocabulary(
        (
            VocabularyEntry(
                canonical_id="bookshelf",
                category_id="bookshelf",
                display_label_zh="书架",
                detection_prompts=("bookshelf",),
            ),
        )
    )
    first = run_reid(ingest_root=ingest, config=_config(), vocab=vocab)
    second = run_reid(ingest_root=ingest, config=_config(), vocab=vocab)

    clusters = sorted(tuple(entity.tracklet_ids) for entity in first.entities)
    assert clusters == [("v1_a", "v2_a", "v3_a"), ("v1_b", "v2_b", "v3_b")]
    assert all(entity.identity_state == IdentityState.MATCHED for entity in first.entities)
    assert first.clarifications == []
    assert [item.model_dump() for item in first.entities] == [
        item.model_dump() for item in second.entities
    ]
    assert first.metrics["g2_evaluated"] is False

"""S3 跨视频匹配的纯本地合成测试。"""

import json
from pathlib import Path

from backend.schemas.core import IdentityState, Observation, Tracklet
from backend.pipeline.vocab import Vocabulary, VocabularyEntry
from backend.tools.reid.assignment import maximise_assignment
from backend.tools.reid.matcher import IdentityConstraints, run_reid, score_pair
from backend.tools.reid.model import (
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


def test_hungarian_finds_global_optimum_and_allows_unmatched():
    assignment = maximise_assignment([[0.90, 0.80], [0.85, 0.10]], unmatched_weight=0.0)
    assert assignment == [1, 0]  # 0.80 + 0.85 > 0.90 + 0.10
    assert maximise_assignment([[0.2]], unmatched_weight=0.5) == [None]


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

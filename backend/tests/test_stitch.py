"""同视频 stitch、低证据过滤与澄清互选封顶的纯本地合成测试。"""

import json
from pathlib import Path

from backend.schemas.core import IdentityState, Observation, Tracklet
from backend.pipeline.vocab import Vocabulary, VocabularyEntry
from backend.tools.reid.matcher import _cap_clarifications, run_reid
from backend.tools.reid.model import (
    ClarifyConfig,
    FilterConfig,
    ReIDConfig,
    StitchConfig,
    ThresholdConfig,
    TrackFeature,
    WeightConfig,
)
from backend.tools.reid.stitch import stitch_features, tag_low_evidence


def _config(*, stitch=None, filter_=None, clarify=None) -> ReIDConfig:
    return ReIDConfig(
        version="test-stitch-v1",
        embedding_dim=2,
        top_k=2,
        weights=WeightConfig(instance=0.85, semantic=0.15, attribute=0, context=0, geometry=0),
        thresholds=ThresholdConfig(match=0.95, new=0.60, margin=0.10, min_quality=0.2),
        stitch=stitch or StitchConfig(enabled=True, min_cosine=0.90),
        filter=filter_ or FilterConfig(),
        clarify=clarify or ClarifyConfig(),
    )


def _feature(tracklet_id, video_id, vector, timestamps, *, canonical="bookshelf", quality=0.9):
    tracklet = Tracklet(
        tracklet_id=tracklet_id,
        video_id=video_id,
        observation_ids=[f"{tracklet_id}_o{i}" for i in range(len(timestamps))],
        prototype_refs=[f"{tracklet_id}.jpg"],
        attributes={"label": canonical},
    )
    return TrackFeature(
        tracklet=tracklet,
        vector=tuple(vector),
        raw_label=canonical,
        canonical_id=canonical,
        category_id=canonical,
        quality=quality,
        aspect_ratio=1.0,
        area=100.0,
        timestamps_ms=tuple(timestamps),
    )


def test_stitch_merges_disjoint_fragments_under_min_id():
    a = _feature("v1_t002", "v1", [1.0, 0.0], (0, 100))
    b = _feature("v1_t007", "v1", [1.0, 0.0], (300, 400))
    result = stitch_features([a, b], _config())
    assert len(result.features) == 1
    merged = result.features[0]
    assert merged.tracklet_id == "v1_t002"
    assert result.members_by_rep == {"v1_t002": ["v1_t002", "v1_t007"]}
    assert merged.timestamps_ms == (0, 100, 300, 400)
    assert set(merged.tracklet.observation_ids) == {
        "v1_t002_o0", "v1_t002_o1", "v1_t007_o0", "v1_t007_o1",
    }
    assert merged.tracklet.attributes["stitched_members"] == "v1_t002,v1_t007"
    assert abs(sum(v * v for v in merged.vector) - 1.0) < 1e-9
    assert result.report["merge_count"] == 1


def test_stitch_cooccurrence_is_hard_veto():
    a = _feature("v1_t001", "v1", [1.0, 0.0], (0, 100))
    b = _feature("v1_t002", "v1", [1.0, 0.0], (100, 300))  # 共享 t=100:物理上两个物体
    result = stitch_features([a, b], _config())
    assert len(result.features) == 2
    assert result.report["vetoes"]["co_occurrence"] == 1


def test_stitch_requires_label_and_cosine():
    a = _feature("v1_t001", "v1", [1.0, 0.0], (0,))
    other_label = _feature("v1_t002", "v1", [1.0, 0.0], (200,), canonical="desk")
    far_vector = _feature("v1_t003", "v1", [0.0, 1.0], (400,))
    result = stitch_features([a, other_label, far_vector], _config())
    assert len(result.features) == 3
    assert result.report["merge_count"] == 0


def test_stitch_cluster_level_veto_is_transitive():
    a = _feature("v1_t001", "v1", [1.0, 0.0], (0,))
    b = _feature("v1_t002", "v1", [1.0, 0.0], (200,))
    c = _feature("v1_t003", "v1", [1.0, 0.0], (200,))  # 与 b 共现
    result = stitch_features([a, b, c], _config())
    reps = sorted(f.tracklet_id for f in result.features)
    # a-b 先合并;c 与簇内 b 共现,不得再并入
    assert reps == ["v1_t001", "v1_t003"]
    assert result.members_by_rep == {"v1_t001": ["v1_t001", "v1_t002"]}
    assert result.report["vetoes"]["co_occurrence"] >= 1


def test_stitch_disabled_is_noop():
    a = _feature("v1_t001", "v1", [1.0, 0.0], (0,))
    b = _feature("v1_t002", "v1", [1.0, 0.0], (200,))
    result = stitch_features([a, b], _config(stitch=StitchConfig(enabled=False)))
    assert [f.tracklet_id for f in result.features] == ["v1_t001", "v1_t002"]
    assert result.report["merge_count"] == 0
    assert result.members_by_rep == {}


def test_stitch_respects_cannot_link_and_user_same():
    a = _feature("v1_t001", "v1", [1.0, 0.0], (0,))
    b = _feature("v1_t002", "v1", [1.0, 0.0], (200,))
    blocked = stitch_features([a, b], _config(), forbidden=frozenset({("v1_t001", "v1_t002")}))
    assert len(blocked.features) == 2
    assert blocked.report["vetoes"]["cannot_link"] == 1

    # 用户 same 优先级最高,共现也让位
    c = _feature("v1_t003", "v1", [1.0, 0.0], (0,))
    forced = stitch_features(
        [a, c], _config(stitch=StitchConfig(enabled=False)), forced_same=frozenset({("v1_t001", "v1_t003")})
    )
    assert len(forced.features) == 1
    assert forced.report["merges"][0]["mode"] == "user_same"


def test_tag_low_evidence_marks_but_protects_constrained():
    config = _config(filter_=FilterConfig(min_observations=2))
    strong = _feature("v1_t001", "v1", [1.0, 0.0], (0, 100))
    weak = _feature("v1_t002", "v1", [1.0, 0.0], (300,))
    protected = _feature("v1_t003", "v1", [1.0, 0.0], (500,))
    low, records = tag_low_evidence(
        [strong, weak, protected], config, {}, protected=frozenset({"v1_t003"})
    )
    assert low == frozenset({"v1_t002"})
    assert len(records) == 1
    assert records[0]["reason"] == "LOW_EVIDENCE_CLARIFICATION_SUPPRESSED"


def test_cap_is_mutual_and_suppresses_star_fanout():
    ambiguous = {
        ("v1_a", "v2_b1"): (("SCORE_OR_MARGIN_UNCERTAIN",), 0.80),
        ("v1_a", "v2_b2"): (("SCORE_OR_MARGIN_UNCERTAIN",), 0.75),
        ("v1_a", "v2_b3"): (("GLOBAL_ASSIGNMENT_CONTENTION",), 0.72),
    }
    video_of = {"v1_a": "v1", "v2_b1": "v2", "v2_b2": "v2", "v2_b3": "v2"}
    capped, suppressed = _cap_clarifications(ambiguous, 1, video_of)
    assert set(capped) == {("v1_a", "v2_b1")}
    assert suppressed == 2
    uncapped, zero = _cap_clarifications(ambiguous, 0, video_of)
    assert uncapped == ambiguous and zero == 0


def _write_ingest(root: Path, rows: dict) -> None:
    for video_id, items in rows.items():
        video_dir = root / video_id
        evidence = video_dir / "evidence"
        evidence.mkdir(parents=True)
        observations, tracklets = [], []
        for tracklet_id, label, vector, timestamps in items:
            embedding = evidence / f"{tracklet_id}_emb.json"
            embedding.write_text(json.dumps({"model": "fake", "vector": vector}))
            observation_ids = []
            for index, timestamp in enumerate(timestamps):
                observation_id = f"{tracklet_id}_o{index}"
                observation_ids.append(observation_id)
                observations.append(
                    Observation(
                        observation_id=observation_id,
                        video_id=video_id,
                        timestamp_ms=timestamp,
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
                    observation_ids=observation_ids,
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


def _vocab() -> Vocabulary:
    return Vocabulary(
        (
            VocabularyEntry(
                canonical_id="bookshelf",
                category_id="bookshelf",
                display_label_zh="书架",
                detection_prompts=("bookshelf",),
            ),
        )
    )


def test_run_reid_stitches_fragments_and_expands_original_ids(tmp_path):
    ingest = tmp_path / "ingest"
    _write_ingest(
        ingest,
        {
            "v1": [
                ("v1_t001", "bookshelf", [1.0, 0.0], [0, 100]),
                ("v1_t002", "bookshelf", [1.0, 0.0], [300, 400]),  # 同物碎片
            ],
            "v2": [("v2_t001", "bookshelf", [1.0, 0.0], [0, 100])],
        },
    )
    config = _config()
    first = run_reid(ingest_root=ingest, config=config, vocab=_vocab())
    second = run_reid(ingest_root=ingest, config=config, vocab=_vocab())

    assert first.metrics["tracklet_count"] == 3
    assert first.metrics["tracklet_count_after_stitch"] == 2
    assert first.metrics["stitch_merge_count"] == 1
    assert len(first.entities) == 1
    entity = first.entities[0]
    assert entity.identity_state == IdentityState.MATCHED
    assert entity.tracklet_ids == ["v1_t001", "v1_t002", "v2_t001"]
    assert first.clarifications == []
    assert first.stitch_report["groups"] == {"v1_t001": ["v1_t001", "v1_t002"]}
    assert [e.model_dump() for e in first.entities] == [e.model_dump() for e in second.entities]


def test_run_reid_low_evidence_keeps_auto_link_but_cannot_ask(tmp_path):
    ingest = tmp_path / "ingest"
    _write_ingest(
        ingest,
        {
            "v1": [
                ("v1_t001", "bookshelf", [1.0, 0.0], [0, 100]),
                ("v1_t002", "bookshelf", [0.8, 0.6], [300]),  # 低证据 + 歧义区分数
            ],
            "v2": [("v2_t001", "bookshelf", [1.0, 0.0], [0])],  # 低证据 + 高分
        },
    )
    config = _config(
        stitch=StitchConfig(enabled=False), filter_=FilterConfig(min_observations=2)
    )
    run = run_reid(ingest_root=ingest, config=config, vocab=_vocab())

    # 高分链接不因低证据而丢:v1_t001 <-> v2_t001 照常自动合并
    assert run.metrics["automatic_link_count"] == 1
    matched = [e for e in run.entities if e.identity_state == IdentityState.MATCHED]
    assert len(matched) == 1 and matched[0].tracklet_ids == ["v1_t001", "v2_t001"]
    # 歧义区的对因低证据端点被摘出人工队列,但实体状态仍然诚实
    assert run.metrics["low_evidence_tracklet_count"] == 2
    assert run.metrics["clarifications_suppressed_low_evidence"] >= 1
    assert run.clarifications == []
    suspected = [e for e in run.entities if e.identity_state == IdentityState.SUSPECTED_DUPLICATE]
    assert [e.tracklet_ids for e in suspected] == [["v1_t002"]]

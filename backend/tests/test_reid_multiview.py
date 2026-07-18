"""多视角候选内重排的 scorer、产物契约与兼容性测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backend.pipeline.vocab import Vocabulary, VocabularyEntry
from backend.schemas.core import Tracklet
from backend.tools.reid.matcher import (
    IdentityConstraints,
    PairScore,
    _pairwise_assignments,
    _rerank_recalled_scores,
)
from backend.tools.reid.model import (
    MultiViewConfig,
    ReIDConfig,
    ThresholdConfig,
    TrackFeature,
    WeightConfig,
    _load_multiview_vectors,
    load_features,
)
from backend.tools.reid.multiview import (
    ARTIFACT_FORMAT_VERSION,
    quantile_calibrate,
    set_similarity,
)
from backend.tools.reid.stitch import _merge_cluster
from backend.tools.sf1.projection import sha256_file
from scripts.reid_multiview_embed import PrototypeRecord, write_artifact


def _config(*, multiview: MultiViewConfig | None = None) -> ReIDConfig:
    return ReIDConfig(
        version="test-multiview-v1",
        embedding_dim=2,
        top_k=1,
        weights=WeightConfig(
            instance=1.0,
            semantic=0.0,
            attribute=0.0,
            context=0.0,
            geometry=0.0,
        ),
        thresholds=ThresholdConfig(
            match=0.9, new=0.6, margin=0.0, min_quality=0.0
        ),
        multiview=multiview or MultiViewConfig(),
    )


def _feature(
    tracklet_id: str,
    video_id: str,
    vector: tuple[float, float],
    *,
    views: tuple[tuple[float, float], ...] = (),
) -> TrackFeature:
    return TrackFeature(
        tracklet=Tracklet(
            tracklet_id=tracklet_id,
            video_id=video_id,
            observation_ids=[],
            prototype_refs=[f"{tracklet_id}-{index}.jpg" for index in range(len(views))],
            attributes={"label": "item"},
        ),
        vector=vector,
        raw_label="item",
        canonical_id="item",
        category_id="item",
        quality=1.0,
        aspect_ratio=1.0,
        area=1.0,
        view_vectors=views,
    )


def test_set_similarity_methods_are_symmetric_bounded_and_view_limited():
    left = ((1.0, 0.0), (0.0, 1.0))
    right = ((0.6, 0.8),)

    assert set_similarity(left, right, method="max_pair", max_views=2) == pytest.approx(
        0.8
    )
    assert set_similarity(
        left, right, method="mean_chamfer", max_views=2
    ) == pytest.approx(0.75)
    symmetric = set_similarity(
        left, right, method="symmetric_top2", max_views=2
    )
    assert symmetric == pytest.approx(0.775)
    assert symmetric == pytest.approx(
        set_similarity(right, left, method="symmetric_top2", max_views=2)
    )
    assert set_similarity(left, right, method="max_pair", max_views=1) == pytest.approx(
        0.6
    )
    with pytest.raises(ValueError, match="same positive dimension"):
        set_similarity(((1.0,),), right, method="max_pair", max_views=2)
    with pytest.raises(ValueError, match="max_views"):
        set_similarity(left, right, method="max_pair", max_views=0)


def test_quantile_calibration_preserves_baseline_multiset_and_is_deterministic():
    pair_a = ("a", "x")
    pair_b = ("b", "x")
    pair_c = ("c", "x")
    # a/b 的 multiview 同分但 baseline 与字典序相反；同分必须保留已有证据顺序。
    baseline = {pair_a: 0.7, pair_b: 0.9, pair_c: 0.8}
    multiview = {pair_a: 0.5, pair_b: 0.5, pair_c: 0.99}

    calibrated = quantile_calibrate(baseline, multiview)

    assert calibrated == {pair_c: 0.9, pair_b: 0.8, pair_a: 0.7}
    assert sorted(calibrated.values()) == sorted(baseline.values())
    with pytest.raises(ValueError, match="pair sets differ"):
        quantile_calibrate(baseline, {pair_a: 0.5})


@pytest.mark.parametrize(
    ("multiview", "error"),
    [
        (MultiViewConfig(enabled=True), "requires artifact and sha256"),
        (
            MultiViewConfig(
                enabled=True, artifact="views.npz", sha256="hash", method="unknown"
            ),
            "unsupported multiview.method",
        ),
        (
            MultiViewConfig(
                enabled=True, artifact="views.npz", sha256="hash", blend=1.1
            ),
            "blend",
        ),
        (
            MultiViewConfig(
                enabled=True,
                artifact="views.npz",
                sha256="hash",
                max_views_per_rep=1,
            ),
            "max_views_per_rep",
        ),
        (
            MultiViewConfig(
                enabled=True,
                artifact="views.npz",
                sha256="hash",
                space="projected",
            ),
            "requires projection.enabled",
        ),
    ],
)
def test_multiview_config_fails_closed(multiview, error):
    with pytest.raises(ValueError, match=error):
        _config(multiview=multiview).validate()


def test_producer_artifact_round_trips_through_loader(tmp_path):
    records = [
        PrototypeRecord("t1", 0, "t1-0.jpg", tmp_path / "t1-0.jpg"),
        PrototypeRecord("t1", 1, "t1-1.jpg", tmp_path / "t1-1.jpg"),
        PrototypeRecord("t2", 0, "t2-0.jpg", tmp_path / "t2-0.jpg"),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    artifact = tmp_path / "views.npz"
    write_artifact(
        artifact,
        tmp_path / "views.manifest.json",
        records,
        vectors,
        model_version="fake@test",
        source_files=[],
        missing=[],
        inputs_sha256="input-hash",
    )

    loaded = _load_multiview_vectors(
        str(artifact),
        tmp_path,
        2,
        expected_sha256=sha256_file(artifact),
    )

    np.testing.assert_allclose(loaded["t1"], ((1.0, 0.0), (0.0, 1.0)))
    np.testing.assert_allclose(loaded["t2"], ((0.6, 0.8),))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _load_multiview_vectors(
            str(artifact), tmp_path, 2, expected_sha256="0" * 64
        )


def _write_raw_artifact(
    path: Path,
    *,
    version=ARTIFACT_FORMAT_VERSION,
    tracklet_ids=None,
    view_index=None,
    vectors=None,
    index_key="view_index",
) -> None:
    default_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    arrays = {
        "format_version": np.asarray(version),
        "tracklet_ids": np.asarray(
            ["t1", "t1"] if tracklet_ids is None else tracklet_ids
        ),
        index_key: np.asarray(
            [0, 1] if view_index is None else view_index, dtype=np.int64
        ),
        "vectors": default_vectors if vectors is None else np.asarray(vectors),
    }
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("version", "unsupported multiview format"),
        ("legacy_index_key", "view_index"),
        ("row_count", "inconsistent row counts"),
        ("dimension", "expects"),
        ("non_finite", "non-finite"),
        ("zero", "zero-norm"),
        ("non_unit", "L2-normalized"),
        ("float64", "float32"),
        ("non_contiguous", "not contiguous"),
    ],
)
def test_artifact_loader_fails_closed_on_schema_or_vector_corruption(
    tmp_path, case, error
):
    artifact = tmp_path / f"{case}.npz"
    kwargs = {}
    if case == "version":
        kwargs["version"] = "wrong-v0"
    elif case == "legacy_index_key":
        kwargs["index_key"] = "view_indices"
    elif case == "row_count":
        kwargs["tracklet_ids"] = ["t1"]
    elif case == "dimension":
        kwargs["vectors"] = np.asarray([[1.0], [1.0]], dtype=np.float32)
    elif case == "non_finite":
        kwargs["vectors"] = np.asarray([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32)
    elif case == "zero":
        kwargs["vectors"] = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    elif case == "non_unit":
        kwargs["vectors"] = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    elif case == "float64":
        kwargs["vectors"] = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    elif case == "non_contiguous":
        kwargs["view_index"] = [0, 2]
    _write_raw_artifact(artifact, **kwargs)

    with pytest.raises((ValueError, KeyError), match=error):
        _load_multiview_vectors(
            str(artifact),
            tmp_path,
            2,
            expected_sha256=sha256_file(artifact),
        )


def _write_minimal_ingest(root: Path, *, prototype_count: int = 2) -> None:
    video = root / "v1"
    evidence = video / "evidence"
    evidence.mkdir(parents=True)
    embedding = evidence / "t1_emb.json"
    embedding.write_text(json.dumps({"model": "fake", "vector": [1.0, 0.0]}))
    tracklet = Tracklet(
        tracklet_id="t1",
        video_id="v1",
        observation_ids=[],
        prototype_refs=[f"t1-{index}.jpg" for index in range(prototype_count)],
        embedding_ref=str(embedding),
        attributes={"label": "item"},
    )
    (video / "tracklets.jsonl").write_text(tracklet.model_dump_json() + "\n")


def _vocab() -> Vocabulary:
    return Vocabulary(
        (
            VocabularyEntry(
                canonical_id="item",
                category_id="item",
                display_label_zh="物品",
                detection_prompts=("item",),
            ),
        )
    )


def test_load_features_fails_when_artifact_omits_a_prototype_view(tmp_path):
    ingest = tmp_path / "ingest"
    _write_minimal_ingest(ingest, prototype_count=2)
    artifact = tmp_path / "views.npz"
    _write_raw_artifact(
        artifact,
        tracklet_ids=["t1"],
        view_index=[0],
        vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="view count mismatch"):
        load_features(
            ingest,
            vocab=_vocab(),
            embedding_dim=2,
            multiview=MultiViewConfig(
                enabled=True,
                artifact=str(artifact),
                sha256=sha256_file(artifact),
            ),
        )


def test_disabled_multiview_does_not_resolve_artifact_or_change_features(tmp_path):
    ingest = tmp_path / "ingest"
    _write_minimal_ingest(ingest, prototype_count=2)

    baseline = load_features(ingest, vocab=_vocab(), embedding_dim=2)
    disabled = load_features(
        ingest,
        vocab=_vocab(),
        embedding_dim=2,
        multiview=MultiViewConfig(
            enabled=False,
            artifact=str(tmp_path / "does-not-exist.npz"),
            sha256="not-a-real-hash",
        ),
    )

    assert disabled == baseline
    assert disabled[0].view_vectors == ()


def test_multiview_reranking_never_expands_baseline_symmetric_topk(
    monkeypatch,
):
    features = [
        _feature("v1_a", "v1", (1.0, 0.0), views=((1.0, 0.0),)),
        _feature("v1_b", "v1", (0.0, 1.0), views=((0.0, 1.0),)),
        _feature("v2_a", "v2", (1.0, 0.0), views=((1.0, 0.0),)),
        # 这条与 v1_b 的逐视角完全相同，但 baseline 不召回该边。
        _feature("v2_b", "v2", (0.0, 1.0), views=((0.0, 1.0),)),
    ]
    totals = {
        ("v1_a", "v2_a"): 0.95,
        ("v1_a", "v2_b"): 0.80,
        ("v1_b", "v2_a"): 0.70,
        ("v1_b", "v2_b"): 0.10,
    }

    def fake_score_pair(a, b, _config, _constraints):
        total = totals[(a.tracklet_id, b.tracklet_id)]
        return PairScore(
            a=a.tracklet_id,
            b=b.tracklet_id,
            instance=total,
            semantic=0.0,
            attribute=None,
            context=0.0,
            geometry=0.0,
            total=total,
        )

    monkeypatch.setattr("backend.tools.reid.matcher.score_pair", fake_score_pair)
    baseline = _pairwise_assignments(features, _config(), IdentityConstraints())
    explicit_disabled = _pairwise_assignments(
        features,
        replace(
            _config(),
            multiview=MultiViewConfig(
                enabled=False, artifact="ignored", sha256="ignored"
            ),
        ),
        IdentityConstraints(),
    )
    enabled = _pairwise_assignments(
        features,
        _config(
            multiview=MultiViewConfig(
                enabled=True,
                artifact="unused-by-pure-helper",
                sha256="unused-by-pure-helper",
                method="max_pair",
                calibration="none",
            )
        ),
        IdentityConstraints(),
    )

    assert explicit_disabled == baseline
    baseline_pairs = {
        (row["tracklet_a"], row["tracklet_b"]) for row in baseline[2]
    }
    enabled_pairs = {(row["tracklet_a"], row["tracklet_b"]) for row in enabled[2]}
    assert enabled_pairs == baseline_pairs
    assert ("v1_b", "v2_b") not in enabled_pairs


def test_reranker_updates_only_recalled_score_and_keeps_audit_components():
    left = _feature("v1_a", "v1", (1.0, 0.0), views=((1.0, 0.0),))
    right = _feature("v2_a", "v2", (0.0, 1.0), views=((1.0, 0.0),))
    key = ("v1_a", "v2_a")
    baseline = PairScore(
        a=key[0],
        b=key[1],
        instance=0.2,
        semantic=0.0,
        attribute=None,
        context=0.0,
        geometry=0.0,
        total=0.2,
    )
    config = _config(
        multiview=MultiViewConfig(
            enabled=True,
            artifact="unused",
            sha256="unused",
            method="max_pair",
            blend=0.5,
            calibration="none",
        )
    )

    reranked = _rerank_recalled_scores(
        {key: baseline}, {key[0]: left, key[1]: right}, config
    )

    assert set(reranked) == {key}
    assert baseline.instance_base is None
    assert reranked[key].instance == pytest.approx(0.6)
    assert reranked[key].total == pytest.approx(0.6)
    assert reranked[key].instance_base == pytest.approx(0.2)
    assert reranked[key].instance_multiview_raw == pytest.approx(1.0)
    assert reranked[key].instance_multiview_calibrated == pytest.approx(1.0)


def test_stitch_interleaves_views_by_rank_before_next_fragment_view():
    a = _feature(
        "v1_a",
        "v1",
        (1.0, 0.0),
        views=((1.0, 0.0), (0.9, 0.1)),
    )
    b = _feature(
        "v1_b",
        "v1",
        (1.0, 0.0),
        views=((0.8, 0.2), (0.7, 0.3), (0.6, 0.4)),
    )

    merged = _merge_cluster([b, a])

    assert merged.view_vectors == (
        (1.0, 0.0),
        (0.8, 0.2),
        (0.9, 0.1),
        (0.7, 0.3),
        (0.6, 0.4),
    )

"""S5 属性接线:missing/unknown 语义、权重让渡、enrichment 加载。"""

from __future__ import annotations

import json

import pytest

from backend.schemas.core import Tracklet
from backend.tools.reid.matcher import _attribute_score, score_pair, IdentityConstraints
from backend.tools.reid.model import (
    COMPARABLE_ATTRIBUTE_KEYS,
    ReIDConfig,
    TrackFeature,
    load_attribute_enrichment,
)


def _feature(tid: str, video: str, attrs: dict[str, str], vector=None) -> TrackFeature:
    tracklet = Tracklet(
        tracklet_id=tid,
        video_id=video,
        observation_ids=[],
        attributes={"label": "storage box", "hero_scoring_version": "v1", **attrs},
    )
    dim_vector = tuple(vector or ([1.0] + [0.0] * 767))
    return TrackFeature(
        tracklet=tracklet,
        vector=dim_vector,
        raw_label="storage box",
        canonical_id="storage_box",
        category_id="storage_box",
        quality=0.9,
        aspect_ratio=1.0,
        area=100.0,
        timestamps_ms=(0, 1000, 2000),
    )


def _config(attr_weight: float = 0.15) -> ReIDConfig:
    config = ReIDConfig.from_yaml("configs/reid_dev_a_v3.yaml")
    from dataclasses import replace

    return replace(
        config,
        weights=replace(
            config.weights,
            instance=config.weights.instance + config.weights.attribute - attr_weight,
            attribute=attr_weight,
        ),
    )


def test_pipeline_metadata_never_scores():
    # label/hero_* 等流水线元数据不在白名单,即使两侧相同也不产生属性分
    a = _feature("v1_t001", "v1", {})
    b = _feature("v2_t001", "v2", {})
    assert _attribute_score(a, b) is None


def test_unknown_keys_excluded_both_sides():
    a = _feature("v1_t001", "v1", {"color_primary": "pink", "material": "unknown"})
    b = _feature("v2_t001", "v2", {"color_primary": "pink", "material": "plastic"})
    # material 一侧 unknown → 只比 color_primary → 1/1
    assert _attribute_score(a, b) == 1.0


def test_mismatch_counts_against():
    a = _feature("v1_t001", "v1", {"color_primary": "pink", "material": "plastic"})
    b = _feature("v2_t001", "v2", {"color_primary": "blue", "material": "plastic"})
    assert _attribute_score(a, b) == pytest.approx(0.5)


def test_weight_redistribution_when_no_comparable_keys():
    config = _config(0.15)
    a = _feature("v1_t001", "v1", {})
    b = _feature("v2_t001", "v2", {})
    score = score_pair(a, b, config, IdentityConstraints())
    assert score.attribute is None
    # 无属性时的 total 应等于把 attr 权重整体拿掉后的归一化结果
    w = config.weights
    expected = (
        w.instance * score.instance + w.semantic * score.semantic
        + w.context * score.context + w.geometry * score.geometry
    ) / (w.total - w.attribute)
    assert score.total == pytest.approx(expected)


def test_attribute_zero_weight_total_invariant():
    """attr 权重 0 时,有无 enrichment 不改变 total(对照组零行为差)。"""
    config = _config(0.0)
    bare_a, bare_b = _feature("v1_t001", "v1", {}), _feature("v2_t001", "v2", {})
    rich_a = _feature("v1_t001", "v1", {"color_primary": "pink", "material": "plastic"})
    rich_b = _feature("v2_t001", "v2", {"color_primary": "blue", "material": "fabric"})
    constraints = IdentityConstraints()
    assert score_pair(bare_a, bare_b, config, constraints).total == pytest.approx(
        score_pair(rich_a, rich_b, config, constraints).total
    )


def test_attribute_variance_separates_lookalikes():
    """同类同嵌入的两对:属性一致对得分高于属性冲突对。"""
    config = _config(0.15)
    anchor = _feature("v1_t001", "v1", {"color_primary": "pink", "material": "plastic"})
    same = _feature("v2_t001", "v2", {"color_primary": "pink", "material": "plastic"})
    other = _feature("v2_t002", "v2", {"color_primary": "blue", "material": "plastic"})
    constraints = IdentityConstraints()
    score_same = score_pair(anchor, same, config, constraints).total
    score_other = score_pair(anchor, other, config, constraints).total
    assert score_same > score_other


def test_load_attribute_enrichment_filters_keys(tmp_path):
    path = tmp_path / "attrs.jsonl"
    rows = [
        {"tracklet_id": "v1_t001", "status": "OK",
         "attributes": {"color_primary": "pink", "label_en": "toy box",
                        "confidence": "high", "material": "plastic"}},
        {"tracklet_id": "v1_t002", "status": "OK", "attributes": {"label_en": "lamp"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    enrichment = load_attribute_enrichment(path)
    assert enrichment == {"v1_t001": {"color_primary": "pink", "material": "plastic"}}
    assert set(enrichment["v1_t001"]) <= set(COMPARABLE_ATTRIBUTE_KEYS)

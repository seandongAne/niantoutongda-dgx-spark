from __future__ import annotations

from backend.tools.reid.neighborhood import (
    NeighborRelation,
    NeighborhoodSignature,
)
from scripts.reid_neighborhood_proxy import _contextualize


def _signature(tracklet_id, video, distance):
    return NeighborhoodSignature(
        tracklet_id=tracklet_id,
        video_id=video,
        relations=(
            NeighborRelation("lamp", 1.0, distance, 0.0, 3),
            NeighborRelation("book", 1.0, distance + 0.1, 1.0, 3),
        ),
    )


def test_contextualize_only_changes_covered_pair_order_and_is_deterministic():
    pair_good = ("v1_t1", "v2_t1")
    pair_bad = ("v1_t1", "v2_t2")
    pair_uncovered = ("v1_t1", "v2_t3")
    baseline = {pair_good: 0.71, pair_bad: 0.79, pair_uncovered: 0.75}
    signatures = {
        "v1_t1": _signature("v1_t1", "v1", 0.1),
        "v2_t1": _signature("v2_t1", "v2", 0.11),
        "v2_t2": _signature("v2_t2", "v2", 0.5),
        "v2_t3": NeighborhoodSignature("v2_t3", "v2", ()),
    }

    first, evidence, counts = _contextualize(
        baseline, signatures, min_shared_anchors=2, blend=1.0
    )
    second, _, _ = _contextualize(
        baseline, signatures, min_shared_anchors=2, blend=1.0
    )

    assert first == second
    assert first[pair_good] > first[pair_bad]
    assert evidence[pair_uncovered] is None
    assert counts == {
        "pairs": 3,
        "score_band_pairs": 3,
        "covered": 2,
        "uncovered": 1,
        "applied": 2,
    }
    assert first[pair_uncovered] == baseline[pair_uncovered]
    assert sorted(first.values()) == sorted(baseline.values())

import json

from backend.pipeline.vocab import Vocabulary, VocabularyEntry
from backend.tools.ingest_compare import compare_ingests


def _vocab():
    return Vocabulary(
        (
            VocabularyEntry("desk", "desk", "书桌", ("desk",)),
            VocabularyEntry("luggage", "suitcase", "行李箱", ("luggage",)),
            VocabularyEntry("mini_fridge", "refrigerator", "玩具冰箱", ("mini fridge",)),
        )
    )


def _run(root, *, hero):
    video = root / "v1"
    (video / "keyframes").mkdir(parents=True)
    (video / "keyframes" / "kf_000000.jpg").write_bytes(b"jpg")
    (video / "observations.jsonl").write_text(
        json.dumps({"observation_id": "o1"}) + "\n" + json.dumps({"observation_id": "o2"}) + "\n"
    )
    attributes = {"label": "luggage mini fridge"}
    if hero:
        attributes["hero_ref"] = "evidence/hero.jpg"
    (video / "tracklets.jsonl").write_text(
        json.dumps({"tracklet_id": "t1", "attributes": {"label": "desk"}}) + "\n"
        + json.dumps({"tracklet_id": "t2", "attributes": attributes})
        + "\n"
    )


def test_comparison_is_explicitly_not_hardval(tmp_path):
    baseline, candidate = tmp_path / "v5", tmp_path / "v6"
    _run(baseline, hero=False)
    _run(candidate, hero=True)
    baseline_log = tmp_path / "v5.log"
    candidate_log = tmp_path / "v6.log"
    baseline_log.write_text("[v1] 1 kf, 2 obs, 2 tracklets, 10s\n")
    candidate_log.write_text(
        "[v1] 1 kf, 2 obs, 2 tracklets, 8s (detect=5.0s, batch=2, tiled_kf=1)\n"
    )

    result = compare_ingests(
        baseline_root=baseline,
        candidate_root=candidate,
        vocab=_vocab(),
        baseline_log=baseline_log,
        candidate_log=candidate_log,
    )

    assert result["hardval_metrics_evaluated"] is False
    assert result["deltas"]["wall_s"] == -2
    assert result["deltas"]["wall_ratio_candidate_over_baseline"] == 0.8
    assert result["candidate"]["videos"]["v1"]["hero_ref_coverage"] == 0.5
    assert result["candidate"]["videos"]["v1"]["unknown_raw_label_counts"] == {
        "luggage mini fridge": 1
    }

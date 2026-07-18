from backend.pipeline.vocab import VocabTranscription, Vocabulary, VocabularyEntry
from scripts.reid_semantic_signature_proxy import (
    _label_similarity,
    build_signatures,
    identity_disjoint_split,
    signature_scores,
    topk_retrieval_metrics,
)


def _vocab():
    return Vocabulary(
        (
            VocabularyEntry("red_mug", "mug", "红杯", ("red mug",)),
            VocabularyEntry("blue_mug", "mug", "蓝杯", ("blue mug",)),
            VocabularyEntry("book", "book", "书", ("book",)),
        )
    )


def test_identity_split_has_no_identity_or_tracklet_overlap():
    samples = {
        f"v{video}_t{identity}": (f"pseudo_{identity}", f"v{video}")
        for identity in range(20)
        for video in (1, 2)
    }

    development, holdout, report = identity_disjoint_split(
        samples, modulus=3, holdout_bucket=0
    )

    assert set(development).isdisjoint(holdout)
    assert {value[0] for value in development.values()}.isdisjoint(
        value[0] for value in holdout.values()
    )
    assert report["tracklet_overlap"] == 0
    assert report["development_identities"] + report["holdout_identities"] == 20


def test_signature_reconciles_names_and_instance_attributes():
    signatures, coverage = build_signatures(
        ["v1_ta", "v2_ta", "v2_tb"],
        vocab=_vocab(),
        raw_labels={"v1_ta": "red mug", "v2_ta": "red mug", "v2_tb": "book"},
        attributes={
            "v1_ta": {"label_zh": "红杯", "color_primary": "red", "shape": "cylinder"},
            "v2_ta": {"label_en": "red mug", "color_primary": "red", "shape": "cylinder"},
            "v2_tb": {"label_en": "book", "color_primary": "blue", "shape": "box"},
        },
    )

    scores = signature_scores(
        [("v1_ta", "v2_ta"), ("v1_ta", "v2_tb")], signatures
    )

    assert coverage["mapped"] == 3
    assert scores[("v1_ta", "v2_ta")] == 1.0
    assert scores[("v1_ta", "v2_tb")] == 0.0


def test_label_similarity_is_calibrated_by_transcription_confidence():
    exact = VocabTranscription("red_mug", "mug", 1.0, "mapped")
    compound = VocabTranscription("red_mug", "mug", 0.8, "mapped")

    assert _label_similarity(exact, compound) == 0.8


def test_topk_metric_reranks_only_the_baseline_candidate_universe():
    samples = {
        "v2_tq": ("same", "v2"),
        "v1_tp": ("same", "v1"),
        "v1_tn": ("different", "v1"),
    }
    query, gallery = ["v2_tq"], ["v1_tp", "v1_tn"]
    baseline = {("v2_tq", "v1_tp"): 0.8, ("v2_tq", "v1_tn"): 0.9}
    candidate = {("v2_tq", "v1_tp"): 0.95, ("v2_tq", "v1_tn"): 0.7}

    before = topk_retrieval_metrics(
        query, gallery, samples, baseline, baseline, top_k=2
    )
    after = topk_retrieval_metrics(
        query, gallery, samples, baseline, candidate, top_k=2
    )

    assert before["recall_at_1"] == 0.0
    assert after["recall_at_1"] == 1.0
    assert after["positive_recall_rate"] == before["positive_recall_rate"] == 1.0
    assert after["paired_top1"]["gains"] == 1
    assert after["paired_top1"]["losses"] == 0
    assert after["query_outcomes"] == [
        {
            "query_id": "v2_tq",
            "baseline_rank": 2,
            "candidate_rank": 1,
            "positive_in_baseline_top_k": True,
        }
    ]

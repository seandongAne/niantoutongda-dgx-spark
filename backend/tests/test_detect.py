"""Alias/canonical-aware cross-batch NMS tests (no torch required)."""

from backend.pipeline.detect import RawDetection, canonical_aware_nms


BOX = (0.0, 0.0, 100.0, 100.0)
OVERLAP_09 = (0.0, 0.0, 100.0, 90.0)


def test_high_overlap_different_canonical_concepts_are_both_kept():
    detections = [
        RawDetection("security camera", 0.90, BOX),
        RawDetection("smart speaker", 0.85, OVERLAP_09),
    ]

    kept = canonical_aware_nms(
        detections,
        prompt_to_canonical={
            "security camera": "security_camera",
            "smart speaker": "night_light",
        },
    )

    assert len(kept) == 2
    assert {d.label for d in kept} == {"security_camera", "night_light"}


def test_high_overlap_aliases_of_same_canonical_keep_higher_score():
    detections = [
        RawDetection("smart speaker", 0.72, BOX),
        RawDetection("cylinder lamp", 0.91, OVERLAP_09),
    ]

    kept = canonical_aware_nms(
        detections,
        prompt_to_canonical={
            "smart speaker": "night_light",
            "cylinder lamp": "night_light",
        },
        prompt_to_category={"smart speaker": "lamp", "cylinder lamp": "lamp"},
    )

    assert kept == [
        RawDetection(
            label="night_light",
            score=0.91,
            box=OVERLAP_09,
            canonical_id="night_light",
            category_id="lamp",
            raw_label="cylinder lamp",
        )
    ]


def test_same_semantic_category_does_not_trigger_nms_across_canonicals():
    detections = [
        RawDetection("smart speaker", 0.90, BOX),
        RawDetection("table lamp", 0.89, OVERLAP_09),
    ]

    kept = canonical_aware_nms(
        detections,
        prompt_to_canonical={
            "smart speaker": "night_light",
            "table lamp": "table_lamp",
        },
        prompt_to_category={"smart speaker": "lamp", "table lamp": "lamp"},
    )

    assert len(kept) == 2
    assert {d.canonical_id for d in kept} == {"night_light", "table_lamp"}


def test_unique_truncated_text_label_resolves_but_ambiguous_one_does_not():
    kept = canonical_aware_nms(
        [RawDetection("camera", 0.9, BOX), RawDetection("lamp", 0.8, OVERLAP_09)],
        prompt_to_canonical={
            "security camera": "security_camera",
            "cylinder lamp": "night_light",
            "table lamp": "table_lamp",
        },
        prompt_to_category={
            "security camera": "camera",
            "cylinder lamp": "lamp",
            "table lamp": "lamp",
        },
    )

    assert kept[0].canonical_id == "security_camera"
    assert kept[1].canonical_id is None

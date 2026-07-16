import json

import pytest

from backend.tools.reid.hard_negative_audit import audit_hard_negatives


def _write_json(path, value):
    path.write_text(json.dumps(value))


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path, *, review_status="visual_draft_pending_data_owner"):
    template = tmp_path / "template.json"
    review = tmp_path / "review.json"
    result = tmp_path / "result"
    result.mkdir()
    _write_json(
        template,
        {
            "dataset_version": "test-v1",
            "hard_negative_pairs": [
                {"group_id": "A", "anchor_ids": ["left", "right"], "category_id": "bookshelf"}
            ],
        },
    )
    _write_json(
        review,
        {
            "dataset_version": "test-v1",
            "review_status": review_status,
            "anchors": {
                "left": {
                    "mapping_completeness": "partial",
                    "review_confidence": "high",
                    "tracklet_ids_by_video": {"v1": ["v1_left", "v1_left_fragment"], "v2": ["v2_left"]},
                },
                "right": {
                    "mapping_completeness": "partial",
                    "review_confidence": "high",
                    "tracklet_ids_by_video": {"v1": ["v1_right"], "v2": ["v2_right"]},
                },
            },
        },
    )
    _write_jsonl(
        result / "entities.jsonl",
        [
            {"entity_id": "crossed", "tracklet_ids": ["v1_left", "bridge", "v2_right"]},
            {"entity_id": "left-fragment", "tracklet_ids": ["v1_left_fragment", "v2_left"]},
            {"entity_id": "right-fragment", "tracklet_ids": ["v1_right"]},
        ],
    )
    _write_jsonl(
        result / "accepted-links.jsonl",
        [
            {"tracklet_a": "v1_left", "tracklet_b": "v2_right", "mode": "automatic", "score": 0.99},
            {"tracklet_a": "v1_left_fragment", "tracklet_b": "v2_left", "mode": "automatic", "score": 0.98},
        ],
    )
    return template, review, result


def test_visual_draft_detects_direct_and_transitive_crossing_without_claiming_g2(tmp_path):
    template, review, result = _fixture(tmp_path)
    summary = audit_hard_negatives(template_path=template, review_path=review, result_dir=result)

    assert summary["review_status"] == "visual_draft_pending_data_owner"
    assert summary["diagnostic_only"] is True
    assert summary["hard_negative_evaluated"] is False
    assert summary["g2_evaluated"] is False
    assert summary["opposite_merge_groups"] == ["A"]
    assert summary["groups"][0]["verdict"] == "VISUAL_DRAFT_CROSSING_OBSERVED_PARTIAL_MAPPING"
    assert summary["groups"][0]["direct_accepted_crossings"][0]["score"] == 0.99
    assert summary["groups"][0]["entity_crossings"][0]["entity_id"] == "crossed"
    assert summary["anchors"]["left"]["same_video_extra_fragment_count"] == 1
    assert summary["anchors"]["left"]["minimum_entity_count_under_one_track_per_video"] == 2
    assert summary["anchors"]["left"]["excess_output_entity_count_after_mutex_floor"] == 0
    assert len(summary["anchors"]["left"]["same_anchor_accepted_links"]) == 1


def test_data_owner_confirmation_still_needs_complete_pair_mapping(tmp_path):
    template, review, result = _fixture(tmp_path, review_status="data_owner_confirmed")
    summary = audit_hard_negatives(template_path=template, review_path=review, result_dir=result)

    assert summary["hard_negative_evaluated"] is False
    assert summary["g2_evaluated"] is False
    assert summary["data_owner_confirmation_required"] is False
    assert summary["mapping_completion_required"] is True
    assert summary["groups"][0]["verdict"] == "CONFIRMED_CROSSING_OBSERVED_PARTIAL_MAPPING"


def test_complete_data_owner_pair_mapping_can_grade_hard_negative_but_not_full_g2(tmp_path):
    template, review, result = _fixture(tmp_path, review_status="data_owner_confirmed")
    value = json.loads(review.read_text())
    for anchor in value["anchors"].values():
        anchor["mapping_completeness"] = "complete"
    _write_json(review, value)

    summary = audit_hard_negatives(template_path=template, review_path=review, result_dir=result)

    assert summary["diagnostic_only"] is False
    assert summary["hard_negative_evaluated"] is True
    assert summary["g2_evaluated"] is False
    assert summary["mapping_completion_required"] is False
    assert summary["groups"][0]["verdict"] == "CONFIRMED_CROSSING_OBSERVED"


def test_rejects_tracklet_assigned_to_two_anchors(tmp_path):
    template, review, result = _fixture(tmp_path)
    value = json.loads(review.read_text())
    value["anchors"]["right"]["tracklet_ids_by_video"]["v1"].append("v1_left")
    _write_json(review, value)

    with pytest.raises(ValueError, match="assigned to both"):
        audit_hard_negatives(template_path=template, review_path=review, result_dir=result)

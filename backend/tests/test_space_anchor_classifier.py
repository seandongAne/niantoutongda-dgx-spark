from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from backend.tools.spatial import SpatialObservation
from scripts.space_anchor_classifier import (
    TrackEvidence,
    _candidate_from_prediction,
    _expected_anchors,
    automatic_visual_instance_ids,
    build_contact_sheet,
    parse_prediction,
)


def _observation(
    track: str,
    frame: str,
    bbox: tuple[float, float, float, float],
    *,
    timestamp: int = 0,
) -> SpatialObservation:
    return SpatialObservation(
        video_id="new",
        timestamp_ms=timestamp,
        frame_ref=frame,
        bbox=bbox,
        region_track_id=track,
        anchor_label="automatic_proposal",
        support_type="surface",
        capacity_class="medium",
        model_confidence=0.8,
        power_state="UNKNOWN",
    )


def test_parallel_category_tracks_share_automatic_visual_instance():
    grouped = {
        "t-a": [
            _observation("t-a", "kf_000001.jpg", (10, 10, 90, 90)),
            _observation("t-a", "kf_000002.jpg", (12, 10, 92, 90)),
        ],
        "t-b": [
            _observation("t-b", "kf_000001.jpg", (11, 10, 91, 90)),
            _observation("t-b", "kf_000002.jpg", (13, 10, 93, 90)),
        ],
        "t-c": [
            _observation("t-c", "kf_000001.jpg", (120, 10, 180, 90)),
            _observation("t-c", "kf_000002.jpg", (122, 10, 182, 90)),
        ],
    }

    instances = automatic_visual_instance_ids(grouped)

    assert instances["t-a"] == instances["t-b"]
    assert instances["t-c"] != instances["t-a"]


def test_contact_sheet_uses_automatic_frame_and_crop(tmp_path, monkeypatch):
    frame = tmp_path / "kf_000010.jpg"
    Image.new("RGB", (320, 180), (230, 230, 230)).save(frame)
    monkeypatch.setattr("scripts.space_anchor_classifier.PROJ", tmp_path)
    observation = _observation("t-a", frame.name, (80, 40, 240, 150), timestamp=100)
    evidence = TrackEvidence(
        track_id="t-a",
        observations=(observation,),
        prototype_refs=(),
        hero_ref=None,
        visual_instance_id="auto_visual_x",
    )

    encoded, sources = build_contact_sheet(evidence)

    assert encoded.startswith(b"\xff\xd8")
    assert sources == [str(frame)]
    with Image.open(Path(frame)) as source:
        assert source.size == (320, 180)


def test_strict_prediction_parser_and_candidate_projection():
    anchors = ["study_desk", "wall_shelf"]
    prediction = {
        "anchor_scores": {"study_desk": 91, "wall_shelf": 4, "other": 5},
        "best_anchor": "study_desk",
        "display_name_zh": "学习桌面",
        "support_type": "surface",
        "support_confidence": 94,
        "capacity_class": "medium",
        "capacity_confidence": 88,
    }
    parsed = parse_prediction(json.dumps(prediction), anchors)
    assert parsed == prediction
    observation = _observation("t-a", "kf_000001.jpg", (0, 0, 100, 100))
    evidence = TrackEvidence(
        track_id="t-a",
        observations=(observation,) * 5,
        prototype_refs=(),
        hero_ref=None,
        visual_instance_id="auto_visual_x",
    )

    candidate = _candidate_from_prediction(
        evidence,
        parsed,
        contact_ref="results/space/evidence/t-a.jpg",
        contact_sha256="a" * 64,
        model="nemotron-test",
    )

    by_anchor = {item.anchor: item for item in candidate.anchor_hypotheses}
    assert by_anchor["study_desk"].mean_confidence == 0.91
    assert by_anchor["study_desk"].label_vote_count == 5
    assert by_anchor["study_desk"].support_type.value == "surface"
    assert by_anchor["study_desk"].capacity_class.value == "medium"
    assert candidate.power_state.value == "UNKNOWN"
    assert candidate.visual_instance_id == "auto_visual_x"


def test_parser_rejects_missing_scores_and_anchor_list_is_deterministic():
    assert parse_prediction(
        '{"anchor_scores":{"study_desk":90,"other":10}}',
        ["study_desk", "wall_shelf"],
    ) is None
    assert _expected_anchors(["wall_shelf,study_desk"]) == ["study_desk", "wall_shelf"]

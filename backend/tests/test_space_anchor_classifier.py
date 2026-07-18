from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from backend.tools.spatial import SpatialObservation
from scripts.space_anchor_classifier import (
    MAIN_MAX_TOKENS,
    Client,
    TrackEmbedding,
    TrackEvidence,
    _candidate_from_prediction,
    _classify_image,
    _expected_anchors,
    aggregate_view_predictions,
    automatic_visual_instance_ids,
    build_classification_views,
    build_contact_sheet,
    calibrate_prediction_geometry,
    classify_one,
    parse_prediction,
    parse_anchor_prediction,
    parse_hard_field_prediction,
    quarantine_low_information_prediction,
    semantic_visual_instance_ids,
)


def _observation(
    track: str,
    frame: str,
    bbox: tuple[float, float, float, float],
    *,
    timestamp: int = 0,
    anchor: str = "automatic_proposal",
    video: str = "new",
) -> SpatialObservation:
    return SpatialObservation(
        video_id=video,
        timestamp_ms=timestamp,
        frame_ref=frame,
        bbox=bbox,
        region_track_id=track,
        anchor_label=anchor,
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


def test_nested_tracks_merge_with_geometry_and_automatic_embedding():
    grouped = {
        "t-a": [
            _observation("t-a", f"kf_{index:06d}.jpg", (0, 0, 100, 20))
            for index in range(3)
        ],
        "t-b": [
            _observation("t-b", f"kf_{index:06d}.jpg", (1, 2, 99, 18))
            for index in range(3)
        ],
    }
    embeddings = {
        "t-a": TrackEmbedding(model="dino", vector=(1.0, 0.0)),
        "t-b": TrackEmbedding(model="dino", vector=(0.8, 0.6)),
    }

    instances = automatic_visual_instance_ids(grouped, embeddings=embeddings)

    assert instances["t-a"] == instances["t-b"]


def test_short_gap_motion_continuation_merges_same_detector_class():
    def box(center_x):
        return (center_x - 50, 0, center_x + 50, 200)

    grouped = {
        "t-a": [
            _observation(
                "t-a",
                f"a_{index}.jpg",
                box(center),
                timestamp=timestamp,
                anchor="display_cabinet",
            )
            for index, (timestamp, center) in enumerate(((0, 100), (100, 110), (200, 120)))
        ],
        "t-b": [
            _observation(
                "t-b",
                f"b_{index}.jpg",
                box(center),
                timestamp=timestamp,
                anchor="display_cabinet",
            )
            for index, (timestamp, center) in enumerate(
                ((1200, 220), (1300, 230), (1400, 240))
            )
        ],
    }
    embeddings = {
        "t-a": TrackEmbedding(model="dino", vector=(1.0, 0.0)),
        "t-b": TrackEmbedding(model="dino", vector=(0.9, 0.435889894)),
    }

    instances = automatic_visual_instance_ids(grouped, embeddings=embeddings)

    assert instances["t-a"] == instances["t-b"]


def test_far_apart_tracks_never_merge_even_with_identical_embedding():
    grouped = {
        "t-a": [
            _observation(
                "t-a",
                "a.jpg",
                (0, 0, 100, 200),
                timestamp=0,
                anchor="display_cabinet",
            )
        ],
        "t-b": [
            _observation(
                "t-b",
                "b.jpg",
                (0, 0, 100, 200),
                timestamp=80_000,
                anchor="display_cabinet",
            )
        ],
    }
    embeddings = {
        track: TrackEmbedding(model="dino", vector=(1.0, 0.0)) for track in grouped
    }

    instances = automatic_visual_instance_ids(grouped, embeddings=embeddings)

    assert instances["t-a"] != instances["t-b"]


def test_semantic_consensus_merges_part_and_whole_tracks():
    frames = [f"kf_{index:06d}.jpg" for index in range(3)]
    grouped = {
        "t-a": [
            _observation("t-a", frame, (0, 0, 200, 100), timestamp=index * 100)
            for index, frame in enumerate(frames)
        ],
        "t-b": [
            _observation("t-b", frame, (10, 20, 190, 80), timestamp=index * 100)
            for index, frame in enumerate(frames)
        ],
    }
    prediction = {
        "anchor_scores": {"vanity": 95, "wall_shelf": 0, "other": 5},
        "anchor_max_scores": {"vanity": 95, "wall_shelf": 0, "other": 5},
        "anchor_vote_counts": {"vanity": 3, "wall_shelf": 0, "other": 0},
        "best_anchor": "vanity",
        "display_name_zh": "花布台面",
        "support_type": "surface",
        "support_confidence": 90,
        "capacity_class": "medium",
        "capacity_confidence": 85,
        "view_count": 3,
    }
    candidates = []
    for track_id in ("t-a", "t-b"):
        evidence = TrackEvidence(
            track_id=track_id,
            observations=tuple(grouped[track_id]),
            prototype_refs=(),
            hero_ref=None,
            visual_instance_id=f"initial-{track_id}",
        )
        candidates.append(
            _candidate_from_prediction(
                evidence,
                prediction,
                contact_ref=f"{track_id}.jpg",
                contact_sha256="a" * 64,
                model="nemotron-test",
            )
        )

    instances = semantic_visual_instance_ids(
        grouped,
        candidates,
        {"t-a": "initial-a", "t-b": "initial-b"},
    )

    assert instances["t-a"] == instances["t-b"]


def test_semantic_consensus_never_merges_long_gap_fragments():
    grouped = {
        "t-a": [
            _observation("t-a", "a.jpg", (0, 0, 200, 100), timestamp=0),
        ],
        "t-b": [
            _observation("t-b", "b.jpg", (0, 0, 200, 100), timestamp=67_000),
        ],
    }
    prediction = {
        "anchor_scores": {"chest_of_drawers": 95, "other": 5},
        "anchor_max_scores": {"chest_of_drawers": 95, "other": 5},
        "anchor_vote_counts": {"chest_of_drawers": 3, "other": 0},
        "best_anchor": "chest_of_drawers",
        "display_name_zh": "红棕色多抽屉柜",
        "support_type": "surface",
        "support_confidence": 90,
        "capacity_class": "medium",
        "capacity_confidence": 85,
        "view_count": 3,
    }
    candidates = [
        _candidate_from_prediction(
            TrackEvidence(
                track_id=track_id,
                observations=tuple(grouped[track_id]),
                prototype_refs=(),
                hero_ref=None,
                visual_instance_id=f"initial-{track_id}",
            ),
            prediction,
            contact_ref=f"{track_id}.jpg",
            contact_sha256="a" * 64,
            model="nemotron-test",
        )
        for track_id in grouped
    ]
    embeddings = {
        "t-a": TrackEmbedding(model="dino", vector=(1.0, 0.0)),
        "t-b": TrackEmbedding(model="dino", vector=(0.6341, 0.7733)),
    }

    instances = semantic_visual_instance_ids(
        grouped,
        candidates,
        {"t-a": "initial-a", "t-b": "initial-b"},
        embeddings=embeddings,
    )

    assert instances["t-a"] != instances["t-b"]


def test_main_vlm_budget_covers_observed_nested_response(monkeypatch):
    client = Client(
        "http://local",
        "nemotron-test",
        ["study_desk"],
        True,
        anchor_descriptions={"study_desk": "requested covered console"},
    )
    captured = {}

    def fake_chat(image_bytes, *, prompt, schema, max_tokens):
        captured["max_tokens"] = max_tokens
        return "{}"

    monkeypatch.setattr(client, "_chat", fake_chat)
    client.chat(b"image")

    assert MAIN_MAX_TOKENS >= 500
    assert captured["max_tokens"] == MAIN_MAX_TOKENS
    assert "requested covered console" in client.prompt


def test_prediction_cache_key_includes_target_prompt(tmp_path, monkeypatch):
    raw = json.dumps(
        {
            "anchor_scores": {"study_desk": 90, "other": 10},
            "best_anchor": "study_desk",
            "display_name_zh": "学习桌",
            "support_type": "surface",
            "support_confidence": 90,
            "capacity_class": "medium",
            "capacity_confidence": 90,
        }
    )
    cache = {}
    cache_path = tmp_path / "cache.jsonl"
    lock = __import__("threading").Lock()
    calls = []

    def client(description):
        result = Client(
            "http://local",
            "nemotron-test",
            ["study_desk"],
            True,
            anchor_descriptions={"study_desk": description},
        )

        def chat(_image_bytes):
            calls.append(description)
            return raw

        monkeypatch.setattr(result, "chat", chat)
        return result

    first, first_hit, *_ = _classify_image(
        client("plain writing desk"),
        b"same-image",
        cache=cache,
        cache_path=cache_path,
        cache_lock=lock,
        track_id="t-a",
        view_index=1,
    )
    second, second_hit, *_ = _classify_image(
        client("floral covered writing desk"),
        b"same-image",
        cache=cache,
        cache_path=cache_path,
        cache_lock=lock,
        track_id="t-a",
        view_index=1,
    )

    assert first is not None and second is not None
    assert first_hit is False and second_hit is False
    assert calls == ["plain writing desk", "floral covered writing desk"]


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


def test_classification_views_are_independent_target_crops(tmp_path, monkeypatch):
    refs = []
    colors = ((220, 10, 10), (10, 220, 10), (10, 10, 220))
    for index, color in enumerate(colors, 1):
        path = tmp_path / f"crop-{index}.jpg"
        Image.new("RGB", (120 + index, 80 + index), color).save(path)
        refs.append(path.name)
    frame = tmp_path / "kf_000010.jpg"
    Image.new("RGB", (320, 180), (230, 230, 230)).save(frame)
    monkeypatch.setattr("scripts.space_anchor_classifier.PROJ", tmp_path)
    evidence = TrackEvidence(
        track_id="t-a",
        observations=(_observation("t-a", frame.name, (80, 40, 240, 150)),),
        prototype_refs=tuple(refs),
        hero_ref=None,
        visual_instance_id="auto_visual_x",
    )

    views = build_classification_views(evidence)

    assert len(views) == 3
    assert [Path(ref).name for _, ref, _ in views] == refs
    assert all(encoded.startswith(b"\xff\xd8") for encoded, _, _ in views)
    assert all(target_pixels > 0 for _, _, target_pixels in views)


def test_classification_fallback_reports_target_not_context_pixels(
    tmp_path, monkeypatch
):
    frame = tmp_path / "kf_000010.jpg"
    Image.new("RGB", (320, 180), (230, 230, 230)).save(frame)
    monkeypatch.setattr("scripts.space_anchor_classifier.PROJ", tmp_path)
    evidence = TrackEvidence(
        track_id="t-a",
        observations=(_observation("t-a", frame.name, (80, 40, 240, 150)),),
        prototype_refs=(),
        hero_ref=None,
        visual_instance_id="auto_visual_x",
    )

    views = build_classification_views(evidence)

    assert len(views) == 1
    assert views[0][2] == 160 * 110


def test_view_aggregation_uses_real_best_anchor_votes():
    predictions = [
        {
            "anchor_scores": {"study_desk": 90, "wall_shelf": 5, "other": 5},
            "best_anchor": "study_desk",
            "display_name_zh": "学习桌一",
            "support_type": "surface",
            "support_confidence": 90,
            "capacity_class": "medium",
            "capacity_confidence": 80,
        },
        {
            "anchor_scores": {"study_desk": 80, "wall_shelf": 10, "other": 10},
            "best_anchor": "study_desk",
            "display_name_zh": "学习桌二",
            "support_type": "surface",
            "support_confidence": 80,
            "capacity_class": "medium",
            "capacity_confidence": 90,
        },
        {
            "anchor_scores": {"study_desk": 10, "wall_shelf": 5, "other": 85},
            "best_anchor": "other",
            "display_name_zh": "其他",
            "support_type": "unknown",
            "support_confidence": 20,
            "capacity_class": "unknown",
            "capacity_confidence": 20,
        },
    ]

    aggregated = aggregate_view_predictions(
        predictions, ["study_desk", "wall_shelf"]
    )

    assert aggregated["view_count"] == 3
    assert aggregated["anchor_vote_counts"] == {
        "other": 1,
        "study_desk": 2,
        "wall_shelf": 0,
    }
    assert aggregated["anchor_scores"]["study_desk"] == 60
    assert aggregated["anchor_max_scores"]["study_desk"] == 90
    assert aggregated["support_type"] == "surface"
    assert aggregated["capacity_class"] == "medium"

    evidence = TrackEvidence(
        track_id="t-a",
        observations=(
            _observation("t-a", "kf_000001.jpg", (0, 0, 100, 100)),
        )
        * 5,
        prototype_refs=(),
        hero_ref=None,
        visual_instance_id="auto_visual_x",
    )
    candidate = _candidate_from_prediction(
        evidence,
        aggregated,
        contact_ref="audit.jpg",
        contact_sha256="a" * 64,
        model="nemotron-test",
    )
    hypotheses = {item.anchor: item for item in candidate.anchor_hypotheses}
    assert candidate.observation_count == 5
    assert candidate.semantic_observation_count == 3
    assert hypotheses["study_desk"].label_vote_count == 2
    assert hypotheses["wall_shelf"].label_vote_count == 0


def test_thin_shelf_geometry_calibrates_one_narrow_support_to_small():
    evidence = TrackEvidence(
        track_id="t-shelf",
        observations=tuple(
            _observation(
                "t-shelf",
                f"kf_{index:06d}.jpg",
                (0, index, 500, 50 + index),
            )
            for index in range(3)
        ),
        prototype_refs=(),
        hero_ref=None,
        visual_instance_id="auto_visual_shelf",
    )
    prediction = {
        "support_type": "shelf",
        "support_confidence": 95,
        "capacity_class": "medium",
        "capacity_confidence": 85,
    }

    projected, calibrations = calibrate_prediction_geometry(evidence, prediction)

    assert projected["capacity_class"] == "small"
    assert projected["capacity_confidence"] == 85
    assert calibrations[0]["median_bbox_aspect_ratio"] == 10.0


def test_low_information_quarantine_preserves_candidate_but_removes_votes():
    prediction = {
        "anchor_scores": {"vanity": 95, "wall_shelf": 0, "other": 0},
        "anchor_max_scores": {"vanity": 95, "wall_shelf": 0, "other": 0},
        "anchor_vote_counts": {"vanity": 3, "wall_shelf": 0, "other": 0},
        "best_anchor": "vanity",
        "display_name_zh": "数码钢琴",
        "support_type": "surface",
        "support_confidence": 95,
        "capacity_class": "medium",
        "capacity_confidence": 85,
        "view_count": 3,
    }

    quarantined = quarantine_low_information_prediction(prediction)

    assert quarantined["best_anchor"] == "other"
    assert quarantined["anchor_vote_counts"]["vanity"] == 0
    assert quarantined["anchor_scores"]["vanity"] == 0
    assert quarantined["support_type"] == "unknown"
    assert quarantined["capacity_class"] == "unknown"


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


def test_parser_accepts_model_display_name_alias_without_relaxing_hard_fields():
    parsed = parse_prediction(
        json.dumps(
            {
                "anchor_scores": {"study_desk": 85, "other": 0},
                "best_anchor": "study_desk",
                "display_name": "学习桌",
                "support_type": "surface",
                "support_confidence": 90,
                "capacity_class": "medium",
                "capacity_confidence": 80,
            }
        ),
        ["study_desk"],
    )

    assert parsed is not None
    assert parsed["display_name_zh"] == "学习桌"


def test_parser_normalizes_nested_nemotron_hard_fields():
    parsed = parse_prediction(
        json.dumps(
            {
                "target_object": "study_desk",
                "anchor_scores": {"study_desk": 90, "other": 5},
                "best_anchor": "study_desk",
                "support_type": {"name": "surface", "confidence": 85},
                "capacity_class": {"name": "medium", "confidence": 75},
            }
        ),
        ["study_desk"],
    )

    assert parsed == {
        "anchor_scores": {"other": 5, "study_desk": 90},
        "best_anchor": "study_desk",
        "display_name_zh": "学习桌面",
        "support_type": "surface",
        "support_confidence": 85,
        "capacity_class": "medium",
        "capacity_confidence": 75,
    }


def test_missing_hard_confidences_can_be_repaired_by_independent_second_call():
    first = json.dumps(
        {
            "anchor_scores": {"chest_of_drawers": 85, "other": 0},
            "best_anchor": "chest_of_drawers",
            "support_type": "surface",
            "capacity_class": "medium",
        }
    )
    repair = json.dumps(
        {
            "support_type": "surface",
            "support_confidence": 88,
            "capacity_class": "medium",
            "capacity_confidence": 81,
        }
    )

    assert parse_prediction(first, ["chest_of_drawers"]) is None
    anchor = parse_anchor_prediction(first, ["chest_of_drawers"])
    hard = parse_hard_field_prediction(repair)
    assert anchor is not None and hard is not None
    assert {**anchor, **hard}["capacity_confidence"] == 81


def test_parser_normalizes_observed_nemotron_omissions_and_description_aliases():
    parsed = parse_prediction(
        json.dumps(
            {
                "anchor_scores": {
                    "study_desk": 100,
                    "wall_shelf": 0,
                },
                "best_anchor": "study_desk",
                "chinese_display_name": "学习桌",
                "support_type": {"description": "surface", "confidence": 95},
                "capacity_class": {"description": "medium", "confidence": 90},
            }
        ),
        ["study_desk", "wall_shelf"],
    )

    assert parsed == {
        "anchor_scores": {"other": 0, "study_desk": 100, "wall_shelf": 0},
        "best_anchor": "study_desk",
        "display_name_zh": "学习桌",
        "support_type": "surface",
        "support_confidence": 95,
        "capacity_class": "medium",
        "capacity_confidence": 90,
    }


def test_parser_normalizes_observed_flat_score_shape():
    parsed = parse_prediction(
        json.dumps(
            {
                "study_desk": 5,
                "wall_shelf": 85,
                "best_anchor": "wall_shelf",
                "support_type": {"name": "shelf", "confidence": 90},
                "capacity_class": {"name": "small", "confidence": 80},
            }
        ),
        ["study_desk", "wall_shelf"],
    )

    assert parsed is not None
    assert parsed["anchor_scores"] == {
        "other": 0,
        "study_desk": 5,
        "wall_shelf": 85,
    }
    assert parsed["best_anchor"] == "wall_shelf"


def test_anchor_parser_normalizes_observed_per_anchor_object_shape():
    raw = json.dumps(
        {
            "study_desk": {
                "anchor_score": 5,
                "support_type": "unknown",
                "capacity_class": "medium",
                "display_name": "小型书桌",
            },
            "wall_shelf": {
                "anchor_score": 85,
                "support_type": "shelf",
                "capacity_class": "small",
                "display_name": "墙面置物架",
            },
        }
    )

    assert parse_prediction(raw, ["study_desk", "wall_shelf"]) is None
    parsed = parse_anchor_prediction(raw, ["study_desk", "wall_shelf"])
    assert parsed == {
        "anchor_scores": {"other": 0, "study_desk": 5, "wall_shelf": 85},
        "best_anchor": "wall_shelf",
        "display_name_zh": "墙面置物架",
    }


def test_unresolved_hard_fields_are_preserved_but_assignment_ineligible(
    tmp_path, monkeypatch
):
    frame = tmp_path / "kf_000001.jpg"
    second = tmp_path / "crop_000002.jpg"
    Image.new("RGB", (320, 180), (230, 230, 230)).save(frame)
    Image.new("RGB", (200, 120), (220, 220, 220)).save(second)
    monkeypatch.setattr("scripts.space_anchor_classifier.PROJ", tmp_path)
    raw = json.dumps(
        {
            "study_desk": {
                "anchor_score": 90,
                "support_type": "surface",
                "capacity_class": "medium",
                "display_name": "学习桌",
            }
        }
    )

    class FakeClient:
        anchors = ["study_desk"]
        model = "nemotron-test"
        schema = {"type": "object"}
        prompt = "classify the central target"
        guided = True

        def chat(self, image_bytes):
            return raw

        def chat_hard_fields(self, image_bytes):
            return raw

    evidence = TrackEvidence(
        track_id="t-a",
        observations=(_observation("t-a", frame.name, (10, 10, 200, 150)),),
        prototype_refs=(frame.name, second.name),
        hero_ref=None,
        visual_instance_id="auto_visual_x",
    )
    candidate, diagnostic = classify_one(
        FakeClient(),
        evidence,
        out_dir=tmp_path / "out",
        evidence_dir=tmp_path / "evidence",
        cache={},
        cache_path=tmp_path / "cache.jsonl",
        cache_lock=__import__("threading").Lock(),
    )

    assert candidate is not None
    assert diagnostic["status"] == "HARD_FIELDS_UNRESOLVED"
    hypothesis = candidate.anchor_hypotheses[0]
    assert hypothesis.support_type is None
    assert hypothesis.support_confidence == 0
    assert hypothesis.capacity_class is None
    assert hypothesis.capacity_confidence == 0


def test_low_information_views_remain_diagnostic_but_cannot_be_assigned(
    tmp_path, monkeypatch
):
    frame = tmp_path / "kf_000001.jpg"
    tiny_a = tmp_path / "crop_000002.jpg"
    tiny_b = tmp_path / "crop_000003.jpg"
    Image.new("RGB", (320, 180), (230, 230, 230)).save(frame)
    Image.new("RGB", (100, 100), (220, 220, 220)).save(tiny_a)
    Image.new("RGB", (100, 100), (210, 210, 210)).save(tiny_b)
    monkeypatch.setattr("scripts.space_anchor_classifier.PROJ", tmp_path)
    raw = json.dumps(
        {
            "anchor_scores": {"vanity": 95, "other": 5},
            "best_anchor": "vanity",
            "display_name_zh": "数码钢琴",
            "support_type": "surface",
            "support_confidence": 95,
            "capacity_class": "medium",
            "capacity_confidence": 85,
        }
    )

    class FakeClient:
        anchors = ["vanity"]
        model = "nemotron-test"
        schema = {"type": "object"}
        prompt = "classify the central target"
        guided = True

        def chat(self, image_bytes):
            return raw

        def chat_hard_fields(self, image_bytes):
            raise AssertionError("hard-field repair should not run")

    evidence = TrackEvidence(
        track_id="t-small",
        observations=(_observation("t-small", frame.name, (10, 10, 200, 150)),),
        prototype_refs=(tiny_a.name, tiny_b.name),
        hero_ref=None,
        visual_instance_id="auto_visual_small",
    )
    candidate, diagnostic = classify_one(
        FakeClient(),
        evidence,
        out_dir=tmp_path / "out",
        evidence_dir=tmp_path / "evidence",
        cache={},
        cache_path=tmp_path / "cache.jsonl",
        cache_lock=__import__("threading").Lock(),
    )

    assert candidate is not None
    assert diagnostic["status"] == "LOW_INFORMATION_VIEWS"
    assert diagnostic["valid_view_count"] == 3
    assert diagnostic["informative_view_count"] == 1
    assert diagnostic["raw_aggregate_prediction"]["best_anchor"] == "vanity"
    assert [item["target_pixel_area"] for item in diagnostic["views"]] == [
        10_000,
        10_000,
        26_600,
    ]
    hypothesis = candidate.anchor_hypotheses[0]
    assert hypothesis.label_vote_count == 0
    assert hypothesis.mean_confidence == 0
    assert hypothesis.support_type is None
    assert hypothesis.capacity_class is None

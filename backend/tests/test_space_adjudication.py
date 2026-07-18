from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.tools.spatial import (
    GateStatus,
    SpatialProducerConfig,
    VisualAdjudicationReview,
    adjudicate_spatial_regions,
    produce_spatial_regions,
    write_spatial_adjudication_outputs,
)
from scripts.space_adjudication_task import main as adjudication_task_main


ANCHORS = ["study_desk", "wall_shelf", "vanity_top", "chest_top", "display_cabinet"]
VIDEO_SHA256 = "b" * 64


def _source(*, observed_anchors: list[str] | None = None):
    observed_anchors = observed_anchors or ANCHORS
    observations = []
    for index, anchor in enumerate(observed_anchors):
        observations.append(
            {
                "video_id": "new_1",
                "timestamp_ms": 1000 + index * 100,
                "frame_ref": f"source-{index}.jpg",
                "bbox": [index * 20, 10, index * 20 + 15, 30],
                "region_track_id": f"track-{anchor}",
                "anchor_label": anchor,
                "display_name_zh": anchor,
                "support_type": "surface",
                "capacity_class": "medium",
                "power_state": "UNKNOWN",
                "model_confidence": 0.95,
                "model_version": "test-spatial-v1",
            }
        )
    result = produce_spatial_regions(
        "new_1",
        observations,
        SpatialProducerConfig(
            min_regions=5,
            min_observations_per_region=2,
            expected_anchor_labels=ANCHORS,
        ),
    )
    return result.candidate_manifest, result.normalized_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_for_anchor(manifest, anchor: str):
    return next(candidate for candidate in manifest.candidates if candidate.anchor == anchor)


def _review_payload(
    root: Path,
    manifest,
    source_hash: str,
    *,
    create_missing: bool = False,
) -> dict:
    frame_dir = root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    decisions = []
    for index, anchor in enumerate(ANCHORS):
        frame = frame_dir / f"{anchor}.jpg"
        frame.write_bytes(f"visual evidence {anchor}".encode("utf-8"))
        candidate = _candidate_for_anchor(manifest, anchor)
        missing = candidate.observation_count == 0
        operation = "CREATE_FROM_FRAME" if missing and create_missing else "KEEP_LABEL"
        decisions.append(
            {
                "decision_id": f"decision-{index + 1:02d}",
                "source_candidate_region_ids": [candidate.region_id],
                "source_track_ids": [] if missing else list(candidate.source_track_ids),
                "operation": operation,
                "status": "VISUALLY_ADJUDICATED",
                "output_region_id": f"region_{anchor}",
                "anchor": anchor,
                "display_name_zh": f"区域 {anchor}",
                "support_type": "surface",
                "capacity_class": "medium",
                "power_state": "UNKNOWN",
                "visual_instance_id": f"visual-instance-{index + 1:02d}",
                "evidence": [
                    {
                        "video_id": "new_1",
                        "timestamp_ms": 54000 + index * 1000,
                        "frame_ref": str(frame.relative_to(root)),
                        "frame_sha256": _sha256(frame),
                        "bbox": [10, 20, 200, 300],
                    }
                ],
                "power_evidence": [],
                "reason_codes": ["AGENT_VISUAL_MATCH"],
                "note_zh": "逐帧视觉核对",
            }
        )
    return {
        "review_id": "hero-s1-space-visual-v1",
        "source_video_id": "new_1",
        "source_video_sha256": VIDEO_SHA256,
        "source_spatial_normalized_hash": source_hash,
        "reviewer_kind": "AGENT_VISUAL",
        "reviewer_id": "codex-visual-review",
        "authorization_ref": "task:user-authorized-agent-visual-review",
        "expected_anchor_labels": list(ANCHORS),
        "decisions": decisions,
    }


def test_visual_overlay_projects_five_regions_with_auditable_provenance(tmp_path):
    manifest, source_hash = _source()
    payload = _review_payload(tmp_path, manifest, source_hash)

    result = adjudicate_spatial_regions(
        manifest,
        source_hash,
        payload,
        project_root=tmp_path,
    )
    reordered = adjudicate_spatial_regions(
        manifest,
        source_hash,
        {**payload, "decisions": list(reversed(payload["decisions"]))},
        project_root=tmp_path,
    )

    assert result.gate_passed
    assert result.metrics.gate_status is GateStatus.PASS
    assert result.metrics.visually_adjudicated_count == 5
    assert result.metrics.projected_region_count == 5
    assert result.normalized_hash == reordered.normalized_hash
    assert result.region_manifest is not None
    assert len(result.region_manifest.entries) == 5
    assert "review=hero-s1-space-visual-v1" in result.region_manifest.notes
    assert source_hash in result.region_manifest.notes
    assert "UNKNOWN power projected near_power=false" in result.region_manifest.notes
    assert result.metrics.power_state_counts == {"UNKNOWN": 5}
    assert all(not entry.region_id.startswith("auto_") for entry in result.region_manifest.entries)
    assert all(
        any(ref.startswith("agent_review:hero-s1-space-visual-v1/") for ref in entry.evidence_refs)
        for entry in result.region_manifest.entries
    )
    assert "AUTO_ACCEPTED" not in json.dumps(
        result.model_dump(mode="json"), ensure_ascii=False
    )

    written = write_spatial_adjudication_outputs(result, tmp_path / "out")
    assert set(written) == {
        "adjudication_manifest",
        "metrics",
        "normalized_hash",
        "region_manifest",
    }
    assert {path.name for path in (tmp_path / "out").iterdir()} == {
        "adjudication_manifest.json",
        "metrics.json",
        "normalized.sha256",
        "regions.json",
    }


def test_create_from_not_observed_expected_placeholder_can_close_coverage(tmp_path):
    manifest, source_hash = _source(observed_anchors=ANCHORS[:-1])
    payload = _review_payload(
        tmp_path,
        manifest,
        source_hash,
        create_missing=True,
    )

    result = adjudicate_spatial_regions(
        manifest,
        source_hash,
        payload,
        project_root=tmp_path,
    )

    assert result.gate_passed
    missing_candidate = _candidate_for_anchor(manifest, ANCHORS[-1])
    assert missing_candidate.source_track_ids == []
    decision = next(
        item
        for item in result.adjudication_manifest.decisions
        if item.anchor == ANCHORS[-1]
    )
    assert decision.operation.value == "CREATE_FROM_FRAME"


@pytest.mark.parametrize("power_state", ["NEAR", "NOT_NEAR"])
def test_claimed_power_state_requires_power_evidence(power_state, tmp_path):
    manifest, source_hash = _source()
    payload = _review_payload(tmp_path, manifest, source_hash)
    payload["decisions"][0]["power_state"] = power_state

    with pytest.raises(ValidationError, match="requires explicit power_evidence"):
        VisualAdjudicationReview.model_validate(payload)


def test_power_evidence_is_verified_and_projected(tmp_path):
    manifest, source_hash = _source()
    payload = _review_payload(tmp_path, manifest, source_hash)
    decision = payload["decisions"][0]
    decision["power_state"] = "NEAR"
    decision["power_evidence"] = [copy.deepcopy(decision["evidence"][0])]

    result = adjudicate_spatial_regions(
        manifest,
        source_hash,
        payload,
        project_root=tmp_path,
    )

    assert result.region_manifest is not None
    region = next(entry for entry in result.region_manifest.entries if entry.anchor == ANCHORS[0])
    assert region.near_power is True
    assert any(ref.startswith("visual-power:") for ref in region.evidence_refs)


def test_auto_accepted_status_and_reserved_output_prefix_are_rejected(tmp_path):
    manifest, source_hash = _source()
    auto_status = _review_payload(tmp_path / "status", manifest, source_hash)
    auto_status["decisions"][0]["status"] = "AUTO_ACCEPTED"
    with pytest.raises(ValidationError, match="AUTO_ACCEPTED is forbidden"):
        VisualAdjudicationReview.model_validate(auto_status)

    auto_output = _review_payload(tmp_path / "output", manifest, source_hash)
    auto_output["decisions"][0]["output_region_id"] = "auto_spoofed_01"
    with pytest.raises(ValidationError, match="reserved auto_ prefix"):
        VisualAdjudicationReview.model_validate(auto_output)


@pytest.mark.parametrize(
    ("duplicate_field", "error"),
    [
        ("output_region_id", "output_region_id duplicates"),
        ("visual_instance_id", "visual_instance_id duplicates"),
        ("source_candidate_region_ids", "source candidate duplicates"),
    ],
)
def test_duplicate_region_instance_or_candidate_binding_fails(
    duplicate_field, error, tmp_path
):
    manifest, source_hash = _source()
    payload = _review_payload(tmp_path, manifest, source_hash)
    if duplicate_field == "source_candidate_region_ids":
        payload["decisions"][1][duplicate_field] = list(
            payload["decisions"][0][duplicate_field]
        )
    else:
        payload["decisions"][1][duplicate_field] = payload["decisions"][0][
            duplicate_field
        ]

    with pytest.raises(ValidationError, match=error):
        VisualAdjudicationReview.model_validate(payload)


def test_unknown_candidate_track_or_source_hash_fails_closed(tmp_path):
    manifest, source_hash = _source()

    unknown_candidate = _review_payload(
        tmp_path / "candidate", manifest, source_hash
    )
    unknown_candidate["decisions"][0]["source_candidate_region_ids"] = [
        "missing-candidate"
    ]
    with pytest.raises(ValueError, match="unknown source candidate"):
        adjudicate_spatial_regions(
            manifest,
            source_hash,
            unknown_candidate,
            project_root=tmp_path / "candidate",
        )

    unknown_track = _review_payload(tmp_path / "track", manifest, source_hash)
    unknown_track["decisions"][0]["source_track_ids"] = ["missing-track"]
    with pytest.raises(ValueError, match="unknown source_track_ids"):
        adjudicate_spatial_regions(
            manifest,
            source_hash,
            unknown_track,
            project_root=tmp_path / "track",
        )

    wrong_hash = _review_payload(tmp_path / "hash", manifest, "c" * 64)
    with pytest.raises(ValueError, match="does not match source hash"):
        adjudicate_spatial_regions(
            manifest,
            source_hash,
            wrong_hash,
            project_root=tmp_path / "hash",
        )


def test_missing_or_modified_evidence_frame_fails_sha_validation(tmp_path):
    manifest, source_hash = _source()
    payload = _review_payload(tmp_path, manifest, source_hash)
    frame_ref = payload["decisions"][0]["evidence"][0]["frame_ref"]
    (tmp_path / frame_ref).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        adjudicate_spatial_regions(
            manifest,
            source_hash,
            payload,
            project_root=tmp_path,
        )

    (tmp_path / frame_ref).unlink()
    with pytest.raises(ValueError, match="does not exist"):
        adjudicate_spatial_regions(
            manifest,
            source_hash,
            payload,
            project_root=tmp_path,
        )


def test_expected_anchor_coverage_is_exact_and_failed_gate_removes_stale_regions(tmp_path):
    manifest, source_hash = _source()
    mismatched = _review_payload(tmp_path / "mismatch", manifest, source_hash)
    mismatched["expected_anchor_labels"][-1] = "unexpected_anchor"
    with pytest.raises(ValueError, match="must exactly match"):
        adjudicate_spatial_regions(
            manifest,
            source_hash,
            mismatched,
            project_root=tmp_path / "mismatch",
        )

    deferred = _review_payload(tmp_path / "deferred", manifest, source_hash)
    deferred["decisions"][-1].update(
        operation="DEFER",
        status="NEEDS_USER",
    )
    result = adjudicate_spatial_regions(
        manifest,
        source_hash,
        deferred,
        project_root=tmp_path / "deferred",
    )
    assert not result.gate_passed
    assert result.region_manifest is None
    assert "expected_five_adjudicated_regions:4/5" in result.metrics.gate_reasons
    assert "needs_user_decisions_present:1" in result.metrics.gate_reasons

    out = tmp_path / "out"
    out.mkdir()
    (out / "regions.json").write_text("stale", encoding="utf-8")
    written = write_spatial_adjudication_outputs(result, out)
    assert not (out / "regions.json").exists()
    assert set(written) == {
        "adjudication_manifest",
        "metrics",
        "normalized_hash",
    }


def test_cli_removes_stale_regions_before_malformed_review_failure(tmp_path):
    manifest, source_hash = _source()
    payload = _review_payload(tmp_path, manifest, "c" * 64)
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    source_hash_path = tmp_path / "normalized.sha256"
    source_hash_path.write_text(source_hash + "\n", encoding="ascii")
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "cli-out"
    out.mkdir()
    (out / "regions.json").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match source hash"):
        adjudication_task_main(
            [
                "--candidates",
                str(candidates_path),
                "--source-hash",
                str(source_hash_path),
                "--review",
                str(review_path),
                "--out-dir",
                str(out),
            ]
        )

    assert not (out / "regions.json").exists()

"""Audit four S3 hard-negative pairs without promoting a visual draft to truth.

The audit joins three small artifacts: the frozen anchor-pair template, a human
review mapping, and one ReID result directory.  A review may explicitly be a
``visual_draft_pending_data_owner`` or ``data_owner_confirmed``.  The former is
diagnostic evidence only; neither status is sufficient to calculate the full
17-anchor G2 gate in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VISUAL_DRAFT = "visual_draft_pending_data_owner"
DATA_OWNER_CONFIRMED = "data_owner_confirmed"
_REVIEW_STATUSES = {VISUAL_DRAFT, DATA_OWNER_CONFIRMED}


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _flatten_anchor(anchor_id: str, raw: dict[str, Any]) -> tuple[dict[str, list[str]], set[str]]:
    by_video_raw = raw.get("tracklet_ids_by_video", {})
    if not isinstance(by_video_raw, dict):
        raise ValueError(f"{anchor_id}: tracklet_ids_by_video must be an object")
    by_video: dict[str, list[str]] = {}
    all_ids: set[str] = set()
    for video_id, values in sorted(by_video_raw.items()):
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"{anchor_id}/{video_id}: expected a list of tracklet ids")
        unique = sorted(set(values))
        for tracklet_id in unique:
            if not tracklet_id.startswith(f"{video_id}_"):
                raise ValueError(
                    f"{anchor_id}/{video_id}: tracklet {tracklet_id!r} has the wrong video prefix"
                )
            if tracklet_id in all_ids:
                raise ValueError(f"{anchor_id}: duplicate tracklet {tracklet_id!r}")
            all_ids.add(tracklet_id)
        if unique:
            by_video[str(video_id)] = unique
    return by_video, all_ids


def audit_hard_negatives(
    *,
    template_path: str | Path,
    review_path: str | Path,
    result_dir: str | Path,
) -> dict[str, Any]:
    """Return a deterministic hard-negative diagnostic summary.

    ``visual_draft_pending_data_owner`` results deliberately avoid PASS/FAIL
    wording.  A direct accepted link and a transitive entity crossing are both
    checked because either one can merge opposite anchors.
    """

    template = _read_json(template_path)
    review = _read_json(review_path)
    result_root = Path(result_dir)
    review_status = str(review.get("review_status", ""))
    if review_status not in _REVIEW_STATUSES:
        raise ValueError(
            f"review_status must be one of {sorted(_REVIEW_STATUSES)}, got {review_status!r}"
        )

    raw_anchors = review.get("anchors")
    if not isinstance(raw_anchors, dict):
        raise ValueError("review anchors must be an object keyed by anchor_id")

    anchor_by_id: dict[str, dict[str, Any]] = {}
    tracklets_by_anchor: dict[str, set[str]] = {}
    owner_by_tracklet: dict[str, str] = {}
    for anchor_id, raw in sorted(raw_anchors.items()):
        if not isinstance(raw, dict):
            raise ValueError(f"{anchor_id}: anchor review must be an object")
        by_video, tracklet_ids = _flatten_anchor(anchor_id, raw)
        for tracklet_id in tracklet_ids:
            previous = owner_by_tracklet.get(tracklet_id)
            if previous is not None:
                raise ValueError(
                    f"tracklet {tracklet_id!r} is assigned to both {previous} and {anchor_id}"
                )
            owner_by_tracklet[tracklet_id] = anchor_id
        completeness = str(raw.get("mapping_completeness", "partial"))
        if completeness not in {"partial", "complete"}:
            raise ValueError(f"{anchor_id}: mapping_completeness must be partial or complete")
        anchor_by_id[anchor_id] = {
            "tracklet_ids_by_video": by_video,
            "mapping_completeness": completeness,
            "review_confidence": str(raw.get("review_confidence", "unspecified")),
        }
        tracklets_by_anchor[anchor_id] = tracklet_ids

    entities = _read_jsonl(result_root / "entities.jsonl")
    accepted_links = _read_jsonl(result_root / "accepted-links.jsonl")
    entity_by_tracklet: dict[str, str] = {}
    entity_rows: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity["entity_id"])
        entity_rows[entity_id] = entity
        for tracklet_id in entity.get("tracklet_ids", []):
            if tracklet_id in entity_by_tracklet:
                raise ValueError(f"tracklet {tracklet_id!r} appears in multiple result entities")
            entity_by_tracklet[str(tracklet_id)] = entity_id

    missing = sorted(set(owner_by_tracklet) - set(entity_by_tracklet))
    if missing:
        raise ValueError(f"review references tracklets absent from result entities: {missing}")

    anchor_diagnostics: dict[str, dict[str, Any]] = {}
    for anchor_id, anchor in sorted(anchor_by_id.items()):
        tracklet_ids = tracklets_by_anchor[anchor_id]
        entity_ids = sorted({entity_by_tracklet[item] for item in tracklet_ids})
        same_video_groups = {
            video_id: values
            for video_id, values in anchor["tracklet_ids_by_video"].items()
            if len(values) > 1
        }
        same_links = [
            row
            for row in accepted_links
            if str(row.get("tracklet_a")) in tracklet_ids
            and str(row.get("tracklet_b")) in tracklet_ids
        ]
        mutex_floor = max(
            (len(values) for values in anchor["tracklet_ids_by_video"].values()),
            default=0,
        )
        anchor_diagnostics[anchor_id] = {
            "mapping_completeness": anchor["mapping_completeness"],
            "review_confidence": anchor["review_confidence"],
            "annotated_tracklet_count": len(tracklet_ids),
            "videos_observed": sorted(anchor["tracklet_ids_by_video"]),
            "same_video_fragment_groups": same_video_groups,
            "same_video_extra_fragment_count": sum(len(values) - 1 for values in same_video_groups.values()),
            "output_entity_ids": entity_ids,
            "output_entity_count": len(entity_ids),
            "minimum_entity_count_under_one_track_per_video": mutex_floor,
            "excess_output_entity_count_after_mutex_floor": max(0, len(entity_ids) - mutex_floor),
            "same_anchor_accepted_links": sorted(
                same_links,
                key=lambda row: (str(row.get("tracklet_a")), str(row.get("tracklet_b"))),
            ),
        }

    pair_rows = template.get("hard_negative_pairs")
    if not isinstance(pair_rows, list) or not pair_rows:
        raise ValueError("template has no hard_negative_pairs")
    group_diagnostics: list[dict[str, Any]] = []
    for pair in pair_rows:
        group_id = str(pair["group_id"])
        anchor_ids = [str(item) for item in pair["anchor_ids"]]
        if len(anchor_ids) != 2:
            raise ValueError(f"hard-negative group {group_id} must contain exactly two anchors")
        if any(anchor_id not in tracklets_by_anchor for anchor_id in anchor_ids):
            raise ValueError(f"hard-negative group {group_id} is missing a reviewed anchor")
        left, right = (tracklets_by_anchor[anchor_ids[0]], tracklets_by_anchor[anchor_ids[1]])
        if left & right:
            raise ValueError(f"hard-negative group {group_id} assigns a tracklet to both anchors")

        direct_crossings = [
            row
            for row in accepted_links
            if (
                str(row.get("tracklet_a")) in left
                and str(row.get("tracklet_b")) in right
            )
            or (
                str(row.get("tracklet_a")) in right
                and str(row.get("tracklet_b")) in left
            )
        ]
        shared_entities = sorted(
            {entity_by_tracklet[item] for item in left}
            & {entity_by_tracklet[item] for item in right}
        )
        entity_crossings = []
        for entity_id in shared_entities:
            members = {str(item) for item in entity_rows[entity_id].get("tracklet_ids", [])}
            entity_crossings.append(
                {
                    "entity_id": entity_id,
                    "left_anchor_tracklets": sorted(members & left),
                    "right_anchor_tracklets": sorted(members & right),
                    "all_entity_tracklets": sorted(members),
                }
            )

        complete = all(anchor_by_id[item]["mapping_completeness"] == "complete" for item in anchor_ids)
        crossing = bool(direct_crossings or entity_crossings)
        if review_status == VISUAL_DRAFT:
            verdict = "VISUAL_DRAFT_CROSSING_OBSERVED" if crossing else "VISUAL_DRAFT_NO_CROSSING_OBSERVED"
        else:
            verdict = "CONFIRMED_CROSSING_OBSERVED" if crossing else "CONFIRMED_NO_CROSSING_OBSERVED"
        if not complete:
            verdict += "_PARTIAL_MAPPING"
        group_diagnostics.append(
            {
                "group_id": group_id,
                "category_id": str(pair["category_id"]),
                "anchor_ids": anchor_ids,
                "mapping_completeness": "complete" if complete else "partial",
                "opposite_merge_observed": crossing,
                "direct_accepted_crossings": sorted(
                    direct_crossings,
                    key=lambda row: (str(row.get("tracklet_a")), str(row.get("tracklet_b"))),
                ),
                "entity_crossings": entity_crossings,
                "verdict": verdict,
            }
        )

    crossed_groups = [row["group_id"] for row in group_diagnostics if row["opposite_merge_observed"]]
    pair_mapping_complete = all(
        row["mapping_completeness"] == "complete" for row in group_diagnostics
    )
    hard_negative_confirmed = review_status == DATA_OWNER_CONFIRMED and pair_mapping_complete
    return {
        "schema_version": "1.0",
        "slice_id": "S3-hard-negative-audit",
        "dataset_version": review.get("dataset_version", template.get("dataset_version")),
        "review_status": review_status,
        "diagnostic_only": not hard_negative_confirmed,
        "hard_negative_evaluated": hard_negative_confirmed,
        "g2_evaluated": False,
        "g2_blocker": "full data-owner-confirmed 17-anchor mapping is outside this four-pair audit",
        "data_owner_confirmation_required": review_status != DATA_OWNER_CONFIRMED,
        "mapping_completion_required": not pair_mapping_complete,
        "opposite_merge_group_count": len(crossed_groups),
        "opposite_merge_groups": crossed_groups,
        "fragmentation_note": (
            "same-video fragment groups are upstream tracklet segmentation evidence; "
            "the S3 one-track-per-video constraint intentionally cannot merge them"
        ),
        "anchors": anchor_diagnostics,
        "groups": group_diagnostics,
    }


def write_audit(summary: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

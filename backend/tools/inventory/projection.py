"""Project noisy ReID entities onto the data-owner-confirmed hero inventory.

The projection deliberately never consults an entity's model label.  The
data-owner-confirmed anchor tracklets are the inventory truth and become stable
``hero_<canonical_id>`` entities.  Raw ReID clusters are evaluated only as a
non-blocking model-link audit through coverage and cross-anchor pollution.

The inventory itself is the 20-row curated ``items.json``/anchor-review join;
all 20 rows remain downstream eligible.  Raw-link gaps stay explicit even when
the four-question clarification budget is exhausted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from backend.schemas.core import IdentityState, ObjectEntity

SCHEMA_VERSION = "1.0"
POLICY_VERSION = "hero-inventory-tracklet-projection-v1"
EXPECTED_INVENTORY_COUNT = 20
DEFAULT_MAX_CLARIFICATIONS = 4
MAX_EVIDENCE_REFS = 6

# These quantities are keyed by the curated canonical identifier, never parsed
# from a model label.  The three entries are the explicit grouped anchors in
# the confirmed hero inventory; every other anchor denotes one physical item.
GROUP_QUANTITIES: dict[str, int] = {
    "pen": 6,  # four pencils + two ballpoint pens
    "tea_bag": 2,
    "book": 3,
}

UNRESOLVED_STATUSES = frozenset(
    {"PARTIAL", "AMBIGUOUS", "CONTAMINATED", "MISSING"}
)


@dataclass(frozen=True)
class InventoryProjection:
    """In-memory deterministic projection and its audit metadata."""

    inventory: tuple[dict[str, Any], ...]
    trusted_entities: tuple[dict[str, Any], ...]
    display: tuple[dict[str, Any], ...]
    clarifications: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    projection_hash: str
    input_hashes: dict[str, str]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256(path.read_bytes())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _unique_index(
    rows: Iterable[dict[str, Any]], key: str, source: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{source} row is missing non-empty {key}")
        if value in result:
            raise ValueError(f"duplicate {key}={value!r} in {source}")
        result[value] = row
    return result


def _validate_and_join(
    items_document: dict[str, Any],
    anchor_review: dict[str, Any],
    *,
    expected_count: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, str]]:
    items = items_document.get("items")
    anchors = anchor_review.get("entities")
    if not isinstance(items, list) or not isinstance(anchors, list):
        raise ValueError("items.items and anchor_review.entities must be arrays")
    if len(items) != expected_count or len(anchors) != expected_count:
        raise ValueError(
            f"strict inventory requires {expected_count} items and anchors; "
            f"got items={len(items)}, anchors={len(anchors)}"
        )
    if anchor_review.get("status") != "data_owner_confirmed":
        raise ValueError("anchor review must have status=data_owner_confirmed")

    item_by_canonical = _unique_index(items, "canonical_id", "items")
    anchor_by_category = _unique_index(anchors, "category_id", "anchor review")
    if set(item_by_canonical) != set(anchor_by_category):
        missing_anchor = sorted(set(item_by_canonical) - set(anchor_by_category))
        missing_item = sorted(set(anchor_by_category) - set(item_by_canonical))
        raise ValueError(
            "canonical/category join is not one-to-one: "
            f"missing_anchor={missing_anchor}, missing_item={missing_item}"
        )

    tracklet_owner: dict[str, str] = {}
    for anchor in anchors:
        anchor_id = anchor.get("anchor_id")
        if not isinstance(anchor_id, str) or not anchor_id:
            raise ValueError("anchor review row is missing non-empty anchor_id")
        confirmed = anchor.get("confirmed_tracklet_ids_by_video")
        if not isinstance(confirmed, dict):
            raise ValueError(f"{anchor_id} has no confirmed tracklet mapping")
        for video_id, tracklet_ids in confirmed.items():
            if not isinstance(video_id, str) or not isinstance(tracklet_ids, list):
                raise ValueError(f"{anchor_id} has malformed confirmed tracklets")
            for tracklet_id in tracklet_ids:
                if not isinstance(tracklet_id, str) or not tracklet_id:
                    raise ValueError(f"{anchor_id} contains an invalid tracklet id")
                previous = tracklet_owner.get(tracklet_id)
                if previous is not None and previous != anchor_id:
                    raise ValueError(
                        f"confirmed tracklet {tracklet_id} belongs to both "
                        f"{previous} and {anchor_id}"
                    )
                tracklet_owner[tracklet_id] = anchor_id

    joined = [
        (item, anchor_by_category[item["canonical_id"]]) for item in items
    ]
    return joined, tracklet_owner


def _candidate_for_anchor(
    entity: dict[str, Any],
    *,
    anchor_id: str,
    confirmed_tracklets: set[str],
    tracklet_video: dict[str, str],
    expected_videos: set[str],
    tracklet_owner: dict[str, str],
) -> dict[str, Any] | None:
    entity_tracklets = set(entity.get("tracklet_ids") or [])
    matched = sorted(entity_tracklets & confirmed_tracklets)
    if not matched:
        return None
    foreign = sorted(
        tracklet_id
        for tracklet_id in entity_tracklets
        if tracklet_id in tracklet_owner
        and tracklet_owner[tracklet_id] != anchor_id
    )
    covered_videos = sorted({tracklet_video[tracklet_id] for tracklet_id in matched})
    confirmed_total = len(confirmed_tracklets)
    expected_video_total = len(expected_videos)
    tracklet_coverage = len(matched) / confirmed_total if confirmed_total else 0.0
    video_coverage = (
        len(set(covered_videos) & expected_videos) / expected_video_total
        if expected_video_total
        else 0.0
    )
    pollution_denominator = len(matched) + len(foreign)
    pollution_ratio = (
        len(foreign) / pollution_denominator if pollution_denominator else 0.0
    )
    confidence = max(
        0.0,
        min(
            1.0,
            0.8 * video_coverage
            + 0.2 * tracklet_coverage
            - 0.8 * pollution_ratio,
        ),
    )
    evidence_refs = sorted(
        {
            ref
            for ref in entity.get("evidence_refs") or []
            if any(
                Path(ref).stem == tracklet_id
                or Path(ref).stem.startswith(f"{tracklet_id}_")
                for tracklet_id in matched
            )
        }
    )[:MAX_EVIDENCE_REFS]
    return {
        "entity_id": entity["entity_id"],
        "identity_state": entity.get("identity_state"),
        "model_confidence": round(float(entity.get("confidence") or 0.0), 6),
        "entity_tracklet_count": len(entity_tracklets),
        "entity_tracklet_ids": sorted(entity_tracklets),
        "matched_tracklet_ids": matched,
        "foreign_confirmed_tracklet_ids": foreign,
        "covered_videos": covered_videos,
        "tracklet_coverage": round(tracklet_coverage, 6),
        "video_coverage": round(video_coverage, 6),
        "pollution_ratio": round(pollution_ratio, 6),
        "unknown_tracklet_count": len(
            [tracklet_id for tracklet_id in entity_tracklets if tracklet_id not in tracklet_owner]
        ),
        "confidence": round(confidence, 6),
        "evidence_refs": evidence_refs,
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Pure, broad-confirmation candidates win; entity id breaks exact ties."""

    return (
        bool(candidate["foreign_confirmed_tracklet_ids"]),
        -candidate["video_coverage"],
        -len(candidate["matched_tracklet_ids"]),
        -candidate["tracklet_coverage"],
        -candidate["model_confidence"],
        candidate["entity_id"],
    )


def _status_and_reason(
    candidates: list[dict[str, Any]], expected_videos: set[str]
) -> tuple[str, dict[str, Any] | None, list[str], str]:
    if not candidates:
        return (
            "MISSING",
            None,
            ["NO_CONFIRMED_TRACK_IN_ENTITIES"],
            "没有 ReID 实体包含该锚点的任何确认轨；实体链接保持缺失。",
        )

    pure = [
        candidate
        for candidate in candidates
        if not candidate["foreign_confirmed_tracklet_ids"]
    ]
    pure_complete = [
        candidate
        for candidate in pure
        if set(candidate["covered_videos"]) >= expected_videos
    ]
    if pure_complete:
        selected = sorted(pure_complete, key=_candidate_sort_key)[0]
        code = (
            "UNIQUE_PURE_ENTITY_COVERS_ALL_CONFIRMED_VIDEOS"
            if len(pure_complete) == 1
            else "BEST_PURE_ENTITY_COVERS_ALL_CONFIRMED_VIDEOS"
        )
        return (
            "CONFIRMED",
            selected,
            [code],
            "至少一个纯净 raw 实体覆盖全部确认视频；按覆盖、确认轨数和稳定主键选择最佳候选。",
        )
    if len(pure) > 1:
        selected = sorted(pure, key=_candidate_sort_key)[0]
        return (
            "AMBIGUOUS",
            selected,
            ["CONFIRMED_TRACKS_SPLIT_ACROSS_ENTITIES"],
            "确认轨分散在多个纯净实体中，尚不能唯一确定代表实体。",
        )
    if len(pure) == 1:
        return (
            "PARTIAL",
            pure[0],
            ["INCOMPLETE_CONFIRMED_VIDEO_COVERAGE"],
            "唯一纯净候选未覆盖全部确认视频；实体链接保持部分状态。",
        )
    selected = sorted(candidates, key=_candidate_sort_key)[0]
    return (
        "CONTAMINATED",
        selected,
        ["ALL_CANDIDATES_CONTAIN_FOREIGN_CONFIRMED_TRACKS"],
        "所有候选都混入其他锚点确认轨；已保留最佳候选供审计，但未提升为真值。",
    )


def _clarification_priority(record: dict[str, Any]) -> float:
    severity = {
        "CONTAMINATED": 4.0,
        "AMBIGUOUS": 3.0,
        "MISSING": 2.0,
        "PARTIAL": 1.0,
    }[record["raw_link"]["status"]]
    # Group anchors carry more physical units, so resolving them removes more
    # downstream packing uncertainty.  The confidence term is entirely based
    # on confirmed-track coverage and pollution.
    return round(
        severity * 100.0
        + (1.0 - record["raw_link"]["confidence"]) * 10.0
        + min(record["quantity"], 10) / 10.0,
        6,
    )


def build_inventory_projection(
    entities: list[dict[str, Any]],
    items_document: dict[str, Any],
    anchor_review: dict[str, Any],
    *,
    input_hashes: dict[str, str] | None = None,
    source_paths: dict[str, str] | None = None,
    expected_count: int = EXPECTED_INVENTORY_COUNT,
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS,
) -> InventoryProjection:
    """Build the strict inventory without using entity labels for alignment."""

    if expected_count != EXPECTED_INVENTORY_COUNT:
        raise ValueError(
            f"hero inventory count is fixed at {EXPECTED_INVENTORY_COUNT}"
        )
    if not 0 <= max_clarifications <= DEFAULT_MAX_CLARIFICATIONS:
        raise ValueError(
            f"max_clarifications must be between 0 and {DEFAULT_MAX_CLARIFICATIONS}"
        )
    joined, tracklet_owner = _validate_and_join(
        items_document, anchor_review, expected_count=expected_count
    )
    entity_by_id = _unique_index(entities, "entity_id", "entities")

    records: list[dict[str, Any]] = []
    trusted_entities: list[dict[str, Any]] = []
    display_rows: list[dict[str, Any]] = []
    for item, anchor in joined:
        anchor_id = anchor["anchor_id"]
        confirmed_by_video = anchor["confirmed_tracklet_ids_by_video"]
        confirmed_tracklets = sorted(
            {
                tracklet_id
                for tracklet_ids in confirmed_by_video.values()
                for tracklet_id in tracklet_ids
            }
        )
        confirmed_tracklet_set = set(confirmed_tracklets)
        tracklet_video = {
            tracklet_id: video_id
            for video_id, tracklet_ids in confirmed_by_video.items()
            for tracklet_id in tracklet_ids
        }
        expected_videos = set(anchor.get("visible_in") or []) | set(
            confirmed_by_video
        )
        candidates = [
            candidate
            for entity in entity_by_id.values()
            if (
                candidate := _candidate_for_anchor(
                    entity,
                    anchor_id=anchor_id,
                    confirmed_tracklets=confirmed_tracklet_set,
                    tracklet_video=tracklet_video,
                    expected_videos=expected_videos,
                    tracklet_owner=tracklet_owner,
                )
            )
            is not None
        ]
        candidates.sort(key=_candidate_sort_key)
        raw_status, selected, raw_reason_codes, raw_audit_reason = _status_and_reason(
            candidates, expected_videos
        )
        quantity = GROUP_QUANTITIES.get(item["canonical_id"], 1)
        projected_entity_id = f"hero_{item['canonical_id']}"
        evidence_refs = sorted(
            {
                ref
                for candidate in candidates
                for ref in candidate["evidence_refs"]
            }
        )[:MAX_EVIDENCE_REFS]
        trusted_entity = ObjectEntity(
            entity_id=projected_entity_id,
            tracklet_ids=confirmed_tracklets,
            label=item["canonical_id"],
            identity_state=IdentityState.MATCHED,
            confidence=1.0,
            evidence_refs=evidence_refs,
        ).model_dump(mode="json")
        trusted_entities.append(trusted_entity)
        hero_crop_ref = evidence_refs[0] if evidence_refs else ""
        source_tracklet_id = next(
            (
                tracklet_id
                for tracklet_id in confirmed_tracklets
                if hero_crop_ref
                and (
                    Path(hero_crop_ref).stem == tracklet_id
                    or Path(hero_crop_ref).stem.startswith(f"{tracklet_id}_")
                )
            ),
            confirmed_tracklets[0] if confirmed_tracklets else "",
        )
        display_rows.append(
            {
                "entity_id": projected_entity_id,
                "display_name_zh": item.get("name_zh", ""),
                "display_source": "data_owner_confirmed_items",
                "description_zh": item.get("description_zh", ""),
                "quantity": quantity,
                "color_primary": "",
                "hero_crop_ref": hero_crop_ref,
                "source_tracklet_id": source_tracklet_id,
                "internal_recall_label": item["canonical_id"],
            }
        )
        raw_link = {
            "status": raw_status,
            "candidate_entity_id": selected["entity_id"] if selected else None,
            "candidate_entity_ids": [candidate["entity_id"] for candidate in candidates],
            "candidate_count": len(candidates),
            "candidate_tracklet_ids": (
                selected["entity_tracklet_ids"] if selected else []
            ),
            "matched_confirmed_tracklet_ids": (
                selected["matched_tracklet_ids"] if selected else []
            ),
            "confirmed_tracklet_coverage": (
                selected["tracklet_coverage"] if selected else 0.0
            ),
            "expected_videos": sorted(expected_videos),
            "covered_videos": selected["covered_videos"] if selected else [],
            "confirmed_video_coverage": (
                selected["video_coverage"] if selected else 0.0
            ),
            "foreign_confirmed_tracklet_ids": (
                selected["foreign_confirmed_tracklet_ids"] if selected else []
            ),
            "pollution_ratio": selected["pollution_ratio"] if selected else 0.0,
            "unknown_tracklet_count": (
                selected["unknown_tracklet_count"] if selected else 0
            ),
            "confidence": selected["confidence"] if selected else 0.0,
            "evidence_refs": selected["evidence_refs"] if selected else [],
            "audit_reason": raw_audit_reason,
            "audit_reason_codes": raw_reason_codes,
        }
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "inventory_id": f"inventory-{anchor_id}",
                "canonical_id": item["canonical_id"],
                "anchor_id": anchor_id,
                "entity_id": projected_entity_id,
                "tracklet_ids": confirmed_tracklets,
                "canonical": {
                    "canonical_id": item["canonical_id"],
                    "name_zh": item.get("name_zh", ""),
                    "description_zh": item.get("description_zh", ""),
                },
                "anchor": {
                    "anchor_id": anchor_id,
                    "category_id": anchor["category_id"],
                    "display_label_zh": anchor.get("display_label_zh", ""),
                },
                "entity": trusted_entity,
                "status": "TRUSTED",
                "downstream_eligible": True,
                "quantity": quantity,
                "quantity_source": (
                    "curated_group_quantity_policy_v1"
                    if item["canonical_id"] in GROUP_QUANTITIES
                    else "single_confirmed_anchor_v1"
                ),
                "confidence": 1.0,
                "confidence_source": "data_owner_confirmed_anchor_tracklets",
                "evidence": {
                    "anchor_review_status": anchor_review["status"],
                    "anchor_review_ref": (
                        f"{(source_paths or {}).get('anchor_review', '')}#{anchor_id}"
                    ),
                    "confirmed_tracklet_ids_by_video": {
                        video_id: sorted(tracklet_ids)
                        for video_id, tracklet_ids in sorted(
                            confirmed_by_video.items()
                        )
                    },
                    "confirmed_tracklet_total": len(confirmed_tracklets),
                    "evidence_refs": evidence_refs,
                },
                "raw_link": raw_link,
                "audit_reason": (
                    "库存身份及全部成员轨来自 data_owner_confirmed 锚点；"
                    f"raw ReID 链接状态 {raw_status} 仅用于模型审计，不阻断下游。"
                ),
                "audit_reason_codes": [
                    "DATA_OWNER_CONFIRMED_ANCHOR",
                    "RAW_ENTITY_LINK_NON_BLOCKING",
                ],
                "clarification": {
                    "state": "NOT_REQUIRED"
                    if raw_status == "CONFIRMED"
                    else "PENDING_RANK",
                    "rank": None,
                    "blocks_downstream": False,
                },
            }
        )

    unresolved = [
        record
        for record in records
        if record["raw_link"]["status"] in UNRESOLVED_STATUSES
    ]
    ranked = sorted(
        unresolved,
        key=lambda record: (
            -_clarification_priority(record),
            record["anchor_id"],
        ),
    )
    clarifications: list[dict[str, Any]] = []
    for rank, record in enumerate(ranked, start=1):
        selected = rank <= max_clarifications
        record["clarification"] = {
            "state": "SELECTED" if selected else "DEFERRED_BY_CAP",
            "rank": rank,
            "blocks_downstream": False,
        }
        if not selected:
            record["raw_link"]["audit_reason_codes"].append(
                "CLARIFICATION_DEFERRED_BY_CAP"
            )
            record["raw_link"]["audit_reason"] += (
                f" 澄清优先级第 {rank}，超过硬上限 {max_clarifications}；"
                "raw 模型链接保持未决，但可信库存继续进入下游。"
            )
            continue
        candidates = record["raw_link"]["candidate_entity_ids"]
        clarification = {
            "schema_version": SCHEMA_VERSION,
            "clarification_id": f"inventory-clarification-{rank:02d}",
            "rank": rank,
            "priority_score": _clarification_priority(record),
            "canonical_id": record["canonical_id"],
            "anchor_id": record["anchor_id"],
            "projected_entity_id": record["entity_id"],
            "status": record["raw_link"]["status"],
            "blocks_downstream": False,
            "candidate_entity_ids": candidates[:4],
            "candidate_count": len(candidates),
            "question_zh": (
                f"可信库存已收录「{record['canonical']['name_zh']}」；"
                "请确认哪些 raw ReID 实体应归并到该投影实体，以改进模型链接。"
            ),
            "reason": record["raw_link"]["audit_reason"],
            "evidence": {
                "expected_videos": record["raw_link"]["expected_videos"],
                "covered_videos": record["raw_link"]["covered_videos"],
                "confirmed_tracklet_coverage": record["raw_link"][
                    "confirmed_tracklet_coverage"
                ],
                "pollution_ratio": record["raw_link"]["pollution_ratio"],
                "evidence_refs": record["raw_link"]["evidence_refs"],
            },
        }
        clarifications.append(clarification)

    raw_status_counts = {
        status: sum(record["raw_link"]["status"] == status for record in records)
        for status in (
            "CONFIRMED",
            "PARTIAL",
            "AMBIGUOUS",
            "CONTAMINATED",
            "MISSING",
        )
    }
    deferred = [
        record["anchor_id"]
        for record in records
        if record["clarification"]["state"] == "DEFERRED_BY_CAP"
    ]
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "raw_entity_count": len(entities),
        "source_entity_count": len(entities),
        "trusted_inventory_count": len(records),
        "inventory_count": len(records),
        "downstream_eligible_count": sum(
            record["downstream_eligible"] for record in records
        ),
        "trusted_entity_count": len(trusted_entities),
        "display_count": len(display_rows),
        "status_counts": {"TRUSTED": len(records)},
        "truth_link_count": len(records),
        "raw_link_status_counts": raw_status_counts,
        "raw_link_complete_count": raw_status_counts["CONFIRMED"],
        "raw_link_unresolved_count": len(unresolved),
        "unresolved_count": len(unresolved),
        "clarification_count": len(clarifications),
        "clarification_cap": max_clarifications,
        "deferred_unresolved_count": len(deferred),
        "deferred_anchor_ids": sorted(deferred),
        "group_anchor_count": sum(
            record["quantity"] > 1 for record in records
        ),
        "physical_quantity_total": sum(record["quantity"] for record in records),
        "mean_raw_link_confidence": round(
            sum(record["raw_link"]["confidence"] for record in records)
            / len(records),
            6,
        ),
        "mean_trusted_confidence": 1.0,
        "label_matching_used": False,
        "gates": {
            "strict_inventory_count": len(records) == expected_count,
            "all_confirmed_anchors_downstream_eligible": all(
                record["status"] == "TRUSTED"
                and record["downstream_eligible"]
                for record in records
            ),
            "object_entity_compatible_count": len(trusted_entities)
            == expected_count,
            "clarification_cap_respected": len(clarifications)
            <= max_clarifications,
            "all_raw_unresolved_explicit": all(
                record["clarification"]["state"]
                in {"SELECTED", "DEFERRED_BY_CAP"}
                for record in unresolved
            ),
        },
    }

    stable_input_hashes = dict(sorted((input_hashes or {}).items()))
    projection_hash = _sha256(
        _canonical_bytes(
            {
                "policy_version": POLICY_VERSION,
                "input_hashes": stable_input_hashes,
                "inventory": records,
                "trusted_entities": trusted_entities,
                "display": display_rows,
                "clarifications": clarifications,
                "metrics": metrics,
            }
        )
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "hero_inventory_projection",
        "policy_version": POLICY_VERSION,
        "projection_hash": projection_hash,
        "sources": {
            name: {
                "path": (source_paths or {}).get(name, ""),
                "sha256": digest,
            }
            for name, digest in stable_input_hashes.items()
        },
        "selection_policy": {
            "trusted_identity_authority": "data_owner_confirmed_anchor_tracklets",
            "projected_entity_id": "hero_<canonical_id>",
            "raw_identity_evidence": "confirmed_tracklet_overlap_only",
            "entity_label_used": False,
            "raw_complete_link_rule": (
                "a pollution-free raw entity covers every confirmed/visible video"
            ),
            "pollution_rule": (
                "tracklet confirmed for another anchor blocks raw-link completion"
            ),
            "raw_link_blocks_downstream": False,
        },
        "quantity_policy": {
            "default": 1,
            "canonical_overrides": dict(sorted(GROUP_QUANTITIES.items())),
        },
        "counts": {
            "raw_entities": len(entities),
            "trusted_inventory": len(records),
            "downstream_eligible": len(records),
            "raw_links_complete": raw_status_counts["CONFIRMED"],
            "raw_links_unresolved": len(unresolved),
            "clarifications": len(clarifications),
            "deferred": len(deferred),
        },
        "outputs": {
            "inventory": "inventory.jsonl",
            "trusted_entities": "trusted_entities.jsonl",
            "display": "display.jsonl",
            "clarifications": "clarifications.jsonl",
            "metrics": "metrics.json",
            "hashes": "hashes.json",
        },
    }
    return InventoryProjection(
        inventory=tuple(records),
        trusted_entities=tuple(trusted_entities),
        display=tuple(display_rows),
        clarifications=tuple(clarifications),
        metrics=metrics,
        manifest=manifest,
        projection_hash=projection_hash,
        input_hashes=stable_input_hashes,
    )


def project_inventory_files(
    *,
    entities_path: Path,
    items_path: Path,
    anchor_review_path: Path,
    max_clarifications: int = DEFAULT_MAX_CLARIFICATIONS,
) -> InventoryProjection:
    """Load the three source artifacts and return their audited projection."""

    paths = {
        "entities": entities_path,
        "items": items_path,
        "anchor_review": anchor_review_path,
    }
    input_hashes = {name: _file_hash(path) for name, path in paths.items()}
    return build_inventory_projection(
        _load_jsonl(entities_path),
        json.loads(items_path.read_text(encoding="utf-8")),
        json.loads(anchor_review_path.read_text(encoding="utf-8")),
        input_hashes=input_hashes,
        source_paths={name: path.as_posix() for name, path in paths.items()},
        max_clarifications=max_clarifications,
    )


def write_inventory_projection(
    projection: InventoryProjection, out_dir: Path
) -> dict[str, Any]:
    """Write JSONL/manifest/metrics and a detached SHA-256 ledger."""

    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "inventory.jsonl": b"".join(
            _canonical_bytes(record) for record in projection.inventory
        ),
        "trusted_entities.jsonl": b"".join(
            _canonical_bytes(record) for record in projection.trusted_entities
        ),
        "display.jsonl": b"".join(
            _canonical_bytes(record) for record in projection.display
        ),
        "clarifications.jsonl": b"".join(
            _canonical_bytes(record) for record in projection.clarifications
        ),
        "metrics.json": _canonical_bytes(projection.metrics),
        "manifest.json": _canonical_bytes(projection.manifest),
    }
    for name, payload in payloads.items():
        (out_dir / name).write_bytes(payload)
    hashes = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "sha256",
        "projection_hash": projection.projection_hash,
        "inputs": projection.input_hashes,
        "outputs": {
            name: _sha256(payload) for name, payload in sorted(payloads.items())
        },
    }
    (out_dir / "hashes.json").write_bytes(_canonical_bytes(hashes))
    return hashes

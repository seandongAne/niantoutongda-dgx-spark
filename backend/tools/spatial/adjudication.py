"""Auditable visual adjudication overlay for automatic spatial candidates.

This module is deliberately separate from the automatic producer.  A visual
review may resolve a model candidate, relabel it, or create a region from a
reviewed frame, but the resulting provenance remains
``VISUALLY_ADJUDICATED``.  It is never promoted to ``AUTO_ACCEPTED``.

The overlay is fail-closed: it is bound to one automatic spatial normalized
hash, every referenced candidate and track must exist, and every evidence
frame is checked on disk against its declared SHA256 before a downstream
``RegionManifest`` can be emitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.schemas.core import CapacityClass, SupportType
from backend.schemas.hero_bundle import RegionEntry, RegionManifest
from backend.tools.spatial.producer import (
    CoverageStatus,
    GateStatus,
    PowerState,
    SpatialCandidate,
    SpatialCandidateManifest,
)

ADJUDICATION_SCHEMA_VERSION = "1.0"
ADJUDICATION_MANIFEST_FILENAME = "adjudication_manifest.json"
ADJUDICATION_METRICS_FILENAME = "metrics.json"
ADJUDICATION_NORMALIZED_HASH_FILENAME = "normalized.sha256"
ADJUDICATED_REGION_MANIFEST_FILENAME = "regions.json"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ACCEPTED_OPERATIONS = frozenset(
    {"KEEP_LABEL", "RELABEL", "CREATE_FROM_FRAME"}
)


class ReviewerKind(str, Enum):
    AGENT_VISUAL = "AGENT_VISUAL"


class VisualOperation(str, Enum):
    KEEP_LABEL = "KEEP_LABEL"
    RELABEL = "RELABEL"
    CREATE_FROM_FRAME = "CREATE_FROM_FRAME"
    REJECT = "REJECT"
    DEFER = "DEFER"


class VisualDecisionStatus(str, Enum):
    VISUALLY_ADJUDICATED = "VISUALLY_ADJUDICATED"
    REJECTED = "REJECTED"
    NEEDS_USER = "NEEDS_USER"


class _AdjudicationContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = ADJUDICATION_SCHEMA_VERSION


def _canonical_anchor(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")


def _clean_nonempty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA256")
    return normalized


class VisualFrameEvidence(_AdjudicationContract):
    video_id: str
    timestamp_ms: int = Field(ge=0)
    frame_ref: str
    frame_sha256: str
    bbox: tuple[float, float, float, float] | None = None
    polygon: tuple[tuple[float, float], ...] | None = None

    @field_validator("video_id", "frame_ref")
    @classmethod
    def _strip_required_strings(cls, value: str, info: Any) -> str:
        return _clean_nonempty(value, field_name=info.field_name)

    @field_validator("frame_sha256")
    @classmethod
    def _valid_frame_sha256(cls, value: str) -> str:
        return _validate_sha256(value, field_name="frame_sha256")

    @field_validator("bbox")
    @classmethod
    def _valid_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError("bbox coordinates must be finite")
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2>x1 and y2>y1")
        return value

    @field_validator("polygon")
    @classmethod
    def _valid_polygon(
        cls, value: tuple[tuple[float, float], ...] | None
    ) -> tuple[tuple[float, float], ...] | None:
        if value is None:
            return None
        if len(value) < 3:
            raise ValueError("polygon must have at least three points")
        if not all(math.isfinite(coordinate) for point in value for coordinate in point):
            raise ValueError("polygon coordinates must be finite")
        return value

    @model_validator(mode="after")
    def _has_geometry(self) -> "VisualFrameEvidence":
        if self.bbox is None and self.polygon is None:
            raise ValueError("visual evidence requires bbox or polygon")
        return self


class VisualAdjudicationDecision(_AdjudicationContract):
    decision_id: str
    source_candidate_region_ids: list[str] = Field(min_length=1)
    source_track_ids: list[str] = Field(default_factory=list)
    operation: VisualOperation
    status: VisualDecisionStatus
    output_region_id: str | None = None
    anchor: str | None = None
    display_name_zh: str | None = None
    support_type: SupportType | None = None
    capacity_class: CapacityClass | None = None
    power_state: PowerState = PowerState.UNKNOWN
    visual_instance_id: str | None = None
    evidence: list[VisualFrameEvidence] = Field(default_factory=list)
    power_evidence: list[VisualFrameEvidence] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    note_zh: str = ""

    @model_validator(mode="before")
    @classmethod
    def _forbid_auto_accepted_provenance(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and value.get("status") == "AUTO_ACCEPTED":
            raise ValueError(
                "AUTO_ACCEPTED is forbidden in the visual adjudication overlay"
            )
        return value

    @field_validator("decision_id")
    @classmethod
    def _clean_decision_id(cls, value: str) -> str:
        return _clean_nonempty(value, field_name="decision_id")

    @field_validator("source_candidate_region_ids", "source_track_ids")
    @classmethod
    def _clean_id_lists(cls, value: list[str], info: Any) -> list[str]:
        cleaned = [
            _clean_nonempty(item, field_name=info.field_name) for item in value
        ]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"{info.field_name} contains duplicates")
        return cleaned

    @field_validator(
        "output_region_id", "anchor", "display_name_zh", "visual_instance_id"
    )
    @classmethod
    def _clean_optional_strings(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _clean_nonempty(value, field_name=info.field_name)

    @field_validator("reason_codes")
    @classmethod
    def _clean_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = [
            _clean_nonempty(item, field_name="reason_codes") for item in value
        ]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("reason_codes contains duplicates")
        return cleaned

    @model_validator(mode="after")
    def _operation_status_and_hard_fields(self) -> "VisualAdjudicationDecision":
        accepted_operation = self.operation.value in _ACCEPTED_OPERATIONS
        if accepted_operation and self.status is not VisualDecisionStatus.VISUALLY_ADJUDICATED:
            raise ValueError(
                f"{self.operation.value} requires VISUALLY_ADJUDICATED status"
            )
        if (
            self.operation is VisualOperation.REJECT
            and self.status is not VisualDecisionStatus.REJECTED
        ):
            raise ValueError("REJECT requires REJECTED status")
        if (
            self.operation is VisualOperation.DEFER
            and self.status is not VisualDecisionStatus.NEEDS_USER
        ):
            raise ValueError("DEFER requires NEEDS_USER status")

        if self.status is VisualDecisionStatus.VISUALLY_ADJUDICATED:
            required = {
                "output_region_id": self.output_region_id,
                "anchor": self.anchor,
                "display_name_zh": self.display_name_zh,
                "support_type": self.support_type,
                "capacity_class": self.capacity_class,
                "visual_instance_id": self.visual_instance_id,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "VISUALLY_ADJUDICATED decision missing hard fields: "
                    + ",".join(missing)
                )
            assert self.output_region_id is not None
            if self.output_region_id.casefold().startswith("auto_"):
                raise ValueError(
                    "VISUALLY_ADJUDICATED output_region_id must not use the "
                    "reserved auto_ prefix"
                )
            if not self.evidence:
                raise ValueError("VISUALLY_ADJUDICATED decision requires frame evidence")
            if self.power_state is not PowerState.UNKNOWN and not self.power_evidence:
                raise ValueError(
                    f"{self.power_state.value} requires explicit power_evidence"
                )
        return self


class VisualAdjudicationReview(_AdjudicationContract):
    review_id: str
    source_video_id: str
    source_video_sha256: str
    source_spatial_normalized_hash: str
    reviewer_kind: Literal[ReviewerKind.AGENT_VISUAL] = ReviewerKind.AGENT_VISUAL
    reviewer_id: str
    authorization_ref: str
    expected_anchor_labels: list[str] = Field(min_length=5, max_length=5)
    decisions: list[VisualAdjudicationDecision] = Field(min_length=1)

    @field_validator(
        "review_id", "source_video_id", "reviewer_id", "authorization_ref"
    )
    @classmethod
    def _strip_required_strings(cls, value: str, info: Any) -> str:
        return _clean_nonempty(value, field_name=info.field_name)

    @field_validator("source_spatial_normalized_hash")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return _validate_sha256(
            value, field_name="source_spatial_normalized_hash"
        )

    @field_validator("source_video_sha256")
    @classmethod
    def _valid_source_video_hash(cls, value: str) -> str:
        return _validate_sha256(value, field_name="source_video_sha256")

    @field_validator("expected_anchor_labels")
    @classmethod
    def _five_unique_expected_anchors(cls, value: list[str]) -> list[str]:
        cleaned = [
            _clean_nonempty(item, field_name="expected_anchor_labels")
            for item in value
        ]
        canonical = [_canonical_anchor(item) for item in cleaned]
        if any(not item for item in canonical):
            raise ValueError("expected_anchor_labels contains an empty canonical label")
        if len(canonical) != len(set(canonical)):
            raise ValueError("expected_anchor_labels contains duplicates")
        return cleaned

    @model_validator(mode="after")
    def _unique_review_identifiers(self) -> "VisualAdjudicationReview":
        decision_ids = [decision.decision_id for decision in self.decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("decision_id duplicates are forbidden")

        candidate_ids = [
            candidate_id
            for decision in self.decisions
            for candidate_id in decision.source_candidate_region_ids
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("source candidate duplicates are forbidden")

        output_ids = [
            decision.output_region_id
            for decision in self.decisions
            if decision.output_region_id is not None
        ]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("output_region_id duplicates are forbidden")
        instance_ids = [
            decision.visual_instance_id
            for decision in self.decisions
            if decision.visual_instance_id is not None
        ]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("visual_instance_id duplicates are forbidden")
        return self


class SpatialAdjudicationManifest(VisualAdjudicationReview):
    gate_status: GateStatus
    gate_reasons: list[str] = Field(default_factory=list)


class SpatialAdjudicationMetrics(_AdjudicationContract):
    review_id: str
    source_video_id: str
    source_video_sha256: str
    source_spatial_normalized_hash: str
    decision_count: int = Field(ge=0)
    visually_adjudicated_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    needs_user_count: int = Field(ge=0)
    expected_anchor_count: int = Field(ge=0)
    accepted_expected_anchor_count: int = Field(ge=0)
    projected_region_count: int = Field(ge=0)
    power_state_counts: dict[str, int] = Field(default_factory=dict)
    gate_status: GateStatus
    gate_reasons: list[str] = Field(default_factory=list)
    normalized_hash: str = ""


class SpatialAdjudicationResult(_AdjudicationContract):
    adjudication_manifest: SpatialAdjudicationManifest
    metrics: SpatialAdjudicationMetrics
    normalized_hash: str
    region_manifest: RegionManifest | None = None

    @property
    def gate_passed(self) -> bool:
        return self.metrics.gate_status is GateStatus.PASS


def _frame_sort_key(evidence: VisualFrameEvidence) -> tuple[Any, ...]:
    return (
        evidence.video_id,
        evidence.timestamp_ms,
        evidence.frame_ref,
        evidence.frame_sha256,
        evidence.bbox or (),
        evidence.polygon or (),
    )


def _normalized_review(review: VisualAdjudicationReview) -> VisualAdjudicationReview:
    decisions: list[VisualAdjudicationDecision] = []
    for decision in review.decisions:
        decisions.append(
            decision.model_copy(
                update={
                    "source_candidate_region_ids": sorted(
                        decision.source_candidate_region_ids
                    ),
                    "source_track_ids": sorted(decision.source_track_ids),
                    "evidence": sorted(decision.evidence, key=_frame_sort_key),
                    "power_evidence": sorted(
                        decision.power_evidence, key=_frame_sort_key
                    ),
                    "reason_codes": sorted(decision.reason_codes),
                }
            )
        )
    return review.model_copy(
        update={
            "expected_anchor_labels": sorted(
                review.expected_anchor_labels, key=_canonical_anchor
            ),
            "decisions": sorted(decisions, key=lambda item: item.decision_id),
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frame_evidence(
    evidence: VisualFrameEvidence,
    *,
    source_video_id: str,
    project_root: Path,
) -> None:
    if evidence.video_id != source_video_id:
        raise ValueError(
            f"evidence video_id {evidence.video_id!r} does not match "
            f"{source_video_id!r}"
        )
    raw_ref = Path(evidence.frame_ref)
    if raw_ref.is_absolute():
        raise ValueError(f"frame_ref must be project-relative: {evidence.frame_ref}")
    root = project_root.resolve()
    frame_path = (root / raw_ref).resolve()
    try:
        frame_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"frame_ref escapes project root: {evidence.frame_ref}"
        ) from exc
    if not frame_path.is_file():
        raise ValueError(f"evidence frame does not exist: {evidence.frame_ref}")
    actual_hash = _sha256_file(frame_path)
    if actual_hash != evidence.frame_sha256:
        raise ValueError(
            f"evidence frame SHA256 mismatch for {evidence.frame_ref}: "
            f"declared={evidence.frame_sha256} actual={actual_hash}"
        )


def _validate_candidate_links(
    decision: VisualAdjudicationDecision,
    *,
    candidates_by_id: Mapping[str, SpatialCandidate],
) -> None:
    try:
        source_candidates = [
            candidates_by_id[candidate_id]
            for candidate_id in decision.source_candidate_region_ids
        ]
    except KeyError as exc:
        raise ValueError(f"unknown source candidate: {exc.args[0]}") from exc

    available_tracks = {
        track_id
        for candidate in source_candidates
        for track_id in candidate.source_track_ids
    }
    unknown_tracks = sorted(set(decision.source_track_ids) - available_tracks)
    if unknown_tracks:
        raise ValueError(
            f"{decision.decision_id}: unknown source_track_ids: {unknown_tracks}"
        )
    if (
        decision.status is VisualDecisionStatus.VISUALLY_ADJUDICATED
        and decision.operation in {VisualOperation.KEEP_LABEL, VisualOperation.RELABEL}
        and not decision.source_track_ids
    ):
        raise ValueError(
            f"{decision.decision_id}: {decision.operation.value} requires source_track_ids"
        )

    if decision.status is not VisualDecisionStatus.VISUALLY_ADJUDICATED:
        return
    assert decision.anchor is not None
    source_anchors = {
        _canonical_anchor(candidate.anchor)
        for candidate in source_candidates
        if candidate.anchor
        and candidate.coverage_status is CoverageStatus.OBSERVED
    }
    output_anchor = _canonical_anchor(decision.anchor)
    if decision.operation is VisualOperation.KEEP_LABEL and source_anchors != {
        output_anchor
    }:
        raise ValueError(
            f"{decision.decision_id}: KEEP_LABEL does not match source candidate labels"
        )
    if (
        decision.operation is VisualOperation.RELABEL
        and source_anchors
        and source_anchors == {output_anchor}
    ):
        raise ValueError(
            f"{decision.decision_id}: RELABEL must change a source candidate label"
        )


def _evidence_ref(evidence: VisualFrameEvidence, *, power: bool = False) -> str:
    prefix = "visual-power" if power else "visual"
    return (
        f"{prefix}:{evidence.video_id}@{evidence.timestamp_ms}ms#"
        f"{evidence.frame_ref}@sha256:{evidence.frame_sha256}"
    )


def _canonical_payload(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {key: _canonical_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_payload(item) for item in value]
    return value


def _normalized_hash(
    manifest: SpatialAdjudicationManifest,
    region_manifest: RegionManifest | None,
    metrics: SpatialAdjudicationMetrics,
) -> str:
    payload = {
        "adjudication_manifest": manifest.model_dump(mode="json"),
        "region_manifest": (
            region_manifest.model_dump(mode="json") if region_manifest else None
        ),
        "metrics": metrics.model_dump(mode="json", exclude={"normalized_hash"}),
    }
    canonical = json.dumps(
        _canonical_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adjudicate_spatial_regions(
    candidate_manifest: SpatialCandidateManifest | Mapping[str, Any],
    source_normalized_hash: str,
    review: VisualAdjudicationReview | Mapping[str, Any],
    *,
    project_root: str | Path,
) -> SpatialAdjudicationResult:
    """Validate a visual overlay and project exactly five trusted regions."""

    candidates = (
        candidate_manifest
        if isinstance(candidate_manifest, SpatialCandidateManifest)
        else SpatialCandidateManifest.model_validate(candidate_manifest)
    )
    validated_review = (
        review
        if isinstance(review, VisualAdjudicationReview)
        else VisualAdjudicationReview.model_validate(review)
    )
    validated_review = _normalized_review(validated_review)
    source_hash = _validate_sha256(
        source_normalized_hash, field_name="source_normalized_hash"
    )
    if validated_review.source_spatial_normalized_hash != source_hash:
        raise ValueError(
            "review source_spatial_normalized_hash does not match source hash"
        )
    if candidates.video_id != validated_review.source_video_id:
        raise ValueError(
            f"candidate video_id {candidates.video_id!r} does not match review "
            f"source_video_id {validated_review.source_video_id!r}"
        )

    source_expected = {
        _canonical_anchor(anchor) for anchor in candidates.config.expected_anchor_labels
    }
    review_expected = {
        _canonical_anchor(anchor) for anchor in validated_review.expected_anchor_labels
    }
    if len(source_expected) != 5 or source_expected != review_expected:
        raise ValueError(
            "review expected_anchor_labels must exactly match the five source "
            "candidate expected anchors"
        )

    candidates_by_id = {candidate.region_id: candidate for candidate in candidates.candidates}
    if len(candidates_by_id) != len(candidates.candidates):
        raise ValueError("candidate manifest contains duplicate region_id values")

    evidence_root = Path(project_root)
    for decision in validated_review.decisions:
        _validate_candidate_links(decision, candidates_by_id=candidates_by_id)
        for evidence in [*decision.evidence, *decision.power_evidence]:
            _validate_frame_evidence(
                evidence,
                source_video_id=validated_review.source_video_id,
                project_root=evidence_root,
            )

    accepted = [
        decision
        for decision in validated_review.decisions
        if decision.status is VisualDecisionStatus.VISUALLY_ADJUDICATED
    ]
    accepted_anchors = [
        _canonical_anchor(decision.anchor or "") for decision in accepted
    ]
    gate_reasons: list[str] = []
    if len(accepted) != 5:
        gate_reasons.append(f"expected_five_adjudicated_regions:{len(accepted)}/5")
    accepted_anchor_set = set(accepted_anchors)
    missing = sorted(review_expected - accepted_anchor_set)
    extra = sorted(accepted_anchor_set - review_expected)
    if missing:
        gate_reasons.append("expected_anchors_not_adjudicated:" + ",".join(missing))
    if extra:
        gate_reasons.append("unexpected_adjudicated_anchors:" + ",".join(extra))
    if len(accepted_anchors) != len(accepted_anchor_set):
        gate_reasons.append("duplicate_adjudicated_anchor")
    needs_user_count = sum(
        decision.status is VisualDecisionStatus.NEEDS_USER
        for decision in validated_review.decisions
    )
    if needs_user_count:
        gate_reasons.append(f"needs_user_decisions_present:{needs_user_count}")

    gate_status = GateStatus.NEEDS_USER if gate_reasons else GateStatus.PASS
    manifest = SpatialAdjudicationManifest.model_validate(
        {
            **validated_review.model_dump(mode="json"),
            "gate_status": gate_status.value,
            "gate_reasons": gate_reasons,
        }
    )

    region_manifest: RegionManifest | None = None
    if gate_status is GateStatus.PASS:
        entries: list[RegionEntry] = []
        for decision in accepted:
            assert decision.output_region_id is not None
            assert decision.anchor is not None
            assert decision.display_name_zh is not None
            assert decision.support_type is not None
            assert decision.capacity_class is not None
            evidence_refs = {
                _evidence_ref(evidence) for evidence in decision.evidence
            }
            evidence_refs.add(
                f"agent_review:{validated_review.review_id}/{decision.decision_id}"
            )
            evidence_refs.update(
                _evidence_ref(evidence, power=True)
                for evidence in decision.power_evidence
            )
            entries.append(
                RegionEntry(
                    region_id=decision.output_region_id,
                    anchor=decision.anchor,
                    display_name_zh=decision.display_name_zh,
                    support_type=decision.support_type,
                    capacity_class=decision.capacity_class,
                    near_power=decision.power_state is PowerState.NEAR,
                    evidence_refs=sorted(evidence_refs),
                )
            )
        unknown_power_region_ids = sorted(
            decision.output_region_id
            for decision in accepted
            if decision.power_state is PowerState.UNKNOWN
            and decision.output_region_id is not None
        )
        notes = (
            "VISUALLY_ADJUDICATED overlay; review="
            + validated_review.review_id
            + "; source spatial normalized hash="
            + source_hash
        )
        if unknown_power_region_ids:
            notes += (
                "; UNKNOWN power projected near_power=false for:"
                + ",".join(unknown_power_region_ids)
            )
        region_manifest = RegionManifest(
            video_id=validated_review.source_video_id,
            entries=sorted(entries, key=lambda entry: entry.region_id),
            notes=notes,
        )

    status_counts = Counter(decision.status.value for decision in validated_review.decisions)
    power_state_counts = Counter(
        decision.power_state.value
        for decision in accepted
    )
    metrics = SpatialAdjudicationMetrics(
        review_id=validated_review.review_id,
        source_video_id=validated_review.source_video_id,
        source_video_sha256=validated_review.source_video_sha256,
        source_spatial_normalized_hash=source_hash,
        decision_count=len(validated_review.decisions),
        visually_adjudicated_count=status_counts[
            VisualDecisionStatus.VISUALLY_ADJUDICATED.value
        ],
        rejected_count=status_counts[VisualDecisionStatus.REJECTED.value],
        needs_user_count=status_counts[VisualDecisionStatus.NEEDS_USER.value],
        expected_anchor_count=len(review_expected),
        accepted_expected_anchor_count=len(review_expected & accepted_anchor_set),
        projected_region_count=(len(accepted) if region_manifest is not None else 0),
        power_state_counts=dict(sorted(power_state_counts.items())),
        gate_status=gate_status,
        gate_reasons=gate_reasons,
    )
    digest = _normalized_hash(manifest, region_manifest, metrics)
    metrics = metrics.model_copy(update={"normalized_hash": digest})
    return SpatialAdjudicationResult(
        adjudication_manifest=manifest,
        metrics=metrics,
        normalized_hash=digest,
        region_manifest=region_manifest,
    )


def load_visual_adjudication(path: str | Path) -> VisualAdjudicationReview:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
    return VisualAdjudicationReview.model_validate(payload)


def remove_stale_adjudicated_regions(out_dir: str | Path) -> None:
    region_path = Path(out_dir) / ADJUDICATED_REGION_MANIFEST_FILENAME
    if region_path.exists():
        region_path.unlink()


def _json_bytes(model: BaseModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_spatial_adjudication_outputs(
    result: SpatialAdjudicationResult, out_dir: str | Path
) -> dict[str, str]:
    """Write diagnostics on every valid result; write regions only on PASS."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / ADJUDICATION_MANIFEST_FILENAME
    metrics_path = destination / ADJUDICATION_METRICS_FILENAME
    hash_path = destination / ADJUDICATION_NORMALIZED_HASH_FILENAME
    region_path = destination / ADJUDICATED_REGION_MANIFEST_FILENAME

    manifest_path.write_bytes(_json_bytes(result.adjudication_manifest))
    metrics_path.write_bytes(_json_bytes(result.metrics))
    hash_path.write_text(result.normalized_hash + "\n", encoding="ascii")
    if result.region_manifest is not None:
        region_path.write_bytes(_json_bytes(result.region_manifest))
    elif region_path.exists():
        region_path.unlink()

    written = {
        "adjudication_manifest": str(manifest_path),
        "metrics": str(metrics_path),
        "normalized_hash": str(hash_path),
    }
    if result.region_manifest is not None:
        written["region_manifest"] = str(region_path)
    return written

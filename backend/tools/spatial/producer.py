"""Fail-closed producer for automatically observed new-home regions.

The producer consumes per-frame anchor/region observations emitted by a model.
It never reads an existing ``RegionManifest``.  Repeated observations are
deduplicated deterministically; only candidates with stable observations and
confident hard fields can enter the compatibility ``RegionManifest`` used by
the current solver.

Power is intentionally three-state in the candidate contract.  ``NEAR`` is
projected to ``near_power=True`` only when an outlet evidence reference is
present and confident.  ``UNKNOWN`` and ``NOT_NEAR`` both project to false, so
absence of evidence never grants a power-placement capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.schemas.core import CapacityClass, SupportType
from backend.schemas.hero_bundle import RegionEntry, RegionManifest

SPATIAL_SCHEMA_VERSION = "1.0"

CANDIDATE_MANIFEST_FILENAME = "candidate_manifest.json"
REGION_MANIFEST_FILENAME = "regions.json"
METRICS_FILENAME = "metrics.json"
NORMALIZED_HASH_FILENAME = "normalized.sha256"


class PowerState(str, Enum):
    NEAR = "NEAR"
    NOT_NEAR = "NOT_NEAR"
    UNKNOWN = "UNKNOWN"


class CoverageStatus(str, Enum):
    OBSERVED = "OBSERVED"
    NOT_OBSERVED = "NOT_OBSERVED"


class CandidateStatus(str, Enum):
    AUTO_ACCEPTED = "AUTO_ACCEPTED"
    NEEDS_USER = "NEEDS_USER"


class GateStatus(str, Enum):
    PASS = "PASS"
    NEEDS_USER = "NEEDS_USER"


class _SpatialContract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_version: Literal["1.0"] = SPATIAL_SCHEMA_VERSION


class SpatialObservation(_SpatialContract):
    """One automatic region observation from a new-home video frame.

    ``bbox`` and ``polygon`` are image-space geometry in any consistent unit.
    A producer may include both, but at least one is required.  Optional
    per-field confidence values let upstream models express uncertainty without
    fabricating a hard support/capacity value.
    """

    video_id: str | None = None
    timestamp_ms: int = Field(
        ge=0,
        validation_alias=AliasChoices("timestamp_ms", "timestamp"),
    )
    frame_ref: str = Field(
        min_length=1,
        validation_alias=AliasChoices("frame_ref", "frame", "frame_id"),
    )
    bbox: tuple[float, float, float, float] | None = None
    polygon: tuple[tuple[float, float], ...] | None = None
    region_track_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "region_track_id", "track_id", "anchor_instance_id"
        ),
    )
    anchor_label: str | None = Field(
        default=None,
        validation_alias=AliasChoices("anchor_label", "anchor"),
    )
    display_name_zh: str | None = None
    support_type: SupportType | None = Field(
        default=None,
        validation_alias=AliasChoices("support_type", "support"),
    )
    capacity_class: CapacityClass | None = Field(
        default=None,
        validation_alias=AliasChoices("capacity_class", "capacity"),
    )
    power_state: PowerState = Field(
        default=PowerState.UNKNOWN,
        validation_alias=AliasChoices("power_state", "power", "near_power"),
    )
    power_evidence_refs: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("power_evidence_refs", "power_evidence"),
    )
    evidence_refs: list[str] = Field(default_factory=list)
    model_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("model_confidence", "confidence"),
    )
    anchor_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    support_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    capacity_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    power_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str = ""

    @field_validator("frame_ref", mode="before")
    @classmethod
    def _frame_to_string(cls, value: Any) -> str:
        return str(value)

    @field_validator("anchor_label", "display_name_zh", "region_track_id")
    @classmethod
    def _strip_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("power_state", mode="before")
    @classmethod
    def _parse_power_state(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return PowerState.NEAR if value else PowerState.NOT_NEAR
        if value is None or value == "":
            return PowerState.UNKNOWN
        normalized = str(value).strip().upper().replace("-", "_")
        aliases = {
            "TRUE": PowerState.NEAR,
            "NEAR_POWER": PowerState.NEAR,
            "NEAR": PowerState.NEAR,
            "FALSE": PowerState.NOT_NEAR,
            "NOT_NEAR_POWER": PowerState.NOT_NEAR,
            "NOT_NEAR": PowerState.NOT_NEAR,
            "UNKNOWN": PowerState.UNKNOWN,
            "UNOBSERVED": PowerState.UNKNOWN,
        }
        return aliases.get(normalized, value)

    @field_validator("power_evidence_refs", "evidence_refs", mode="before")
    @classmethod
    def _evidence_to_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("bbox")
    @classmethod
    def _valid_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        if not all(math.isfinite(v) for v in value):
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
        if not all(math.isfinite(coord) for point in value for coord in point):
            raise ValueError("polygon coordinates must be finite")
        return value

    @model_validator(mode="after")
    def _has_image_geometry(self) -> "SpatialObservation":
        if self.bbox is None and self.polygon is None:
            raise ValueError("one of bbox or polygon is required")
        return self


class SpatialProducerConfig(_SpatialContract):
    """Acceptance thresholds for automatic region production."""

    min_regions: int = Field(default=5, ge=1)
    min_observations_per_region: int = Field(default=2, ge=1)
    min_model_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_hard_field_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    min_power_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    min_field_consensus: float = Field(default=0.67, gt=0.5, le=1.0)
    dedupe_iou_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    expected_anchor_labels: list[str] = Field(default_factory=list)
    require_expected_coverage: bool = True

    @field_validator("expected_anchor_labels")
    @classmethod
    def _clean_expected_anchors(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        canonical = [_canonical_anchor(item) for item in cleaned]
        if len(canonical) != len(set(canonical)):
            raise ValueError("expected_anchor_labels contains duplicates")
        return sorted(cleaned, key=_canonical_anchor)


class SpatialCandidate(_SpatialContract):
    region_id: str
    anchor: str | None = None
    display_name_zh: str
    support_type: SupportType | None = None
    capacity_class: CapacityClass | None = None
    power_state: PowerState = PowerState.UNKNOWN
    coverage_status: CoverageStatus
    status: CandidateStatus
    model_confidence: float = Field(ge=0.0, le=1.0)
    observation_count: int = Field(ge=0)
    first_timestamp_ms: int | None = Field(default=None, ge=0)
    last_timestamp_ms: int | None = Field(default=None, ge=0)
    representative_bbox: tuple[float, float, float, float] | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    power_evidence_refs: list[str] = Field(default_factory=list)
    model_versions: list[str] = Field(default_factory=list)
    decision_reasons: list[str] = Field(default_factory=list)


class SpatialCandidateManifest(_SpatialContract):
    video_id: str
    source_kind: Literal["AUTO_OBSERVATION_JSONL"] = "AUTO_OBSERVATION_JSONL"
    config: SpatialProducerConfig
    status: GateStatus
    gate_reasons: list[str] = Field(default_factory=list)
    candidates: list[SpatialCandidate] = Field(default_factory=list)


class SpatialMetrics(_SpatialContract):
    video_id: str
    observation_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    observed_candidate_count: int = Field(ge=0)
    not_observed_count: int = Field(ge=0)
    auto_accepted_count: int = Field(ge=0)
    needs_user_count: int = Field(ge=0)
    projected_region_count: int = Field(ge=0)
    min_regions_required: int = Field(ge=1)
    region_gate_passed: bool
    expected_anchor_count: int = Field(ge=0)
    observed_expected_anchor_count: int = Field(ge=0)
    expected_coverage_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    power_state_counts: dict[str, int] = Field(default_factory=dict)
    gate_status: GateStatus
    gate_reasons: list[str] = Field(default_factory=list)
    normalized_hash: str = ""


class SpatialProductionResult(_SpatialContract):
    candidate_manifest: SpatialCandidateManifest
    region_manifest: RegionManifest | None = None
    metrics: SpatialMetrics
    normalized_hash: str

    @property
    def gate_passed(self) -> bool:
        return self.metrics.gate_status is GateStatus.PASS


def _canonical_anchor(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")


def _anchor_slug(value: str | None) -> str:
    canonical = _canonical_anchor(value)
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", canonical).strip("_")
    if ascii_slug:
        return ascii_slug[:40]
    if canonical:
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
        return f"anchor_{digest}"
    return "unknown"


def _geometry_bbox(
    observation: SpatialObservation,
) -> tuple[float, float, float, float]:
    if observation.bbox is not None:
        return observation.bbox
    assert observation.polygon is not None
    xs = [point[0] for point in observation.polygon]
    ys = [point[1] for point in observation.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _observation_sort_key(observation: SpatialObservation) -> tuple[Any, ...]:
    bbox = _geometry_bbox(observation)
    return (
        observation.region_track_id or "",
        _canonical_anchor(observation.anchor_label),
        round((bbox[0] + bbox[2]) / 2.0, 8),
        round((bbox[1] + bbox[3]) / 2.0, 8),
        observation.timestamp_ms,
        observation.frame_ref,
    )


def _cluster_observations(
    observations: Sequence[SpatialObservation], config: SpatialProducerConfig
) -> list[list[SpatialObservation]]:
    tracked: dict[str, list[SpatialObservation]] = defaultdict(list)
    untracked: list[SpatialObservation] = []
    for observation in observations:
        if observation.region_track_id:
            tracked[observation.region_track_id].append(observation)
        else:
            untracked.append(observation)

    clusters: list[list[SpatialObservation]] = [
        sorted(items, key=_observation_sort_key)
        for _, items in sorted(tracked.items())
    ]
    untracked_clusters: list[list[SpatialObservation]] = []
    for observation in sorted(untracked, key=_observation_sort_key):
        anchor = _canonical_anchor(observation.anchor_label)
        bbox = _geometry_bbox(observation)
        matches: list[tuple[float, int]] = []
        for index, cluster in enumerate(untracked_clusters):
            cluster_anchor = _canonical_anchor(cluster[0].anchor_label)
            if anchor != cluster_anchor:
                continue
            overlap = max(_bbox_iou(bbox, _geometry_bbox(item)) for item in cluster)
            if overlap >= config.dedupe_iou_threshold:
                matches.append((overlap, index))
        if matches:
            _, index = max(matches, key=lambda item: (item[0], -item[1]))
            untracked_clusters[index].append(observation)
        else:
            untracked_clusters.append([observation])
    clusters.extend(
        sorted(cluster, key=_observation_sort_key) for cluster in untracked_clusters
    )
    return clusters


def _field_confidence(observation: SpatialObservation, field: str) -> float:
    confidence_field = {
        "anchor_label": "anchor_confidence",
        "support_type": "support_confidence",
        "capacity_class": "capacity_confidence",
    }[field]
    explicit = getattr(observation, confidence_field)
    return observation.model_confidence if explicit is None else explicit


def _consensus(
    observations: Sequence[SpatialObservation],
    *,
    field: str,
    config: SpatialProducerConfig,
    normalize: Any = None,
) -> tuple[Any | None, bool]:
    normalize = normalize or (lambda value: value)
    confident_values: list[tuple[Any, Any]] = []
    for observation in observations:
        value = getattr(observation, field)
        if value is None:
            continue
        if _field_confidence(observation, field) < config.min_hard_field_confidence:
            continue
        confident_values.append((normalize(value), value))
    if not confident_values:
        return None, False
    counts = Counter(item[0] for item in confident_values)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None, False
    winning_key, winning_count = ordered[0]
    # Missing or low-confidence observations count against a hard-field claim.
    if winning_count / len(observations) < config.min_field_consensus:
        return None, False
    original = next(value for key, value in confident_values if key == winning_key)
    return original, True


def _representative_bbox(
    observations: Sequence[SpatialObservation],
) -> tuple[float, float, float, float]:
    boxes = [_geometry_bbox(observation) for observation in observations]
    return tuple(
        round(float(statistics.median(box[index] for box in boxes)), 6)
        for index in range(4)
    )  # type: ignore[return-value]


def _observation_evidence(video_id: str, observation: SpatialObservation) -> str:
    return f"{video_id}@{observation.timestamp_ms}ms#{observation.frame_ref}"


def _aggregate_power(
    observations: Sequence[SpatialObservation], config: SpatialProducerConfig
) -> tuple[PowerState, list[str], list[str]]:
    valid_near_refs: set[str] = set()
    valid_near = False
    valid_not_near = False
    invalid_near = False
    for observation in observations:
        confidence = (
            observation.model_confidence
            if observation.power_confidence is None
            else observation.power_confidence
        )
        if observation.power_state is PowerState.NEAR:
            refs = {ref.strip() for ref in observation.power_evidence_refs if ref.strip()}
            if confidence >= config.min_power_confidence and refs:
                valid_near = True
                valid_near_refs.update(refs)
            else:
                invalid_near = True
        elif (
            observation.power_state is PowerState.NOT_NEAR
            and confidence >= config.min_power_confidence
        ):
            valid_not_near = True

    reasons: list[str] = []
    if invalid_near:
        reasons.append("power_near_missing_or_low_confidence_outlet_evidence")
    if valid_near and valid_not_near:
        reasons.append("power_state_conflict")
        return PowerState.UNKNOWN, sorted(valid_near_refs), reasons
    if valid_near and not invalid_near:
        return PowerState.NEAR, sorted(valid_near_refs), reasons
    if valid_not_near and not invalid_near:
        return PowerState.NOT_NEAR, [], reasons
    return PowerState.UNKNOWN, [], reasons


def _aggregate_cluster(
    video_id: str,
    observations: Sequence[SpatialObservation],
    config: SpatialProducerConfig,
) -> SpatialCandidate:
    anchor, anchor_ok = _consensus(
        observations,
        field="anchor_label",
        config=config,
        normalize=_canonical_anchor,
    )
    support, support_ok = _consensus(
        observations,
        field="support_type",
        config=config,
    )
    capacity, capacity_ok = _consensus(
        observations,
        field="capacity_class",
        config=config,
    )
    confidence = round(
        math.fsum(sorted(observation.model_confidence for observation in observations))
        / len(observations),
        6,
    )
    power_state, power_refs, power_reasons = _aggregate_power(observations, config)

    reasons: list[str] = []
    if len(observations) < config.min_observations_per_region:
        reasons.append("insufficient_cross_frame_observations")
    if confidence < config.min_model_confidence:
        reasons.append("model_confidence_below_threshold")
    if not anchor_ok:
        reasons.append("anchor_uncertain")
    if not support_ok:
        reasons.append("support_type_uncertain")
    if not capacity_ok:
        reasons.append("capacity_class_uncertain")
    reasons.extend(power_reasons)

    display_candidates = sorted(
        (
            (-observation.model_confidence, observation.display_name_zh)
            for observation in observations
            if observation.display_name_zh
        ),
        key=lambda item: (item[0], item[1]),
    )
    display_name = (
        display_candidates[0][1]
        if display_candidates
        else (str(anchor) if anchor else "未确认区域")
    )
    evidence = {
        _observation_evidence(video_id, observation) for observation in observations
    }
    evidence.update(
        ref.strip()
        for observation in observations
        for ref in observation.evidence_refs
        if ref.strip()
    )
    if power_state is PowerState.NEAR:
        evidence.update(power_refs)

    timestamps = [observation.timestamp_ms for observation in observations]
    return SpatialCandidate(
        region_id="pending",
        anchor=str(anchor) if anchor is not None else None,
        display_name_zh=display_name,
        support_type=support if support_ok else None,
        capacity_class=capacity if capacity_ok else None,
        power_state=power_state,
        coverage_status=CoverageStatus.OBSERVED,
        status=(
            CandidateStatus.NEEDS_USER
            if reasons
            else CandidateStatus.AUTO_ACCEPTED
        ),
        model_confidence=confidence,
        observation_count=len(observations),
        first_timestamp_ms=min(timestamps),
        last_timestamp_ms=max(timestamps),
        representative_bbox=_representative_bbox(observations),
        evidence_refs=sorted(evidence),
        power_evidence_refs=power_refs,
        model_versions=sorted(
            {
                observation.model_version.strip()
                for observation in observations
                if observation.model_version.strip()
            }
        ),
        decision_reasons=sorted(set(reasons)),
    )


def _candidate_position(candidate: SpatialCandidate) -> tuple[Any, ...]:
    if candidate.representative_bbox is None:
        return (math.inf, math.inf, "", "")
    x1, y1, x2, y2 = candidate.representative_bbox
    return (
        round((x1 + x2) / 2.0, 6),
        round((y1 + y2) / 2.0, 6),
        candidate.support_type.value if candidate.support_type else "",
        candidate.capacity_class.value if candidate.capacity_class else "",
    )


def _assign_stable_region_ids(
    candidates: Sequence[SpatialCandidate],
) -> list[SpatialCandidate]:
    by_slug: dict[str, list[SpatialCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_slug[_anchor_slug(candidate.anchor)].append(candidate)
    assigned: list[SpatialCandidate] = []
    for slug, items in sorted(by_slug.items()):
        observed = sorted(
            (
                item
                for item in items
                if item.coverage_status is CoverageStatus.OBSERVED
            ),
            key=_candidate_position,
        )
        for ordinal, item in enumerate(observed, 1):
            assigned.append(
                item.model_copy(update={"region_id": f"auto_{slug}_{ordinal:02d}"})
            )
        for item in sorted(
            (
                item
                for item in items
                if item.coverage_status is CoverageStatus.NOT_OBSERVED
            ),
            key=lambda candidate: candidate.display_name_zh,
        ):
            assigned.append(
                item.model_copy(update={"region_id": f"auto_{slug}_not_observed"})
            )
    return sorted(assigned, key=lambda candidate: candidate.region_id)


def _missing_expected_candidates(
    candidates: Sequence[SpatialCandidate], config: SpatialProducerConfig
) -> list[SpatialCandidate]:
    observed = {
        _canonical_anchor(candidate.anchor)
        for candidate in candidates
        if candidate.coverage_status is CoverageStatus.OBSERVED and candidate.anchor
    }
    missing: list[SpatialCandidate] = []
    for anchor in config.expected_anchor_labels:
        if _canonical_anchor(anchor) in observed:
            continue
        missing.append(
            SpatialCandidate(
                region_id="pending",
                anchor=anchor,
                display_name_zh=anchor,
                coverage_status=CoverageStatus.NOT_OBSERVED,
                status=CandidateStatus.NEEDS_USER,
                model_confidence=0.0,
                observation_count=0,
                decision_reasons=["expected_anchor_not_observed"],
            )
        )
    return missing


def _canonical_payload(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {key: _canonical_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_payload(item) for item in value]
    return value


def _normalized_hash(
    candidate_manifest: SpatialCandidateManifest,
    region_manifest: RegionManifest | None,
    metrics: SpatialMetrics,
) -> str:
    payload = {
        "candidate_manifest": candidate_manifest.model_dump(mode="json"),
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


def produce_spatial_regions(
    video_id: str,
    observations: Iterable[SpatialObservation | Mapping[str, Any]],
    config: SpatialProducerConfig | None = None,
) -> SpatialProductionResult:
    """Produce gated candidates and an optional downstream RegionManifest."""

    video_id = video_id.strip()
    if not video_id:
        raise ValueError("video_id must be non-empty")
    config = config or SpatialProducerConfig()
    validated: list[SpatialObservation] = []
    for index, raw in enumerate(observations, 1):
        observation = (
            raw
            if isinstance(raw, SpatialObservation)
            else SpatialObservation.model_validate(raw)
        )
        if observation.video_id is not None and observation.video_id != video_id:
            raise ValueError(
                f"observation {index} video_id {observation.video_id!r} "
                f"does not match {video_id!r}"
            )
        validated.append(observation)

    clusters = _cluster_observations(validated, config)
    candidates = [
        _aggregate_cluster(video_id, cluster, config) for cluster in clusters
    ]
    candidates.extend(_missing_expected_candidates(candidates, config))
    candidates = _assign_stable_region_ids(candidates)

    accepted = [
        candidate
        for candidate in candidates
        if candidate.status is CandidateStatus.AUTO_ACCEPTED
        and candidate.coverage_status is CoverageStatus.OBSERVED
    ]
    not_observed = [
        candidate
        for candidate in candidates
        if candidate.coverage_status is CoverageStatus.NOT_OBSERVED
    ]
    gate_reasons: list[str] = []
    if len(accepted) < config.min_regions:
        gate_reasons.append(
            f"min_regions_not_met:{len(accepted)}/{config.min_regions}"
        )
    if config.require_expected_coverage and not_observed:
        labels = ",".join(sorted(candidate.anchor or "unknown" for candidate in not_observed))
        gate_reasons.append(f"expected_anchors_not_observed:{labels}")
    gate_status = GateStatus.NEEDS_USER if gate_reasons else GateStatus.PASS

    candidate_manifest = SpatialCandidateManifest(
        video_id=video_id,
        config=config,
        status=gate_status,
        gate_reasons=gate_reasons,
        candidates=candidates,
    )

    region_manifest: RegionManifest | None = None
    if gate_status is GateStatus.PASS:
        entries = [
            RegionEntry(
                region_id=candidate.region_id,
                anchor=candidate.anchor or "",
                display_name_zh=candidate.display_name_zh,
                support_type=candidate.support_type,
                capacity_class=candidate.capacity_class,
                near_power=candidate.power_state is PowerState.NEAR,
                evidence_refs=sorted(
                    set(candidate.evidence_refs + candidate.power_evidence_refs)
                ),
            )
            for candidate in accepted
        ]
        region_manifest = RegionManifest(
            video_id=video_id,
            entries=sorted(entries, key=lambda entry: entry.region_id),
            notes=(
                "AUTO_PRODUCED from model observation JSONL; UNKNOWN power is "
                "projected as near_power=false."
            ),
        )

    expected_count = len(config.expected_anchor_labels)
    observed_expected = expected_count - len(not_observed)
    power_counts = Counter(candidate.power_state.value for candidate in candidates)
    metrics = SpatialMetrics(
        video_id=video_id,
        observation_count=len(validated),
        candidate_count=len(candidates),
        observed_candidate_count=len(candidates) - len(not_observed),
        not_observed_count=len(not_observed),
        auto_accepted_count=len(accepted),
        needs_user_count=sum(
            candidate.status is CandidateStatus.NEEDS_USER for candidate in candidates
        ),
        projected_region_count=(len(accepted) if region_manifest is not None else 0),
        min_regions_required=config.min_regions,
        region_gate_passed=region_manifest is not None,
        expected_anchor_count=expected_count,
        observed_expected_anchor_count=observed_expected,
        expected_coverage_rate=(
            round(observed_expected / expected_count, 6) if expected_count else None
        ),
        power_state_counts=dict(sorted(power_counts.items())),
        gate_status=gate_status,
        gate_reasons=gate_reasons,
    )
    digest = _normalized_hash(candidate_manifest, region_manifest, metrics)
    metrics = metrics.model_copy(update={"normalized_hash": digest})
    return SpatialProductionResult(
        candidate_manifest=candidate_manifest,
        region_manifest=region_manifest,
        metrics=metrics,
        normalized_hash=digest,
    )


def load_observations_jsonl(
    path: str | Path, *, video_id: str | None = None
) -> list[SpatialObservation]:
    """Load strict JSONL observations, retaining line-numbered errors."""

    source = Path(path)
    observations: list[SpatialObservation] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            try:
                observation = SpatialObservation.model_validate(payload)
            except ValueError as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            if video_id and observation.video_id and observation.video_id != video_id:
                raise ValueError(
                    f"{source}:{line_number}: video_id {observation.video_id!r} "
                    f"does not match {video_id!r}"
                )
            observations.append(observation)
    return observations


def _write_json(path: Path, model: BaseModel) -> None:
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_spatial_outputs(
    result: SpatialProductionResult, out_dir: str | Path
) -> dict[str, str]:
    """Write deterministic artifacts and remove a stale production manifest.

    A failed gate must not leave a previous successful ``regions.json`` in the
    same output directory, otherwise a downstream run could accidentally
    consume stale trusted regions.
    """

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    candidate_path = destination / CANDIDATE_MANIFEST_FILENAME
    region_path = destination / REGION_MANIFEST_FILENAME
    metrics_path = destination / METRICS_FILENAME
    hash_path = destination / NORMALIZED_HASH_FILENAME

    _write_json(candidate_path, result.candidate_manifest)
    if result.region_manifest is not None:
        _write_json(region_path, result.region_manifest)
    elif region_path.exists():
        region_path.unlink()
    _write_json(metrics_path, result.metrics)
    hash_path.write_text(result.normalized_hash + "\n", encoding="ascii")

    written = {
        "candidate_manifest": str(candidate_path),
        "metrics": str(metrics_path),
        "normalized_hash": str(hash_path),
    }
    if result.region_manifest is not None:
        written["region_manifest"] = str(region_path)
    return written

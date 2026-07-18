"""Independent semantic scorer for automatic spatial-region production.

The scorer is intentionally a sidecar: it consumes the automatic
``RegionManifest`` and a frozen truth set that contains semantic labels only.
It never projects, rewrites, or relabels regions and therefore cannot become a
fallback producer for the automatic pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.schemas.core import CapacityClass, SupportType
from backend.schemas.hero_bundle import RegionEntry, RegionManifest

SCORING_SCHEMA_VERSION = "1.0"
SCORE_MANIFEST_FILENAME = "score_manifest.json"
SCORE_METRICS_FILENAME = "metrics.json"
SCORE_NORMALIZED_HASH_FILENAME = "normalized.sha256"


def canonical_anchor(value: str) -> str:
    """Return the stable semantic key used for truth/prediction matching."""

    text = unicodedata.normalize("NFKC", value).casefold().strip()
    canonical = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")
    if not canonical:
        raise ValueError("anchor must contain at least one letter or number")
    return canonical


class _ScoringContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = SCORING_SCHEMA_VERSION


class FrozenSpatialTruthEntry(_ScoringContract):
    """One semantic truth row, deliberately free of prediction identifiers."""

    anchor: str
    support_type: SupportType
    capacity_class: CapacityClass
    near_power: bool | None = None

    @field_validator("anchor")
    @classmethod
    def _canonical_anchor(cls, value: str) -> str:
        return canonical_anchor(value)


class FrozenSpatialTruthManifest(_ScoringContract):
    """Frozen scorer input containing only semantic ground-truth fields."""

    entries: list[FrozenSpatialTruthEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_anchors(self) -> "FrozenSpatialTruthManifest":
        anchors = [entry.anchor for entry in self.entries]
        if len(anchors) != len(set(anchors)):
            raise ValueError("truth anchors must be unique after canonicalization")
        return self


class SemanticMatchStatus(str, Enum):
    EXACT = "EXACT"
    MISMATCH = "MISMATCH"
    MISSING = "MISSING"


class ExtraPredictionReason(str, Enum):
    UNEXPECTED_ANCHOR = "UNEXPECTED_ANCHOR"
    DUPLICATE_ANCHOR = "DUPLICATE_ANCHOR"


class SpatialSemanticMatch(_ScoringContract):
    anchor: str
    status: SemanticMatchStatus
    expected_support_type: SupportType
    predicted_support_type: SupportType | None = None
    support_type_matches: bool = False
    expected_capacity_class: CapacityClass
    predicted_capacity_class: CapacityClass | None = None
    capacity_class_matches: bool = False
    expected_near_power: bool | None = None
    predicted_near_power: bool | None = None
    power_matches: bool | None = None


class SpatialExtraPrediction(_ScoringContract):
    anchor: str
    reason: ExtraPredictionReason
    support_type: SupportType
    capacity_class: CapacityClass
    near_power: bool


class SpatialScoreManifest(_ScoringContract):
    """Identifier-free semantic comparison emitted by the scoring sidecar."""

    source_video_id: str
    required_expected_anchor_count: int = Field(ge=1)
    truth_anchor_count: int = Field(ge=1)
    prediction_entry_count: int = Field(ge=0)
    score_numerator: int = Field(ge=0)
    score_denominator: int = Field(ge=1)
    score: str
    matches: list[SpatialSemanticMatch]
    extra_predictions: list[SpatialExtraPrediction]
    accepted: bool
    gate_reasons: list[str]


class SpatialScoreMetrics(_ScoringContract):
    source_video_id: str
    required_expected_anchor_count: int = Field(ge=1)
    truth_anchor_count: int = Field(ge=1)
    prediction_entry_count: int = Field(ge=0)
    matched_anchor_count: int = Field(ge=0)
    exact_semantic_match_count: int = Field(ge=0)
    missing_anchor_count: int = Field(ge=0)
    extra_prediction_count: int = Field(ge=0)
    support_type_mismatch_count: int = Field(ge=0)
    capacity_class_mismatch_count: int = Field(ge=0)
    informational_power_mismatch_count: int = Field(ge=0)
    score: str
    acceptance_passed: bool
    gate_reasons: list[str]
    normalized_hash: str = ""


class SpatialScoringResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    score_manifest: SpatialScoreManifest
    metrics: SpatialScoreMetrics
    normalized_hash: str


def _canonical_payload(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {key: _canonical_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_payload(item) for item in value]
    return value


def _normalized_hash(
    manifest: SpatialScoreManifest,
    metrics: SpatialScoreMetrics,
) -> str:
    payload = {
        "score_manifest": manifest.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json", exclude={"normalized_hash"}),
    }
    canonical = json.dumps(
        _canonical_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prediction_key(entry: RegionEntry) -> tuple[str, str, bool]:
    return (entry.support_type.value, entry.capacity_class.value, entry.near_power)


def _best_prediction_key(
    entry: RegionEntry,
    truth: FrozenSpatialTruthEntry,
) -> tuple[int, int, str, str, bool]:
    """Prefer the closest semantic duplicate, then use a stable lexical tie-break."""

    return (
        int(entry.support_type is not truth.support_type),
        int(entry.capacity_class is not truth.capacity_class),
        *_prediction_key(entry),
    )


def _extra_prediction(
    anchor: str,
    entry: RegionEntry,
    reason: ExtraPredictionReason,
) -> SpatialExtraPrediction:
    return SpatialExtraPrediction(
        anchor=anchor,
        reason=reason,
        support_type=entry.support_type,
        capacity_class=entry.capacity_class,
        near_power=entry.near_power,
    )


def score_spatial_regions(
    region_manifest: RegionManifest | Mapping[str, Any],
    truth_manifest: FrozenSpatialTruthManifest | Mapping[str, Any],
    *,
    required_expected_anchor_count: int = 5,
) -> SpatialScoringResult:
    """Compare automatic regions with semantic-only truth, failing closed.

    Power is retained as informational evidence but does not affect the score or
    acceptance.  Support type and capacity class are both required by the input
    contracts and both must match for an anchor to score as exact.
    """

    if required_expected_anchor_count < 1:
        raise ValueError("required_expected_anchor_count must be at least 1")

    prediction = (
        region_manifest
        if isinstance(region_manifest, RegionManifest)
        else RegionManifest.model_validate(region_manifest)
    )
    truth = (
        truth_manifest
        if isinstance(truth_manifest, FrozenSpatialTruthManifest)
        else FrozenSpatialTruthManifest.model_validate(truth_manifest)
    )

    truth_by_anchor = {entry.anchor: entry for entry in truth.entries}
    prediction_by_anchor: dict[str, list[RegionEntry]] = {}
    for entry in prediction.entries:
        prediction_by_anchor.setdefault(canonical_anchor(entry.anchor), []).append(entry)

    matches: list[SpatialSemanticMatch] = []
    extras: list[SpatialExtraPrediction] = []
    for anchor in sorted(truth_by_anchor):
        expected = truth_by_anchor[anchor]
        candidates = prediction_by_anchor.pop(anchor, [])
        if not candidates:
            matches.append(
                SpatialSemanticMatch(
                    anchor=anchor,
                    status=SemanticMatchStatus.MISSING,
                    expected_support_type=expected.support_type,
                    expected_capacity_class=expected.capacity_class,
                    expected_near_power=expected.near_power,
                )
            )
            continue

        ordered = sorted(candidates, key=lambda item: _best_prediction_key(item, expected))
        selected, duplicates = ordered[0], ordered[1:]
        support_matches = selected.support_type is expected.support_type
        capacity_matches = selected.capacity_class is expected.capacity_class
        power_matches = (
            None
            if expected.near_power is None
            else selected.near_power == expected.near_power
        )
        matches.append(
            SpatialSemanticMatch(
                anchor=anchor,
                status=(
                    SemanticMatchStatus.EXACT
                    if support_matches and capacity_matches
                    else SemanticMatchStatus.MISMATCH
                ),
                expected_support_type=expected.support_type,
                predicted_support_type=selected.support_type,
                support_type_matches=support_matches,
                expected_capacity_class=expected.capacity_class,
                predicted_capacity_class=selected.capacity_class,
                capacity_class_matches=capacity_matches,
                expected_near_power=expected.near_power,
                predicted_near_power=selected.near_power,
                power_matches=power_matches,
            )
        )
        extras.extend(
            _extra_prediction(anchor, duplicate, ExtraPredictionReason.DUPLICATE_ANCHOR)
            for duplicate in duplicates
        )

    for anchor in sorted(prediction_by_anchor):
        extras.extend(
            _extra_prediction(anchor, entry, ExtraPredictionReason.UNEXPECTED_ANCHOR)
            for entry in sorted(prediction_by_anchor[anchor], key=_prediction_key)
        )
    extras.sort(
        key=lambda item: (
            item.anchor,
            item.reason.value,
            item.support_type.value,
            item.capacity_class.value,
            item.near_power,
        )
    )

    exact_count = sum(match.status is SemanticMatchStatus.EXACT for match in matches)
    matched_count = sum(match.status is not SemanticMatchStatus.MISSING for match in matches)
    missing_count = len(matches) - matched_count
    support_mismatches = sum(
        match.status is not SemanticMatchStatus.MISSING
        and not match.support_type_matches
        for match in matches
    )
    capacity_mismatches = sum(
        match.status is not SemanticMatchStatus.MISSING
        and not match.capacity_class_matches
        for match in matches
    )
    power_mismatches = sum(match.power_matches is False for match in matches)
    score = f"{exact_count}/{len(truth.entries)}"

    gate_reasons: list[str] = []
    if len(truth.entries) != required_expected_anchor_count:
        gate_reasons.append(
            "truth_expected_anchor_count_mismatch:"
            f"{len(truth.entries)}!={required_expected_anchor_count}"
        )
    if missing_count:
        gate_reasons.append(f"missing_expected_anchors:{missing_count}")
    duplicate_count = sum(
        extra.reason is ExtraPredictionReason.DUPLICATE_ANCHOR for extra in extras
    )
    unexpected_count = len(extras) - duplicate_count
    if duplicate_count:
        gate_reasons.append(f"duplicate_prediction_anchors:{duplicate_count}")
    if unexpected_count:
        gate_reasons.append(f"unexpected_prediction_anchors:{unexpected_count}")
    if support_mismatches:
        gate_reasons.append(f"support_type_mismatches:{support_mismatches}")
    if capacity_mismatches:
        gate_reasons.append(f"capacity_class_mismatches:{capacity_mismatches}")

    accepted = not gate_reasons and exact_count == required_expected_anchor_count
    manifest = SpatialScoreManifest(
        source_video_id=prediction.video_id,
        required_expected_anchor_count=required_expected_anchor_count,
        truth_anchor_count=len(truth.entries),
        prediction_entry_count=len(prediction.entries),
        score_numerator=exact_count,
        score_denominator=len(truth.entries),
        score=score,
        matches=matches,
        extra_predictions=extras,
        accepted=accepted,
        gate_reasons=gate_reasons,
    )
    metrics = SpatialScoreMetrics(
        source_video_id=prediction.video_id,
        required_expected_anchor_count=required_expected_anchor_count,
        truth_anchor_count=len(truth.entries),
        prediction_entry_count=len(prediction.entries),
        matched_anchor_count=matched_count,
        exact_semantic_match_count=exact_count,
        missing_anchor_count=missing_count,
        extra_prediction_count=len(extras),
        support_type_mismatch_count=support_mismatches,
        capacity_class_mismatch_count=capacity_mismatches,
        informational_power_mismatch_count=power_mismatches,
        score=score,
        acceptance_passed=accepted,
        gate_reasons=gate_reasons,
    )
    digest = _normalized_hash(manifest, metrics)
    metrics = metrics.model_copy(update={"normalized_hash": digest})
    return SpatialScoringResult(
        score_manifest=manifest,
        metrics=metrics,
        normalized_hash=digest,
    )


def load_frozen_spatial_truth(
    path: str | Path,
) -> FrozenSpatialTruthManifest:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
    return FrozenSpatialTruthManifest.model_validate(payload)


def load_region_manifest(path: str | Path) -> RegionManifest:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
    return RegionManifest.model_validate(payload)


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


def write_spatial_scoring_outputs(
    result: SpatialScoringResult,
    out_dir: str | Path,
) -> dict[str, str]:
    """Write scorer diagnostics only; never create a solver regions manifest."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / SCORE_MANIFEST_FILENAME
    metrics_path = destination / SCORE_METRICS_FILENAME
    hash_path = destination / SCORE_NORMALIZED_HASH_FILENAME

    manifest_path.write_bytes(_json_bytes(result.score_manifest))
    metrics_path.write_bytes(_json_bytes(result.metrics))
    hash_path.write_text(result.normalized_hash + "\n", encoding="ascii")
    return {
        "score_manifest": str(manifest_path),
        "metrics": str(metrics_path),
        "normalized_hash": str(hash_path),
    }

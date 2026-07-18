"""Deterministic one-to-one assignment of automatic spatial anchors.

The assignment layer consumes only automatic candidate statistics.  It has no
manual region, review, or track-ID override input.  Expected anchor labels are
the generic vocabulary to cover; candidate hypotheses contain model label
votes and confidence summaries.

Each expected anchor and each automatic visual instance may be selected at
most once.  The solver maximizes assignment cardinality first and the summed
explainable edge score second.  Absolute score and alternative-assignment
margins are gated separately, so deterministic tie-breaking never turns an
ambiguous result into trusted output.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.schemas.core import CapacityClass, SupportType
from backend.tools.spatial.producer import GateStatus, PowerState

ASSIGNMENT_SCHEMA_VERSION = "1.0"
_SCORE_SCALE = 100_000_000


class _AssignmentContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = ASSIGNMENT_SCHEMA_VERSION


def _canonical_anchor(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")


def _clean_nonempty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


class AutomaticAnchorHypothesis(_AssignmentContract):
    """Automatic label statistics for one candidate and one anchor."""

    anchor: str
    label_vote_count: int = Field(ge=1)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    max_confidence: float = Field(ge=0.0, le=1.0)
    proposal_display_name_zh: str = ""
    support_type: SupportType | None = None
    support_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    capacity_class: CapacityClass | None = None
    capacity_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("anchor")
    @classmethod
    def _clean_anchor(cls, value: str) -> str:
        cleaned = _clean_nonempty(value, field_name="anchor")
        if not _canonical_anchor(cleaned):
            raise ValueError("anchor has an empty canonical form")
        return cleaned

    @model_validator(mode="after")
    def _mean_cannot_exceed_max(self) -> "AutomaticAnchorHypothesis":
        if self.mean_confidence > self.max_confidence:
            raise ValueError("mean_confidence cannot exceed max_confidence")
        return self


class AutomaticAnchorCandidate(_AssignmentContract):
    """One automatically derived visual candidate offered to the solver.

    ``visual_instance_id`` groups parallel category tracks that automatic
    geometry/tracking considers the same physical object.  The solver may use
    at most one candidate from a visual instance.
    """

    candidate_id: str
    visual_instance_id: str
    observation_count: int = Field(ge=1)
    display_name_zh: str
    power_state: PowerState = PowerState.UNKNOWN
    power_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    power_evidence_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    source_track_ids: list[str] = Field(default_factory=list)
    model_versions: list[str] = Field(default_factory=list)
    anchor_hypotheses: list[AutomaticAnchorHypothesis] = Field(min_length=1)

    @field_validator("candidate_id", "visual_instance_id", "display_name_zh")
    @classmethod
    def _clean_required_strings(cls, value: str, info: Any) -> str:
        return _clean_nonempty(value, field_name=info.field_name)

    @field_validator(
        "power_evidence_refs", "evidence_refs", "source_track_ids", "model_versions"
    )
    @classmethod
    def _clean_string_lists(cls, value: list[str], info: Any) -> list[str]:
        cleaned = [
            _clean_nonempty(item, field_name=info.field_name) for item in value
        ]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"{info.field_name} contains duplicates")
        return cleaned

    @model_validator(mode="after")
    def _valid_hypothesis_counts(self) -> "AutomaticAnchorCandidate":
        canonical = [
            _canonical_anchor(hypothesis.anchor)
            for hypothesis in self.anchor_hypotheses
        ]
        if len(canonical) != len(set(canonical)):
            raise ValueError("anchor_hypotheses contains duplicate anchors")
        for hypothesis in self.anchor_hypotheses:
            if hypothesis.label_vote_count > self.observation_count:
                raise ValueError(
                    f"{hypothesis.anchor}: label_vote_count exceeds observation_count"
                )
        return self


class AnchorAssignmentConfig(_AssignmentContract):
    min_candidate_observations: int = Field(default=2, ge=1)
    min_label_vote_count: int = Field(default=2, ge=1)
    min_label_vote_share: float = Field(default=0.60, ge=0.0, le=1.0)
    min_mean_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    min_hard_field_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    min_power_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    min_assignment_score: float = Field(default=0.70, ge=0.0, le=1.0)
    min_runner_up_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    support_saturation_observations: int = Field(default=5, ge=1)
    mean_confidence_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    max_confidence_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    label_vote_share_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    observation_support_weight: float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _score_weights_sum_to_one(self) -> "AnchorAssignmentConfig":
        total = math.fsum(
            [
                self.mean_confidence_weight,
                self.max_confidence_weight,
                self.label_vote_share_weight,
                self.observation_support_weight,
            ]
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("assignment score weights must sum to 1.0")
        return self


class AnchorScoreComponents(_AssignmentContract):
    mean_confidence: float = Field(ge=0.0, le=1.0)
    max_confidence: float = Field(ge=0.0, le=1.0)
    label_vote_share: float = Field(ge=0.0, le=1.0)
    observation_support: float = Field(ge=0.0, le=1.0)


class AnchorAssignmentEdge(_AssignmentContract):
    candidate_id: str
    visual_instance_id: str
    anchor: str
    score: float = Field(ge=0.0, le=1.0)
    score_components: AnchorScoreComponents
    support_type: SupportType | None = None
    support_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    capacity_class: CapacityClass | None = None
    capacity_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class SelectedAnchorAssignment(_AssignmentContract):
    anchor: str
    candidate_id: str
    visual_instance_id: str
    display_name_zh: str
    support_type: SupportType
    capacity_class: CapacityClass
    source_power_state: PowerState
    power_state: PowerState
    power_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    power_evidence_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str]
    source_track_ids: list[str] = Field(default_factory=list)
    model_versions: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)
    score_components: AnchorScoreComponents
    local_runner_up_candidate_id: str | None = None
    local_runner_up_score: float | None = Field(default=None, ge=0.0, le=1.0)
    runner_up_total_score: float | None = Field(default=None, ge=0.0)
    runner_up_margin: float | None = Field(default=None, ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class AutomaticAnchorAssignmentResult(_AssignmentContract):
    source_kind: Literal["AUTOMATIC_ANCHOR_HYPOTHESES"] = (
        "AUTOMATIC_ANCHOR_HYPOTHESES"
    )
    config: AnchorAssignmentConfig
    status: GateStatus
    expected_anchor_labels: list[str]
    input_hash: str
    total_score: float = Field(ge=0.0)
    runner_up_total_score: float | None = Field(default=None, ge=0.0)
    runner_up_margin: float | None = Field(default=None, ge=0.0)
    assignments: list[SelectedAnchorAssignment] = Field(default_factory=list)
    unassigned_anchor_labels: list[str] = Field(default_factory=list)
    edges: list[AnchorAssignmentEdge] = Field(default_factory=list)
    gate_reasons: list[str] = Field(default_factory=list)
    normalized_hash: str

    @property
    def gate_passed(self) -> bool:
        return self.status is GateStatus.PASS


@dataclass(frozen=True)
class _Edge:
    candidate_id: str
    visual_instance_id: str
    anchor_index: int
    anchor: str
    score_units: int
    score: float
    components: AnchorScoreComponents
    hypothesis: AutomaticAnchorHypothesis

    @property
    def key(self) -> tuple[str, str]:
        return self.candidate_id, self.anchor


@dataclass(frozen=True)
class _Solution:
    selected: tuple[_Edge | None, ...]
    score_units: int

    @property
    def assigned_count(self) -> int:
        return sum(edge is not None for edge in self.selected)

    @property
    def mask(self) -> int:
        mask = 0
        for index, edge in enumerate(self.selected):
            if edge is not None:
                mask |= 1 << index
        return mask

    @property
    def signature(self) -> tuple[tuple[str, str], ...]:
        missing = "\uffff"
        return tuple(
            (edge.visual_instance_id, edge.candidate_id)
            if edge is not None
            else (missing, missing)
            for edge in self.selected
        )


def _is_better(candidate: _Solution, incumbent: _Solution | None) -> bool:
    if incumbent is None:
        return True
    if candidate.assigned_count != incumbent.assigned_count:
        return candidate.assigned_count > incumbent.assigned_count
    if candidate.score_units != incumbent.score_units:
        return candidate.score_units > incumbent.score_units
    return candidate.signature < incumbent.signature


def _solve(
    edges: Sequence[_Edge],
    *,
    anchor_count: int,
    excluded_edge: tuple[str, str] | None = None,
) -> _Solution:
    empty = _Solution(selected=(None,) * anchor_count, score_units=0)
    by_instance: dict[str, list[_Edge]] = {}
    for edge in edges:
        if excluded_edge is not None and edge.key == excluded_edge:
            continue
        by_instance.setdefault(edge.visual_instance_id, []).append(edge)

    states: dict[int, _Solution] = {0: empty}
    for instance_id in sorted(by_instance):
        options = sorted(
            by_instance[instance_id],
            key=lambda edge: (
                edge.anchor_index,
                -edge.score_units,
                edge.candidate_id,
            ),
        )
        next_states = dict(states)
        for mask, solution in states.items():
            for edge in options:
                bit = 1 << edge.anchor_index
                if mask & bit:
                    continue
                selected = list(solution.selected)
                selected[edge.anchor_index] = edge
                proposed = _Solution(
                    selected=tuple(selected),
                    score_units=solution.score_units + edge.score_units,
                )
                new_mask = mask | bit
                if _is_better(proposed, next_states.get(new_mask)):
                    next_states[new_mask] = proposed
        states = next_states

    best: _Solution | None = None
    for solution in states.values():
        if _is_better(solution, best):
            best = solution
    assert best is not None
    return best


def _score_hypothesis(
    candidate: AutomaticAnchorCandidate,
    hypothesis: AutomaticAnchorHypothesis,
    config: AnchorAssignmentConfig,
) -> tuple[float, AnchorScoreComponents]:
    vote_share = hypothesis.label_vote_count / candidate.observation_count
    observation_support = min(
        1.0,
        hypothesis.label_vote_count / config.support_saturation_observations,
    )
    components = AnchorScoreComponents(
        mean_confidence=round(hypothesis.mean_confidence, 8),
        max_confidence=round(hypothesis.max_confidence, 8),
        label_vote_share=round(vote_share, 8),
        observation_support=round(observation_support, 8),
    )
    score = math.fsum(
        [
            config.mean_confidence_weight * components.mean_confidence,
            config.max_confidence_weight * components.max_confidence,
            config.label_vote_share_weight * components.label_vote_share,
            config.observation_support_weight * components.observation_support,
        ]
    )
    return round(score, 8), components


def _edge_rejection_reasons(
    candidate: AutomaticAnchorCandidate,
    hypothesis: AutomaticAnchorHypothesis,
    components: AnchorScoreComponents,
    config: AnchorAssignmentConfig,
) -> list[str]:
    reasons: list[str] = []
    if candidate.observation_count < config.min_candidate_observations:
        reasons.append("candidate_observation_count_below_threshold")
    if hypothesis.label_vote_count < config.min_label_vote_count:
        reasons.append("label_vote_count_below_threshold")
    if components.label_vote_share < config.min_label_vote_share:
        reasons.append("label_vote_share_below_threshold")
    if hypothesis.mean_confidence < config.min_mean_confidence:
        reasons.append("mean_confidence_below_threshold")
    if hypothesis.support_type is None:
        reasons.append("support_type_missing")
    elif (
        hypothesis.support_confidence is None
        or hypothesis.support_confidence < config.min_hard_field_confidence
    ):
        reasons.append("support_confidence_missing_or_below_threshold")
    if hypothesis.capacity_class is None:
        reasons.append("capacity_class_missing")
    elif (
        hypothesis.capacity_confidence is None
        or hypothesis.capacity_confidence < config.min_hard_field_confidence
    ):
        reasons.append("capacity_confidence_missing_or_below_threshold")
    if not candidate.evidence_refs:
        reasons.append("candidate_evidence_missing")
    return sorted(reasons)


def _effective_power_state(
    candidate: AutomaticAnchorCandidate,
    config: AnchorAssignmentConfig,
) -> tuple[PowerState, list[str]]:
    if candidate.power_state is PowerState.UNKNOWN:
        return PowerState.UNKNOWN, ["power_state_unknown_non_blocking"]
    if (
        candidate.power_confidence is None
        or candidate.power_confidence < config.min_power_confidence
    ):
        return PowerState.UNKNOWN, [
            "power_state_downgraded_low_confidence_non_blocking"
        ]
    if candidate.power_state is PowerState.NEAR and not candidate.power_evidence_refs:
        return PowerState.UNKNOWN, [
            "power_near_downgraded_missing_evidence_non_blocking"
        ]
    return candidate.power_state, []


def _canonical_payload(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {key: _canonical_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_payload(item) for item in value]
    return value


def _hash_payload(payload: object) -> str:
    canonical = json.dumps(
        _canonical_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_candidates(
    candidates: Sequence[AutomaticAnchorCandidate],
) -> list[AutomaticAnchorCandidate]:
    normalized: list[AutomaticAnchorCandidate] = []
    for candidate in candidates:
        normalized.append(
            candidate.model_copy(
                update={
                    "power_evidence_refs": sorted(candidate.power_evidence_refs),
                    "evidence_refs": sorted(candidate.evidence_refs),
                    "source_track_ids": sorted(candidate.source_track_ids),
                    "model_versions": sorted(candidate.model_versions),
                    "anchor_hypotheses": sorted(
                        candidate.anchor_hypotheses,
                        key=lambda item: _canonical_anchor(item.anchor),
                    ),
                }
            )
        )
    return sorted(normalized, key=lambda item: item.candidate_id)


def assign_automatic_anchors(
    expected_anchor_labels: Sequence[str],
    candidates: Sequence[AutomaticAnchorCandidate | Mapping[str, Any]],
    config: AnchorAssignmentConfig | None = None,
) -> AutomaticAnchorAssignmentResult:
    """Globally assign expected anchors to distinct automatic visual instances."""

    config = config or AnchorAssignmentConfig()
    expected_pairs = [
        (
            _canonical_anchor(_clean_nonempty(anchor, field_name="expected_anchor_labels")),
            _clean_nonempty(anchor, field_name="expected_anchor_labels"),
        )
        for anchor in expected_anchor_labels
    ]
    if not expected_pairs:
        raise ValueError("expected_anchor_labels must not be empty")
    if any(not canonical for canonical, _ in expected_pairs):
        raise ValueError("expected_anchor_labels contains an empty canonical label")
    if len({canonical for canonical, _ in expected_pairs}) != len(expected_pairs):
        raise ValueError("expected_anchor_labels contains duplicates")
    expected_pairs.sort(key=lambda item: item[0])
    expected = [label for _, label in expected_pairs]
    expected_index = {
        canonical: index for index, (canonical, _) in enumerate(expected_pairs)
    }

    validated = [
        item
        if isinstance(item, AutomaticAnchorCandidate)
        else AutomaticAnchorCandidate.model_validate(item)
        for item in candidates
    ]
    candidate_ids = [candidate.candidate_id for candidate in validated]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id duplicates are forbidden")
    validated = _normalize_candidates(validated)

    input_hash = _hash_payload(
        {
            "config": config.model_dump(mode="json"),
            "expected_anchor_labels": expected,
            "candidates": [
                candidate.model_dump(mode="json") for candidate in validated
            ],
        }
    )

    candidates_by_id = {candidate.candidate_id: candidate for candidate in validated}
    eligible_edges: list[_Edge] = []
    diagnostic_edges: list[AnchorAssignmentEdge] = []
    for candidate in validated:
        for hypothesis in candidate.anchor_hypotheses:
            canonical = _canonical_anchor(hypothesis.anchor)
            if canonical not in expected_index:
                continue
            anchor_index = expected_index[canonical]
            anchor = expected[anchor_index]
            score, components = _score_hypothesis(candidate, hypothesis, config)
            rejection_reasons = _edge_rejection_reasons(
                candidate,
                hypothesis,
                components,
                config,
            )
            diagnostic_edges.append(
                AnchorAssignmentEdge(
                    candidate_id=candidate.candidate_id,
                    visual_instance_id=candidate.visual_instance_id,
                    anchor=anchor,
                    score=score,
                    score_components=components,
                    support_type=hypothesis.support_type,
                    support_confidence=hypothesis.support_confidence,
                    capacity_class=hypothesis.capacity_class,
                    capacity_confidence=hypothesis.capacity_confidence,
                    eligible=not rejection_reasons,
                    rejection_reasons=rejection_reasons,
                )
            )
            if not rejection_reasons:
                eligible_edges.append(
                    _Edge(
                        candidate_id=candidate.candidate_id,
                        visual_instance_id=candidate.visual_instance_id,
                        anchor_index=anchor_index,
                        anchor=anchor,
                        score_units=int(round(score * _SCORE_SCALE)),
                        score=score,
                        components=components,
                        hypothesis=hypothesis,
                    )
                )

    diagnostic_edges.sort(key=lambda item: (item.anchor, item.candidate_id))
    eligible_edges.sort(
        key=lambda item: (item.visual_instance_id, item.anchor_index, item.candidate_id)
    )
    solution = _solve(eligible_edges, anchor_count=len(expected))
    complete_mask = (1 << len(expected)) - 1
    complete = solution.mask == complete_mask

    alternate_by_edge: dict[tuple[str, str], _Solution | None] = {}
    if complete:
        for edge in solution.selected:
            assert edge is not None
            alternative = _solve(
                eligible_edges,
                anchor_count=len(expected),
                excluded_edge=edge.key,
            )
            alternate_by_edge[edge.key] = (
                alternative if alternative.mask == complete_mask else None
            )

    assignments: list[SelectedAnchorAssignment] = []
    gate_reasons: list[str] = []
    unassigned: list[str] = []
    for anchor_index, anchor in enumerate(expected):
        edge = solution.selected[anchor_index]
        if edge is None:
            unassigned.append(anchor)
            continue
        candidate = candidates_by_id[edge.candidate_id]
        local_alternatives = sorted(
            (
                alternative
                for alternative in eligible_edges
                if alternative.anchor_index == anchor_index
                and alternative.candidate_id != edge.candidate_id
                and alternative.visual_instance_id != edge.visual_instance_id
            ),
            key=lambda item: (-item.score_units, item.candidate_id),
        )
        local_runner = local_alternatives[0] if local_alternatives else None
        alternative_solution = alternate_by_edge.get(edge.key)
        runner_total = (
            round(alternative_solution.score_units / _SCORE_SCALE, 8)
            if alternative_solution is not None
            else None
        )
        margin = (
            round(
                (solution.score_units - alternative_solution.score_units)
                / _SCORE_SCALE,
                8,
            )
            if alternative_solution is not None
            else None
        )
        effective_power, warnings = _effective_power_state(candidate, config)
        assert edge.hypothesis.support_type is not None
        assert edge.hypothesis.capacity_class is not None
        display_name = (
            edge.hypothesis.proposal_display_name_zh.strip()
            or candidate.display_name_zh
        )
        assignments.append(
            SelectedAnchorAssignment(
                anchor=anchor,
                candidate_id=candidate.candidate_id,
                visual_instance_id=candidate.visual_instance_id,
                display_name_zh=display_name,
                support_type=edge.hypothesis.support_type,
                capacity_class=edge.hypothesis.capacity_class,
                source_power_state=candidate.power_state,
                power_state=effective_power,
                power_confidence=candidate.power_confidence,
                power_evidence_refs=sorted(candidate.power_evidence_refs),
                evidence_refs=sorted(candidate.evidence_refs),
                source_track_ids=sorted(candidate.source_track_ids),
                model_versions=sorted(candidate.model_versions),
                score=edge.score,
                score_components=edge.components,
                local_runner_up_candidate_id=(
                    local_runner.candidate_id if local_runner else None
                ),
                local_runner_up_score=(local_runner.score if local_runner else None),
                runner_up_total_score=runner_total,
                runner_up_margin=margin,
                warnings=warnings,
            )
        )
        if edge.score < config.min_assignment_score:
            gate_reasons.append(
                f"assignment_score_below_threshold:{anchor}:"
                f"{edge.score:.8f}/{config.min_assignment_score:.8f}"
            )
        if margin is not None and margin < config.min_runner_up_margin:
            gate_reasons.append(
                f"assignment_margin_below_threshold:{anchor}:"
                f"{margin:.8f}/{config.min_runner_up_margin:.8f}"
            )

    if not complete:
        gate_reasons.insert(
            0,
            "complete_one_to_one_assignment_not_found:"
            + ",".join(unassigned),
        )

    alternate_scores = [
        alternative.score_units
        for alternative in alternate_by_edge.values()
        if alternative is not None
    ]
    runner_up_units = max(alternate_scores) if alternate_scores else None
    total_score = round(solution.score_units / _SCORE_SCALE, 8)
    runner_up_total = (
        round(runner_up_units / _SCORE_SCALE, 8)
        if runner_up_units is not None
        else None
    )
    runner_up_margin = (
        round((solution.score_units - runner_up_units) / _SCORE_SCALE, 8)
        if runner_up_units is not None
        else None
    )
    gate_reasons = sorted(set(gate_reasons))
    status = GateStatus.NEEDS_USER if gate_reasons else GateStatus.PASS

    result_payload = {
        "config": config.model_dump(mode="json"),
        "status": status.value,
        "expected_anchor_labels": expected,
        "input_hash": input_hash,
        "total_score": total_score,
        "runner_up_total_score": runner_up_total,
        "runner_up_margin": runner_up_margin,
        "assignments": [item.model_dump(mode="json") for item in assignments],
        "unassigned_anchor_labels": unassigned,
        "edges": [item.model_dump(mode="json") for item in diagnostic_edges],
        "gate_reasons": gate_reasons,
    }
    normalized_hash = _hash_payload(result_payload)
    return AutomaticAnchorAssignmentResult(
        **result_payload,
        normalized_hash=normalized_hash,
    )


def load_automatic_anchor_candidates(
    path: str | Path,
) -> list[AutomaticAnchorCandidate]:
    """Load a strict JSON array of automatic assignment candidates."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{source}: candidate root must be a JSON array")
    return [AutomaticAnchorCandidate.model_validate(item) for item in payload]

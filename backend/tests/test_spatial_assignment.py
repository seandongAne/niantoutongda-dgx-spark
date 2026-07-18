from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from backend.tools.spatial import (
    AnchorAssignmentConfig,
    AutomaticAnchorCandidate,
    GateStatus,
    assign_automatic_anchors,
)
from backend.tools.spatial.assignment import ExpectedAnchorContractManifest


def _score_only_config(
    *,
    min_score: float = 0.50,
    min_margin: float = 0.04,
) -> AnchorAssignmentConfig:
    return AnchorAssignmentConfig(
        min_candidate_observations=1,
        min_label_vote_count=1,
        min_label_vote_share=0.0,
        min_mean_confidence=0.0,
        min_assignment_score=min_score,
        min_runner_up_margin=min_margin,
        support_saturation_observations=1,
        mean_confidence_weight=1.0,
        max_confidence_weight=0.0,
        label_vote_share_weight=0.0,
        observation_support_weight=0.0,
    )


def _candidate(
    candidate_id: str,
    instance_id: str,
    hypotheses: dict[str, float],
    *,
    power_state: str = "UNKNOWN",
    power_confidence: float | None = None,
    power_evidence_refs: list[str] | None = None,
    support_type: str | None = "surface",
    capacity_class: str | None = "medium",
    support_confidence: float | None = 0.95,
    capacity_confidence: float | None = 0.95,
    evidence_refs: list[str] | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "visual_instance_id": instance_id,
        "observation_count": 10,
        "display_name_zh": f"自动候选 {candidate_id}",
        "power_state": power_state,
        "power_confidence": power_confidence,
        "power_evidence_refs": power_evidence_refs or [],
        "evidence_refs": (
            evidence_refs if evidence_refs is not None else [f"auto:{candidate_id}"]
        ),
        "source_track_ids": [f"track-{candidate_id}"],
        "model_versions": ["automatic-anchor-test-v1"],
        "anchor_hypotheses": [
            {
                "anchor": anchor,
                "label_vote_count": 10,
                "mean_confidence": confidence,
                "max_confidence": confidence,
                "proposal_display_name_zh": anchor,
                "support_type": support_type,
                "support_confidence": support_confidence,
                "capacity_class": capacity_class,
                "capacity_confidence": capacity_confidence,
            }
            for anchor, confidence in hypotheses.items()
        ],
    }


def test_global_assignment_beats_independent_argmax_and_is_deterministic():
    candidates = [
        _candidate("candidate-1", "instance-1", {"anchor_a": 0.90, "anchor_b": 0.85}),
        _candidate("candidate-2", "instance-2", {"anchor_a": 0.80}),
        _candidate("candidate-3", "instance-3", {"anchor_b": 0.70}),
    ]
    config = _score_only_config()

    result = assign_automatic_anchors(
        ["anchor_b", "anchor_a"],
        candidates,
        config,
    )
    reordered = assign_automatic_anchors(
        ["anchor_a", "anchor_b"],
        [
            {
                **candidate,
                "anchor_hypotheses": list(
                    reversed(candidate["anchor_hypotheses"])
                ),
            }
            for candidate in reversed(candidates)
        ],
        config,
    )

    assert result.gate_passed
    assert result.status is GateStatus.PASS
    assert [(item.anchor, item.candidate_id) for item in result.assignments] == [
        ("anchor_a", "candidate-2"),
        ("anchor_b", "candidate-1"),
    ]
    assert result.total_score == 1.65
    assert result.runner_up_total_score == 1.60
    assert result.runner_up_margin == 0.05
    assert all(item.runner_up_margin == 0.05 for item in result.assignments)
    assert result.input_hash == reordered.input_hash
    assert result.normalized_hash == reordered.normalized_hash


def test_visual_instance_is_a_one_to_one_resource_across_parallel_candidates():
    candidates = [
        _candidate("candidate-a", "shared-instance", {"anchor_a": 0.90}),
        _candidate("candidate-b", "shared-instance", {"anchor_b": 0.90}),
        _candidate("candidate-c", "instance-c", {"anchor_a": 0.70}),
        _candidate("candidate-d", "instance-d", {"anchor_b": 0.60}),
    ]

    result = assign_automatic_anchors(
        ["anchor_a", "anchor_b"],
        candidates,
        _score_only_config(min_margin=0.05),
    )

    assert result.gate_passed
    assert [(item.anchor, item.candidate_id) for item in result.assignments] == [
        ("anchor_a", "candidate-c"),
        ("anchor_b", "candidate-b"),
    ]
    assert len({item.visual_instance_id for item in result.assignments}) == 2
    assert result.runner_up_margin == 0.10


def test_equal_score_tie_is_deterministic_but_fails_margin_gate():
    candidates = [
        _candidate("candidate-z", "instance-z", {"anchor_a": 0.80}),
        _candidate("candidate-a", "instance-a", {"anchor_a": 0.80}),
    ]
    config = _score_only_config(min_margin=0.01)

    result = assign_automatic_anchors(["anchor_a"], candidates, config)
    reordered = assign_automatic_anchors(
        ["anchor_a"], list(reversed(candidates)), config
    )

    assert not result.gate_passed
    assert result.assignments[0].candidate_id == "candidate-a"
    assert result.assignments[0].runner_up_margin == 0.0
    assert result.runner_up_margin == 0.0
    assert any(
        reason.startswith("assignment_margin_below_threshold:anchor_a")
        for reason in result.gate_reasons
    )
    assert result.normalized_hash == reordered.normalized_hash


def test_unknown_power_is_explicit_and_non_blocking():
    result = assign_automatic_anchors(
        ["anchor_a"],
        [_candidate("candidate-a", "instance-a", {"anchor_a": 0.90})],
        _score_only_config(),
    )

    assert result.gate_passed
    assignment = result.assignments[0]
    assert assignment.source_power_state.value == "UNKNOWN"
    assert assignment.power_state.value == "UNKNOWN"
    assert assignment.power_evidence_refs == []
    assert assignment.warnings == ["power_state_unknown_non_blocking"]
    assert not any("power" in reason for reason in result.gate_reasons)


def test_low_confidence_near_downgrades_but_valid_evidenced_near_is_preserved():
    low = assign_automatic_anchors(
        ["anchor_a"],
        [
            _candidate(
                "candidate-a",
                "instance-a",
                {"anchor_a": 0.90},
                power_state="NEAR",
                power_confidence=0.40,
                power_evidence_refs=["auto:outlet-a"],
            )
        ],
        _score_only_config(),
    )
    valid = assign_automatic_anchors(
        ["anchor_a"],
        [
            _candidate(
                "candidate-a",
                "instance-a",
                {"anchor_a": 0.90},
                power_state="NEAR",
                power_confidence=0.95,
                power_evidence_refs=["auto:outlet-a"],
            )
        ],
        _score_only_config(),
    )

    assert low.gate_passed and valid.gate_passed
    assert low.assignments[0].source_power_state.value == "NEAR"
    assert low.assignments[0].power_state.value == "UNKNOWN"
    assert low.assignments[0].warnings == [
        "power_state_downgraded_low_confidence_non_blocking"
    ]
    assert valid.assignments[0].power_state.value == "NEAR"
    assert valid.assignments[0].warnings == []


def test_incomplete_matching_returns_best_partial_mapping_and_fails_closed():
    result = assign_automatic_anchors(
        ["anchor_a", "anchor_b"],
        [_candidate("candidate-a", "instance-a", {"anchor_a": 0.90})],
        _score_only_config(),
    )

    assert not result.gate_passed
    assert [(item.anchor, item.candidate_id) for item in result.assignments] == [
        ("anchor_a", "candidate-a")
    ]
    assert result.unassigned_anchor_labels == ["anchor_b"]
    assert any(
        reason == "complete_one_to_one_assignment_not_found:anchor_b"
        for reason in result.gate_reasons
    )


def test_missing_hard_fields_and_evidence_reject_but_invalid_near_only_downgrades():
    candidates = [
        _candidate(
            "missing-support",
            "instance-a",
            {"anchor_a": 0.90},
            support_type=None,
        ),
        _candidate(
            "missing-evidence",
            "instance-b",
            {"anchor_b": 0.90},
            evidence_refs=[],
        ),
        _candidate(
            "unsafe-near",
            "instance-c",
            {"anchor_c": 0.90},
            power_state="NEAR",
            power_confidence=0.95,
        ),
    ]

    result = assign_automatic_anchors(
        ["anchor_a", "anchor_b", "anchor_c"],
        candidates,
        _score_only_config(),
    )

    assert not result.gate_passed
    assert [(item.anchor, item.candidate_id) for item in result.assignments] == [
        ("anchor_c", "unsafe-near")
    ]
    by_candidate = {edge.candidate_id: edge for edge in result.edges}
    assert by_candidate["missing-support"].rejection_reasons == [
        "support_type_missing"
    ]
    assert by_candidate["missing-evidence"].rejection_reasons == [
        "candidate_evidence_missing"
    ]
    assert by_candidate["unsafe-near"].eligible is True
    selected = result.assignments[0]
    assert selected.source_power_state.value == "NEAR"
    assert selected.power_state.value == "UNKNOWN"
    assert selected.warnings == [
        "power_near_downgraded_missing_evidence_non_blocking"
    ]


def test_low_absolute_score_keeps_proposal_but_blocks_trust():
    result = assign_automatic_anchors(
        ["anchor_a"],
        [_candidate("candidate-a", "instance-a", {"anchor_a": 0.60})],
        _score_only_config(min_score=0.70),
    )

    assert not result.gate_passed
    assert result.assignments[0].candidate_id == "candidate-a"
    assert result.assignments[0].score == 0.60
    assert result.gate_reasons == [
        "assignment_score_below_threshold:anchor_a:0.60000000/0.70000000"
    ]


def test_production_contract_rejects_higher_score_semantic_mismatch():
    candidates = [
        _candidate(
            "higher-but-wrong",
            "instance-wrong",
            {"display_cabinet": 0.95},
            support_type="shelf",
            capacity_class="medium",
        ),
        _candidate(
            "lower-and-matching",
            "instance-matching",
            {"display_cabinet": 0.85},
            support_type="shelf",
            capacity_class="small",
        ),
    ]
    contracts = [
        {
            "anchor": "display_cabinet",
            "support_type": "shelf",
            "capacity_class": "small",
        }
    ]

    result = assign_automatic_anchors(
        ["display_cabinet"],
        candidates,
        _score_only_config(),
        anchor_contracts=contracts,
    )

    assert result.gate_passed
    assert result.assignments[0].candidate_id == "lower-and-matching"
    # The selected hard fields remain the model hypothesis; the contract is a
    # gate and never supplies replacement values.
    assert result.assignments[0].support_type.value == "shelf"
    assert result.assignments[0].capacity_class.value == "small"
    by_id = {edge.candidate_id: edge for edge in result.edges}
    assert by_id["higher-but-wrong"].eligible is False
    assert by_id["higher-but-wrong"].rejection_reasons == [
        "capacity_class_contract_mismatch"
    ]
    assert by_id["higher-but-wrong"].required_support_type.value == "shelf"
    assert by_id["higher-but-wrong"].required_capacity_class.value == "small"
    assert result.expected_anchor_contracts[0].anchor == "display_cabinet"


def test_contract_mismatch_fails_closed_and_contract_is_hashed():
    candidate = _candidate(
        "candidate-a",
        "instance-a",
        {"anchor_a": 0.90},
        support_type="surface",
        capacity_class="medium",
    )
    matching = [
        {
            "anchor": "anchor_a",
            "support_type": "surface",
            "capacity_class": "medium",
        }
    ]
    mismatching = [
        {
            "anchor": "anchor_a",
            "support_type": "shelf",
            "capacity_class": "small",
        }
    ]

    passed = assign_automatic_anchors(
        ["anchor_a"],
        [candidate],
        _score_only_config(),
        anchor_contracts=matching,
    )
    failed = assign_automatic_anchors(
        ["anchor_a"],
        [candidate],
        _score_only_config(),
        anchor_contracts=mismatching,
    )

    assert passed.gate_passed
    assert not failed.gate_passed
    assert failed.assignments == []
    assert failed.unassigned_anchor_labels == ["anchor_a"]
    assert failed.gate_reasons == [
        "complete_one_to_one_assignment_not_found:anchor_a"
    ]
    assert failed.edges[0].rejection_reasons == [
        "capacity_class_contract_mismatch",
        "support_type_contract_mismatch",
    ]
    assert passed.input_hash != failed.input_hash
    assert passed.normalized_hash != failed.normalized_hash


def test_contract_requires_exact_unique_expected_anchor_coverage():
    candidates = [
        _candidate("candidate-a", "instance-a", {"anchor_a": 0.90}),
        _candidate("candidate-b", "instance-b", {"anchor_b": 0.90}),
    ]
    contracts = [
        {
            "anchor": "anchor_b",
            "support_type": "surface",
            "capacity_class": "medium",
        },
        {
            "anchor": "anchor_a",
            "support_type": "surface",
            "capacity_class": "medium",
        },
    ]
    first = assign_automatic_anchors(
        ["anchor_a", "anchor_b"],
        candidates,
        _score_only_config(),
        anchor_contracts=contracts,
    )
    reordered = assign_automatic_anchors(
        ["anchor_b", "anchor_a"],
        list(reversed(candidates)),
        _score_only_config(),
        anchor_contracts=list(reversed(contracts)),
    )
    assert first.input_hash == reordered.input_hash
    assert first.normalized_hash == reordered.normalized_hash

    with pytest.raises(ValueError, match="must exactly cover"):
        assign_automatic_anchors(
            ["anchor_a", "anchor_b"],
            candidates,
            _score_only_config(),
            anchor_contracts=contracts[:1],
        )
    with pytest.raises(ValueError, match="duplicate"):
        ExpectedAnchorContractManifest.model_validate(
            {"anchors": [contracts[0], {**contracts[0], "anchor": "ANCHOR B"}]}
        )


def test_zero_vote_hypothesis_is_valid_diagnostic_but_ineligible():
    candidate = _candidate("candidate-a", "instance-a", {"anchor_a": 0.0})
    candidate["anchor_hypotheses"][0]["label_vote_count"] = 0
    candidate["semantic_observation_count"] = 3

    result = assign_automatic_anchors(
        ["anchor_a"],
        [candidate],
        _score_only_config(),
    )

    assert not result.gate_passed
    assert result.edges[0].rejection_reasons == [
        "label_vote_count_below_threshold"
    ]


def test_semantic_vote_share_uses_view_count_while_support_uses_raw_count():
    candidate = _candidate("candidate-a", "instance-a", {"anchor_a": 0.90})
    candidate["observation_count"] = 30
    candidate["semantic_observation_count"] = 3
    candidate["anchor_hypotheses"][0]["label_vote_count"] = 2
    config = AnchorAssignmentConfig(
        min_candidate_observations=2,
        min_label_vote_count=2,
        min_label_vote_share=0.60,
        min_mean_confidence=0.50,
        min_assignment_score=0.0,
        min_runner_up_margin=0.0,
        support_saturation_observations=5,
    )

    result = assign_automatic_anchors(["anchor_a"], [candidate], config)

    assert result.gate_passed
    components = result.assignments[0].score_components
    assert components.label_vote_share == 0.66666667
    assert components.observation_support == 1.0


def test_vlm_hard_fields_require_independent_confidence():
    candidate = _candidate(
        "candidate-a",
        "instance-a",
        {"anchor_a": 0.90},
        support_confidence=0.40,
        capacity_confidence=None,
    )

    result = assign_automatic_anchors(
        ["anchor_a"],
        [candidate],
        _score_only_config(),
    )

    assert not result.gate_passed
    assert result.assignments == []
    assert result.edges[0].rejection_reasons == [
        "capacity_confidence_missing_or_below_threshold",
        "support_confidence_missing_or_below_threshold",
    ]


def test_vote_share_and_confidence_thresholds_are_edge_diagnostics():
    candidate = _candidate("candidate-a", "instance-a", {"anchor_a": 0.90})
    candidate["anchor_hypotheses"][0].update(
        label_vote_count=2,
        mean_confidence=0.40,
        max_confidence=0.90,
    )
    config = AnchorAssignmentConfig(
        min_candidate_observations=2,
        min_label_vote_count=3,
        min_label_vote_share=0.50,
        min_mean_confidence=0.60,
    )

    result = assign_automatic_anchors(["anchor_a"], [candidate], config)

    assert not result.gate_passed
    assert result.edges[0].eligible is False
    assert result.edges[0].rejection_reasons == [
        "label_vote_count_below_threshold",
        "label_vote_share_below_threshold",
        "mean_confidence_below_threshold",
    ]


def test_contract_rejects_duplicate_candidates_hypotheses_and_expected_anchors():
    candidate = _candidate("candidate-a", "instance-a", {"anchor_a": 0.90})
    with pytest.raises(ValueError, match="candidate_id duplicates"):
        assign_automatic_anchors(
            ["anchor_a"],
            [candidate, copy.deepcopy(candidate)],
            _score_only_config(),
        )

    duplicate_hypothesis = copy.deepcopy(candidate)
    duplicate_hypothesis["anchor_hypotheses"].append(
        copy.deepcopy(duplicate_hypothesis["anchor_hypotheses"][0])
    )
    with pytest.raises(ValidationError, match="duplicate anchors"):
        AutomaticAnchorCandidate.model_validate(duplicate_hypothesis)

    with pytest.raises(ValueError, match="expected_anchor_labels contains duplicates"):
        assign_automatic_anchors(
            ["Anchor A", "anchor-a"],
            [candidate],
            _score_only_config(),
        )


def test_score_weight_contract_rejects_non_unit_sum():
    with pytest.raises(ValidationError, match="weights must sum to 1.0"):
        AnchorAssignmentConfig(mean_confidence_weight=0.50)

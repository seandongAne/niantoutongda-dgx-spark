import pytest
from pydantic import ValidationError

from backend.schemas.core import (
    Assignment,
    AuditEvent,
    ClarificationRequest,
    IdentityState,
    LifeGroup,
    ObjectEntity,
    Observation,
    PlacementPlan,
    Region,
)


def test_observation_roundtrip():
    ob = Observation(
        observation_id="ob_1",
        video_id="v1",
        timestamp_ms=1234,
        bbox=(0.1, 0.2, 0.5, 0.6),
        crop_ref="evidence/v1_ob1.jpg",
        quality=0.9,
        model_version="gdino-base@ms",
    )
    assert Observation.model_validate_json(ob.model_dump_json()) == ob


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ObjectEntity(
            entity_id="e1",
            tracklet_ids=["t1"],
            label="lamp",
            identity_state=IdentityState.MATCHED,
            confidence=0.9,
            evidence_refs=["x.jpg"],
            bogus_field=1,
        )


def test_unknown_schema_version_rejected():
    ev = AuditEvent(
        event_id="ev1",
        event_type="EntityResolved",
        actor="object_memory",
        config_version="reid-v0",
        created_at="2026-07-15T12:00:00Z",
    )
    raw = ev.model_dump()
    raw["schema_version"] = "9.9"
    with pytest.raises(ValidationError):
        AuditEvent.model_validate(raw)


def test_clarification_decision_enum():
    req = ClarificationRequest(
        request_id="c1", candidate_a="t1", candidate_b="t2", reason_codes=["low_margin"]
    )
    assert req.decision is None
    with pytest.raises(ValidationError):
        ClarificationRequest(
            request_id="c2", candidate_a="t1", candidate_b="t2",
            reason_codes=[], decision="maybe",
        )


def test_placement_plan_status_literal():
    plan = PlacementPlan(
        plan_id="p1",
        assignments=[Assignment(group_id="g1", region_id="r1")],
        hard_constraints=["power", "capacity"],
        solver_status="PLAN_READY",
    )
    assert plan.solver_status == "PLAN_READY"
    with pytest.raises(ValidationError):
        PlacementPlan(
            plan_id="p2", assignments=[], hard_constraints=[], solver_status="MAGIC",
        )


def test_lifegroup_and_region():
    g = LifeGroup(group_id="g1", entity_ids=["e1", "e2"], source="auto", evidence_refs=["a.jpg"])
    r = Region(
        region_id="r1", anchor="bed", support_type="surface",
        capacity_class="small", evidence_refs=["r.jpg"],
    )
    assert g.source == "auto" and r.support_type.value == "surface"

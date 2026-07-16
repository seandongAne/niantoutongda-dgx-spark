import pytest
from pydantic import ValidationError

from backend.schemas.hero_bundle import (
    EvidenceSource,
    GroupEvidence,
    HeroGroup,
    RegionEntry,
    RegionManifest,
)


def _entry(region_id: str = "r1", **kw) -> dict:
    base = dict(
        region_id=region_id,
        anchor="bed",
        display_name_zh="床头柜",
        support_type="surface",
        capacity_class="small",
        evidence_refs=["new_1.mp4@00:37"],
    )
    base.update(kw)
    return base


def test_region_entry_requires_evidence():
    with pytest.raises(ValidationError):
        RegionEntry(**_entry(evidence_refs=[]))


def test_region_manifest_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        RegionManifest(video_id="new_1", entries=[_entry("r1"), _entry("r1")])


def test_region_entry_to_core_region_near_power_attribute():
    core = RegionEntry(**_entry(near_power=True)).to_core_region()
    assert core.attributes == {"near_power": "true"}
    assert core.evidence_refs == ["new_1.mp4@00:37"]


def test_cooccurrence_cannot_dominate_group():
    with pytest.raises(ValidationError):
        HeroGroup(
            group_id="g01",
            name_zh="睡前组合",
            entity_ids=["e1"],
            dominant_source=EvidenceSource.COOCCURRENCE,
            member_evidence=[],
        )


def test_to_life_group_source_mapping():
    group = HeroGroup(
        group_id="g01",
        name_zh="睡前组合",
        entity_ids=["e2", "e1"],
        dominant_source=EvidenceSource.NARRATION,
        member_evidence=[
            GroupEvidence(entity_id="e1", source=EvidenceSource.NARRATION, detail="旁白")
        ],
    )
    life = group.to_life_group()
    assert life.source == "user"
    assert life.entity_ids == ["e1", "e2"]
    assert life.evidence_refs == ["narration:e1:旁白"]

"""AcceptanceManifest 的显式任务卡验收范围合同。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.schemas.hero_bundle import (
    AcceptanceAdjudication,
    AcceptanceManifest,
    AcceptancePhoto,
)


def _photo() -> AcceptancePhoto:
    return AcceptancePhoto(photo_ref="acceptance.jpg", region_id="desk")


def test_omitted_selected_card_ids_preserves_legacy_all_cards_scope():
    manifest = AcceptanceManifest(photos=[_photo()])

    assert manifest.selected_card_ids is None
    assert manifest.includes_card("card-01")
    assert manifest.includes_card("card-02")
    assert "selected_card_ids" in manifest.model_dump()


def test_explicit_scope_only_includes_selected_cards():
    manifest = AcceptanceManifest(
        photos=[_photo()], selected_card_ids=["card-01", "card-03"]
    )

    assert manifest.includes_card("card-01")
    assert not manifest.includes_card("card-02")
    assert manifest.includes_card("card-03")


@pytest.mark.parametrize(
    "selected_card_ids",
    [[], ["card-01", "card-01"], [""], ["   "]],
)
def test_explicit_scope_rejects_empty_duplicate_or_blank_ids(selected_card_ids):
    with pytest.raises(ValidationError):
        AcceptanceManifest(photos=[_photo()], selected_card_ids=selected_card_ids)


def test_explicit_scope_rejects_adjudication_for_unselected_card():
    with pytest.raises(ValidationError, match="不在 selected_card_ids"):
        AcceptanceManifest(
            photos=[_photo()],
            selected_card_ids=["card-01"],
            adjudications=[
                AcceptanceAdjudication(
                    card_id="card-02", decision="reject_redo"
                )
            ],
        )


def test_explicit_scope_round_trips_through_json():
    manifest = AcceptanceManifest(
        photos=[_photo()], selected_card_ids=["card-01"]
    )

    restored = AcceptanceManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest
    assert restored.includes_card("card-01")
    assert not restored.includes_card("card-02")

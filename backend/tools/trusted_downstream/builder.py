"""可信库存 → closure 冻结组合/独立装箱项/箱单。

本模块不做视觉推断，也不根据展示名猜组。canonical→entity 的唯一来源是
20 条 data-owner-confirmed 且 downstream eligible 的库存投影；分组成员的
唯一来源是 technical closure。任一数量、集合、状态或投影不一致都显式失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from backend.schemas.hero_bundle import EvidenceSource, GroupEvidence, HeroGroup


PLACEMENT_HINTS: dict[str, str] = {
    "study_stationery": "书桌",
    "cups_and_drinks": "展示柜",
    "toiletries_and_care": "梳妆台",
    "technical_toys_pack": "墙上搁板",
    "technical_snacks_pack": "斗柜",
}

TECHNICAL_PACK_UNITS: tuple[dict[str, Any], ...] = (
    {
        "group_id": "technical_toys_pack",
        "name_zh": "玩具技术装箱单元",
        "canonical_item_ids": ("plush_toy", "rubiks_cube"),
    },
    {
        "group_id": "technical_snacks_pack",
        "name_zh": "零食技术装箱单元",
        "canonical_item_ids": ("popping_candy", "hawthorn_sticks", "biscuits"),
    },
)


@dataclass(frozen=True)
class TrustedItem:
    canonical_id: str
    entity_id: str
    display_name_zh: str
    display_name_source: str
    quantity: int

    def as_box_item(self) -> dict[str, str | int]:
        return {
            "canonical_id": self.canonical_id,
            "entity_id": self.entity_id,
            "display_name_zh": self.display_name_zh,
            "display_name_source": self.display_name_source,
            "quantity": self.quantity,
        }


@dataclass(frozen=True)
class TrustedDownstreamBuild:
    closure_id: str
    groups: tuple[HeroGroup, ...]
    placement_groups: tuple[HeroGroup, ...]
    independent_items: tuple[dict[str, Any], ...]
    boxlist: dict[str, Any]
    trusted_items: tuple[TrustedItem, ...]


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _normalized_status(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _validate_trusted_status(record: dict[str, Any], canonical_id: str) -> None:
    if record.get("downstream_eligible") is not True:
        raise ValueError(f"inventory item {canonical_id} is not downstream eligible")

    status = _normalized_status(record.get("status"))
    if status == "data-owner-confirmed":
        return
    if status == "trusted":
        evidence = record.get("evidence")
        anchor_status = (
            _normalized_status(evidence.get("anchor_review_status"))
            if isinstance(evidence, dict)
            else ""
        )
        if anchor_status == "data-owner-confirmed":
            return
    raise ValueError(
        f"inventory item {canonical_id} is not data-owner-confirmed: "
        f"status={record.get('status')!r}"
    )


def _projected_entity_id(record: dict[str, Any], canonical_id: str) -> str:
    projected = record.get("projected_entity_id")
    nested = record.get("entity")
    nested_id = nested.get("entity_id") if isinstance(nested, dict) else None
    if projected is not None and nested_id is not None and projected != nested_id:
        raise ValueError(
            f"projected entity mismatch for {canonical_id}: {projected!r} != {nested_id!r}"
        )
    return _string(
        projected if projected is not None else nested_id,
        f"inventory[{canonical_id}].projected_entity_id/entity.entity_id",
    )


def _closure_partition(
    closure: object,
) -> tuple[str, list[str], list[dict[str, Any]], list[str]]:
    if not isinstance(closure, dict):
        raise ValueError("closure root must be an object")
    closure_id = _string(closure.get("closure_id"), "closure_id")
    canonical_order = closure.get("canonical_item_ids")
    groups = closure.get("life_groups")
    independent = closure.get("independent_pack_item_ids")
    if not isinstance(canonical_order, list) or len(canonical_order) != 20:
        raise ValueError("closure must contain exactly 20 canonical_item_ids")
    canonical_order = [
        _string(value, "closure.canonical_item_ids[]") for value in canonical_order
    ]
    if len(set(canonical_order)) != 20:
        raise ValueError("closure canonical_item_ids must be unique")
    if not isinstance(groups, list) or len(groups) != 3:
        raise ValueError("closure must contain exactly three life_groups")
    if not isinstance(independent, list) or len(independent) != 5:
        raise ValueError("closure must contain exactly five independent pack items")
    independent = [
        _string(value, "closure.independent_pack_item_ids[]") for value in independent
    ]

    normalized_groups: list[dict[str, Any]] = []
    partition: list[str] = []
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("closure life_group must be an object")
        group_id = _string(group.get("group_id"), "life_group.group_id")
        if group_id in group_ids:
            raise ValueError(f"duplicate closure group_id: {group_id}")
        group_ids.add(group_id)
        name_zh = _string(group.get("name_zh"), f"life_group[{group_id}].name_zh")
        members = group.get("canonical_item_ids")
        if not isinstance(members, list) or not members:
            raise ValueError(f"life_group {group_id} has no canonical members")
        members = [
            _string(value, f"life_group[{group_id}].canonical_item_ids[]")
            for value in members
        ]
        normalized_groups.append(
            {"group_id": group_id, "name_zh": name_zh, "canonical_item_ids": members}
        )
        partition.extend(members)
    partition.extend(independent)
    if len(partition) != 20 or len(set(partition)) != 20:
        raise ValueError("closure groups and independent items must partition 20 items once")
    if set(partition) != set(canonical_order):
        raise ValueError("closure item partition differs from canonical_item_ids")
    return closure_id, canonical_order, normalized_groups, independent


def _display_index(display_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(display_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"display row {row_number} must be an object")
        entity_id = row.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            # 全量展示表可以包含尚未命名的辅助行；它们不能提供可信名字。
            continue
        if entity_id in index:
            raise ValueError(f"duplicate display entity_id: {entity_id}")
        index[entity_id] = row
    return index


def _trusted_inventory(
    inventory_rows: list[dict[str, Any]],
    canonical_order: list[str],
    display_rows: list[dict[str, Any]],
) -> dict[str, TrustedItem]:
    if len(inventory_rows) != 20:
        raise ValueError(f"trusted inventory must contain exactly 20 rows, got {len(inventory_rows)}")
    display = _display_index(display_rows)
    by_canonical: dict[str, TrustedItem] = {}
    entity_ids: set[str] = set()
    for row_number, record in enumerate(inventory_rows, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"inventory row {row_number} must be an object")
        canonical = record.get("canonical")
        if not isinstance(canonical, dict):
            raise ValueError(f"inventory row {row_number} missing canonical object")
        canonical_id = _string(
            canonical.get("canonical_id"),
            f"inventory row {row_number} canonical.canonical_id",
        )
        if canonical_id in by_canonical:
            raise ValueError(f"duplicate inventory canonical_id: {canonical_id}")
        _validate_trusted_status(record, canonical_id)
        entity_id = _projected_entity_id(record, canonical_id)
        if entity_id in entity_ids:
            raise ValueError(f"duplicate projected entity_id: {entity_id}")
        entity_ids.add(entity_id)

        quantity = record.get("quantity", 1)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError(f"invalid quantity for {canonical_id}: {quantity!r}")

        display_name = ""
        display_source = ""
        display_row = display.get(entity_id)
        if display_row is not None:
            candidate = display_row.get("display_name_zh")
            if isinstance(candidate, str) and candidate.strip():
                display_name = candidate.strip()
                display_source = "display"
        if not display_name:
            candidate = canonical.get("name_zh")
            if isinstance(candidate, str) and candidate.strip():
                display_name = candidate.strip()
                display_source = "inventory.canonical"
        if not display_name:
            raise ValueError(f"no display name in inventory/display for {canonical_id}")

        by_canonical[canonical_id] = TrustedItem(
            canonical_id=canonical_id,
            entity_id=entity_id,
            display_name_zh=display_name,
            display_name_source=display_source,
            quantity=quantity,
        )

    expected = set(canonical_order)
    actual = set(by_canonical)
    if actual != expected:
        raise ValueError(
            "inventory canonical set differs from closure: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return by_canonical


def build_trusted_downstream(
    closure: object,
    inventory_rows: list[dict[str, Any]],
    display_rows: list[dict[str, Any]],
) -> TrustedDownstreamBuild:
    closure_id, canonical_order, closure_groups, independent_ids = _closure_partition(
        closure
    )
    trusted = _trusted_inventory(inventory_rows, canonical_order, display_rows)

    technical_members = [
        canonical_id
        for unit in TECHNICAL_PACK_UNITS
        for canonical_id in unit["canonical_item_ids"]
    ]
    if technical_members != independent_ids:
        raise ValueError(
            "closure independent item order differs from technical packing policy: "
            f"closure={independent_ids}, policy={technical_members}"
        )

    groups: list[HeroGroup] = []
    placement_groups: list[HeroGroup] = []
    boxes: list[dict[str, Any]] = []
    for index, frozen in enumerate(closure_groups, start=1):
        member_ids = frozen["canonical_item_ids"]
        members = [trusted[canonical_id] for canonical_id in member_ids]
        group = HeroGroup(
            group_id=frozen["group_id"],
            name_zh=frozen["name_zh"],
            entity_ids=[item.entity_id for item in members],
            dominant_source=EvidenceSource.CONFIRMATION,
            member_evidence=[
                GroupEvidence(
                    entity_id=item.entity_id,
                    source=EvidenceSource.CONFIRMATION,
                    detail=f"技术 closure 冻结成员:{item.canonical_id}",
                    refs=[
                        f"closure:{closure_id}:{frozen['group_id']}:{item.canonical_id}",
                        f"trusted-inventory:{item.canonical_id}:{item.entity_id}",
                    ],
                )
                for item in members
            ],
        )
        groups.append(group)
        placement_groups.append(
            group.model_copy(
                update={"target_region_hint": PLACEMENT_HINTS[group.group_id]}
            )
        )
        boxes.append(
            {
                "box_id": f"box-group-{index:02d}",
                "box_type": "life_group",
                "group_id": group.group_id,
                "box_label_zh": f"{group.name_zh}箱",
                "items": [item.as_box_item() for item in members],
            }
        )

    unit_by_canonical = {
        canonical_id: unit["group_id"]
        for unit in TECHNICAL_PACK_UNITS
        for canonical_id in unit["canonical_item_ids"]
    }
    independent_items: list[dict[str, Any]] = []
    for canonical_id in independent_ids:
        item = trusted[canonical_id]
        row = {
            "schema_version": "1.0",
            **item.as_box_item(),
            "packing_kind": "independent",
            "is_life_group": False,
            "group_id": None,
            "placement_unit_id": unit_by_canonical[canonical_id],
            "reason_code": "CLOSURE_INDEPENDENT_PACK_ITEM",
        }
        independent_items.append(row)

    for index, frozen in enumerate(TECHNICAL_PACK_UNITS, start=1):
        members = [trusted[canonical_id] for canonical_id in frozen["canonical_item_ids"]]
        placement_group = HeroGroup(
            group_id=frozen["group_id"],
            name_zh=frozen["name_zh"],
            entity_ids=[item.entity_id for item in members],
            dominant_source=EvidenceSource.TEMPLATE,
            member_evidence=[
                GroupEvidence(
                    entity_id=item.entity_id,
                    source=EvidenceSource.TEMPLATE,
                    detail=f"技术装箱策略 v1:{item.canonical_id}",
                    refs=[
                        f"closure:{closure_id}:independent:{item.canonical_id}",
                        f"trusted-inventory:{item.canonical_id}:{item.entity_id}",
                    ],
                )
                for item in members
            ],
            target_region_hint=PLACEMENT_HINTS[frozen["group_id"]],
        )
        placement_groups.append(placement_group)
        boxes.append(
            {
                "box_id": f"box-technical-{index:02d}",
                "box_type": "technical_pack_unit",
                "group_id": placement_group.group_id,
                "box_label_zh": f"{placement_group.name_zh}箱",
                "items": [item.as_box_item() for item in members],
            }
        )

    trusted_items = tuple(trusted[canonical_id] for canonical_id in canonical_order)
    boxlist = {
        "schema_version": "1.0",
        "closure_id": closure_id,
        "canonical_item_order": canonical_order,
        "canonical_item_count": len(trusted_items),
        "box_count": len(boxes),
        "boxes": boxes,
    }
    return TrustedDownstreamBuild(
        closure_id=closure_id,
        groups=tuple(groups),
        placement_groups=tuple(placement_groups),
        independent_items=tuple(independent_items),
        boxlist=boxlist,
        trusted_items=trusted_items,
    )

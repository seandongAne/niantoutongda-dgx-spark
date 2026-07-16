"""结构化任务卡 — 布局结果 → 每组一张"装箱/摆放/验收"卡。

卡片是确定性拼装:组 + 区域 + 布局指派 + 替代区域,不引入任何模型判断。
验收清单与 VerificationCheckRequest 的口径一致(物品出现 + 区域正确),
方便 EXEC 直接对着清单发验收复核。
"""

from __future__ import annotations

from backend.schemas.hero_bundle import (
    HeroGroup,
    RegionManifest,
    TaskCard,
    TaskCardItem,
)
from backend.tools.solver.layout_solver import LayoutResult


def build_task_cards(
    groups: list[HeroGroup],
    layout: LayoutResult,
    manifest: RegionManifest,
    entity_display: dict[str, dict[str, str]],
) -> list[TaskCard]:
    if layout.status != "PLAN_READY":
        raise ValueError(f"布局未就绪({layout.status}),不生成任务卡")
    region_names = {e.region_id: e.display_name_zh for e in manifest.entries}
    cards: list[TaskCard] = []
    for idx, group in enumerate(sorted(groups, key=lambda g: g.group_id), start=1):
        region_id = layout.assignments.get(group.group_id)
        if region_id is None:
            raise ValueError(f"布局结果缺少组 {group.group_id} 的指派")
        items = [
            TaskCardItem(
                entity_id=eid,
                display_name_zh=entity_display.get(eid, {}).get(
                    "display_name_zh", eid
                ),
                hero_crop_ref=entity_display.get(eid, {}).get("hero_crop_ref", ""),
            )
            for eid in sorted(group.entity_ids)
        ]
        region_name = region_names.get(region_id, region_id)
        cards.append(
            TaskCard(
                card_id=f"card-{idx:02d}",
                group_id=group.group_id,
                box_label_zh=f"{group.name_zh}箱",
                items=items,
                target_region_id=region_id,
                target_region_name_zh=region_name,
                alternative_region_id=layout.alternatives.get(group.group_id),
                placement_notes=(
                    [f"旁白去向提示:{group.target_region_hint}"]
                    if group.target_region_hint
                    else []
                ),
                verification_checklist=[
                    f"{it.display_name_zh} 出现在「{region_name}」" for it in items
                ],
            )
        )
    return cards


def task_card_markdown(card: TaskCard) -> str:
    lines = [
        f"## {card.card_id} · {card.box_label_zh}",
        "",
        f"- 目标区域:**{card.target_region_name_zh}**(`{card.target_region_id}`)",
    ]
    if card.alternative_region_id:
        lines.append(f"- 备选区域:`{card.alternative_region_id}`")
    for note in card.placement_notes:
        lines.append(f"- {note}")
    lines += ["", "| 物品 | 实体 |", "|---|---|"]
    lines += [f"| {it.display_name_zh} | `{it.entity_id}` |" for it in card.items]
    lines += ["", "验收清单:"]
    lines += [f"- [ ] {check}" for check in card.verification_checklist]
    return "\n".join(lines) + "\n"

"""GROUP 生活组合构建 — 确定性规则,按证据优先级裁决。

优先级(2026-07-16 用户裁决,EVIDENCE_PRIORITY 数值越小越强):
  1. 旁白明确分组:解析成功的 "和X一组" 边 → 并查集连通分量成组;
  2. 用户轻确认:只补旁白没覆盖的实体;与旁白冲突时旁白胜出并记录冲突;
  3. 模板语义:实体展示名命中模板关键词 → 归入模板指定组;
  4. 画面共现:只给已成组的成员追加佐证,永远不产生归属。
仍未归组的实体 → GroupClarification(轻确认队列)。全程排序保证确定性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from backend.schemas.core import AuditEvent
from backend.schemas.hero_bundle import (
    EvidenceSource,
    GroupClarification,
    GroupConfirmation,
    GroupEvidence,
    HeroGroup,
    NarrationItem,
    NarrationResolution,
)
from backend.tools.grouping.narration import match_name_to_entities

# 组命名规则:成员旁白 target_location 命中关键词 → 组名。按插入序最长优先无必要,
# 关键词都很短且互斥;新场景可通过 build_groups 参数覆盖。
DEFAULT_GROUP_NAME_RULES: dict[str, str] = {
    "床头": "睡前组合",
    "床": "睡前组合",
    "书桌": "学习组合",
    "桌": "学习组合",
    "柜": "收纳组合",
    "架": "收纳组合",
    "角落": "收纳组合",
    "充电": "收纳组合",
}

COOCCURRENCE_MIN_COUNT = 2


@dataclass
class GroupBuild:
    groups: list[HeroGroup] = field(default_factory=list)
    unassigned_entity_ids: list[str] = field(default_factory=list)
    clarifications: list[GroupClarification] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 固定小根做代表,保证确定性
            self.parent[max(ra, rb)] = min(ra, rb)


def _pick_group_name(
    target_locations: list[str], rules: dict[str, str], fallback: str
) -> tuple[str, str]:
    """返回 (组名, 命中提示)。取成员 target_location 中最先命中的规则。"""
    for loc in sorted(target_locations):
        for keyword in sorted(rules, key=len, reverse=True):
            if keyword in loc:
                return rules[keyword], loc
    return fallback, target_locations[0] if target_locations else ""


def build_groups(
    entity_display: dict[str, dict[str, str]],
    narration_items: list[NarrationItem],
    resolutions: list[NarrationResolution],
    confirmations: list[GroupConfirmation] = (),
    template_rules: dict[str, str] | None = None,
    cooccurrence: dict[tuple[str, str], int] | None = None,
    *,
    group_name_rules: dict[str, str] | None = None,
    cooccurrence_min_count: int = COOCCURRENCE_MIN_COUNT,
    config_version: str = "group-v1",
    created_at: str | None = None,
) -> GroupBuild:
    build = GroupBuild()
    name_rules = group_name_rules or DEFAULT_GROUP_NAME_RULES
    template_rules = template_rules or {}
    cooccurrence = cooccurrence or {}
    items_by_id = {i.item_id: i for i in narration_items}
    resolved: dict[str, str] = {}  # item_id -> entity_id
    entity_item: dict[str, str] = {}  # entity_id -> item_id(取排序最先)

    # ---- 第 1 层:旁白 ----
    for res in sorted(resolutions, key=lambda r: r.item_id):
        if res.entity_id is None:
            item = items_by_id.get(res.item_id)
            build.clarifications.append(
                GroupClarification(
                    entity_id="",
                    question_zh=(
                        f"旁白「{item.raw_text if item else res.item_id}」"
                        f"对应哪件物品?(候选 {len(res.candidate_entity_ids)} 个)"
                    ),
                    reason="unresolved_narration",
                    candidate_group_ids=[],
                )
            )
            continue
        resolved[res.item_id] = res.entity_id
        entity_item.setdefault(res.entity_id, res.item_id)

    uf = _UnionFind()
    partner_edges: dict[str, list[str]] = {}
    for item_id, entity_id in sorted(resolved.items()):
        uf.find(entity_id)
        item = items_by_id[item_id]
        for partner_name in item.group_partners:
            candidates, method = match_name_to_entities(
                partner_name, [], entity_display
            )
            if method in ("name_unique", "name_color"):
                uf.union(entity_id, candidates[0])
                partner_edges.setdefault(entity_id, []).append(candidates[0])
            else:
                build.conflicts.append(
                    f"旁白伙伴名「{partner_name}」({item_id})未能唯一解析,忽略该边"
                )

    components: dict[str, list[str]] = {}
    for entity_id in sorted(uf.parent):
        components.setdefault(uf.find(entity_id), []).append(entity_id)

    assigned: dict[str, str] = {}  # entity_id -> group_id
    groups: dict[str, HeroGroup] = {}
    for idx, root in enumerate(sorted(components), start=1):
        members = sorted(components[root])
        group_id = f"g{idx:02d}"
        targets = [
            items_by_id[entity_item[m]].target_location
            for m in members
            if m in entity_item and items_by_id[entity_item[m]].target_location
        ]
        name_zh, hint = _pick_group_name(targets, name_rules, f"组合{idx}")
        evidence = [
            GroupEvidence(
                entity_id=m,
                source=EvidenceSource.NARRATION,
                detail=items_by_id[entity_item[m]].raw_text if m in entity_item else "旁白伙伴边",
                refs=[entity_item[m]] if m in entity_item else [],
            )
            for m in members
        ]
        groups[group_id] = HeroGroup(
            group_id=group_id,
            name_zh=name_zh,
            entity_ids=members,
            dominant_source=EvidenceSource.NARRATION,
            member_evidence=evidence,
            target_region_hint=hint,
        )
        for m in members:
            assigned[m] = group_id

    # ---- 第 2 层:用户轻确认(只补空白,冲突记录、旁白胜出) ----
    name_to_gid = {g.name_zh: gid for gid, g in sorted(groups.items())}
    for conf in sorted(confirmations, key=lambda c: c.entity_id):
        if conf.entity_id in assigned:
            current = groups[assigned[conf.entity_id]]
            if conf.decision == "remove" or conf.group_name_zh != current.name_zh:
                build.conflicts.append(
                    f"轻确认({conf.entity_id}→{conf.group_name_zh}/{conf.decision})"
                    f"与旁白归属({current.name_zh})冲突,按优先级保留旁白"
                )
            continue
        if conf.decision != "assign":
            continue
        gid = name_to_gid.get(conf.group_name_zh)
        if gid is None:
            gid = f"g{len(groups) + 1:02d}"
            groups[gid] = HeroGroup(
                group_id=gid,
                name_zh=conf.group_name_zh,
                entity_ids=[],
                dominant_source=EvidenceSource.CONFIRMATION,
                member_evidence=[],
            )
            name_to_gid[conf.group_name_zh] = gid
        groups[gid].entity_ids.append(conf.entity_id)
        groups[gid].entity_ids.sort()
        groups[gid].member_evidence.append(
            GroupEvidence(
                entity_id=conf.entity_id,
                source=EvidenceSource.CONFIRMATION,
                detail=conf.note or "用户轻确认",
            )
        )
        assigned[conf.entity_id] = gid

    # ---- 第 3 层:模板语义(实体展示名命中关键词) ----
    for entity_id in sorted(entity_display):
        if entity_id in assigned:
            continue
        display = entity_display[entity_id].get("display_name_zh", "")
        for keyword in sorted(template_rules, key=len, reverse=True):
            if keyword and keyword in display:
                group_name = template_rules[keyword]
                gid = name_to_gid.get(group_name)
                if gid is None:
                    gid = f"g{len(groups) + 1:02d}"
                    groups[gid] = HeroGroup(
                        group_id=gid,
                        name_zh=group_name,
                        entity_ids=[],
                        dominant_source=EvidenceSource.TEMPLATE,
                        member_evidence=[],
                    )
                    name_to_gid[group_name] = gid
                groups[gid].entity_ids.append(entity_id)
                groups[gid].entity_ids.sort()
                groups[gid].member_evidence.append(
                    GroupEvidence(
                        entity_id=entity_id,
                        source=EvidenceSource.TEMPLATE,
                        detail=f"模板:「{keyword}」→{group_name}",
                    )
                )
                assigned[entity_id] = gid
                break

    # ---- 第 4 层:共现只佐证(给同组成员对追加证据,不产生归属) ----
    for (a, b), count in sorted(cooccurrence.items()):
        if count < cooccurrence_min_count:
            continue
        if assigned.get(a) is not None and assigned.get(a) == assigned.get(b):
            groups[assigned[a]].member_evidence.append(
                GroupEvidence(
                    entity_id=a,
                    source=EvidenceSource.COOCCURRENCE,
                    detail=f"与 {b} 画面共现 {count} 次(佐证)",
                )
            )

    # ---- 收尾:未归组实体 → 轻确认队列 ----
    for entity_id in sorted(entity_display):
        if entity_id not in assigned:
            build.unassigned_entity_ids.append(entity_id)
            build.clarifications.append(
                GroupClarification(
                    entity_id=entity_id,
                    question_zh=(
                        f"「{entity_display[entity_id].get('display_name_zh', entity_id)}」"
                        "归入哪个组合?"
                    ),
                    reason="unassigned_entity",
                    candidate_group_ids=sorted(groups),
                )
            )

    build.groups = [groups[gid] for gid in sorted(groups)]
    ts = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    build.audit_events.append(
        AuditEvent(
            event_id=f"group-build-{ts}",
            event_type="group.build",
            actor="GROUP",
            input_refs=sorted(
                {f"narration:{i.item_id}" for i in narration_items}
                | {f"confirmation:{c.entity_id}" for c in confirmations}
            ),
            output_refs=[f"group:{g.group_id}" for g in build.groups],
            config_version=config_version,
            created_at=ts,
        )
    )
    return build

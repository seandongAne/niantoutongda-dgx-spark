"""旁白 → 结构化条目 → 实体解析。

旁白是 GROUP 的主证据(2026-07-16 裁决):"和台灯一组"才决定分组,
画面共现只佐证。解析失败或同名歧义一律进轻确认队列,不猜。

parse_narration_line 是对人话的启发式切分,只服务
"这是什么、是谁的、现在在哪、搬到新家想放哪、和什么打包在一起"
这一拍摄要求钦定的句式;ASR 稳定后如需更强解析,替换本函数即可,
NarrationItem 契约不变。
"""

from __future__ import annotations

import re

from backend.schemas.hero_bundle import NarrationItem, NarrationResolution

# 中文颜色词 → S5 color_primary 词域(小写英文)。按最长匹配优先。
COLOR_WORDS_ZH: dict[str, str] = {
    "蓝色": "blue", "蓝": "blue",
    "粉色": "pink", "粉红": "pink", "粉": "pink",
    "红色": "red", "红": "red",
    "白色": "white", "白": "white",
    "黑色": "black", "黑": "black",
    "灰色": "gray", "灰": "gray",
    "绿色": "green", "绿": "green",
    "黄色": "yellow", "黄": "yellow",
    "橙色": "orange", "橙": "orange",
    "紫色": "purple", "紫": "purple",
    "棕色": "brown", "棕": "brown", "咖啡色": "brown",
    "米色": "beige", "米白": "beige",
}
_COLOR_KEYS = sorted(COLOR_WORDS_ZH, key=len, reverse=True)

_SPLIT = re.compile(r"[,,。;;.]+")
_PARTNER = re.compile(r"[和跟与](?P<names>.+?)(?:一组|一起|打包)")
_PARTNER_SEP = re.compile(r"[、和跟与]")
_TARGET = re.compile(r"(?:搬(?:过去|到)?|挪到?|去)?放(?:在|到)?(?P<loc>.+)")
_SOURCE = re.compile(r"^(?:现在|原来|之前)?(?:在|放在)(?P<loc>.+)")


def extract_colors(text: str) -> list[str]:
    found: list[str] = []
    rest = text
    for key in _COLOR_KEYS:
        if key in rest:
            color = COLOR_WORDS_ZH[key]
            if color not in found:
                found.append(color)
            rest = rest.replace(key, "")
    return found


def parse_narration_line(item_id: str, raw_text: str) -> NarrationItem:
    """按拍摄要求句式切分一句旁白。例:
    "蓝色水壶,妹妹的,现在在书桌上,搬过去放床头,和台灯一组。"
    """
    segments = [s.strip() for s in _SPLIT.split(raw_text) if s.strip()]
    if not segments:
        raise ValueError(f"旁白为空: {item_id}")
    label = segments[0]
    owner = ""
    source_location = ""
    target_location = ""
    partners: list[str] = []
    for seg in segments[1:]:
        if m := _PARTNER.search(seg):
            partners.extend(
                p.strip() for p in _PARTNER_SEP.split(m.group("names")) if p.strip()
            )
            continue
        if m := _SOURCE.match(seg):
            source_location = m.group("loc").strip()
            continue
        if ("搬" in seg or "挪" in seg or seg.startswith("放") or seg.startswith("去")) and (
            m := _TARGET.search(seg)
        ):
            target_location = m.group("loc").strip()
            continue
        if seg.endswith("的") and 1 < len(seg) <= 6:
            owner = seg[:-1]
            continue
    return NarrationItem(
        item_id=item_id,
        raw_text=raw_text,
        label_zh=label,
        owner=owner,
        source_location=source_location,
        target_location=target_location,
        group_partners=partners,
        color_words=extract_colors(raw_text),
    )


def _base_name(name: str) -> str:
    """剥掉颜色词后的物品基名,用于同款不同色的名称对齐。"""
    base = name
    for key in _COLOR_KEYS:
        base = base.replace(key, "")
    return base.strip()


def match_name_to_entities(
    name: str,
    colors: list[str],
    entity_display: dict[str, dict[str, str]],
) -> tuple[list[str], str]:
    """按名称(双向包含)+颜色消歧匹配实体。

    entity_display: entity_id → {"display_name_zh": ..., "color_primary": ...}
    返回 (候选实体列表, method)。唯一命中才算解析成功。
    """
    base = _base_name(name)
    if not base:
        return [], "unresolved_no_match"
    candidates = []
    for eid in sorted(entity_display):
        disp = _base_name(entity_display[eid].get("display_name_zh", ""))
        if disp and (base in disp or disp in base):
            candidates.append(eid)
    if len(candidates) == 1:
        return candidates, "name_unique"
    if not candidates:
        return [], "unresolved_no_match"
    want = colors or extract_colors(name)
    if want:
        by_color = [
            eid
            for eid in candidates
            if entity_display[eid].get("color_primary", "") in want
        ]
        if len(by_color) == 1:
            return by_color, "name_color"
        if by_color:
            candidates = by_color
    return candidates, "unresolved_ambiguous"


def resolve_narration(
    items: list[NarrationItem],
    entity_display: dict[str, dict[str, str]],
) -> list[NarrationResolution]:
    resolutions: list[NarrationResolution] = []
    for item in sorted(items, key=lambda i: i.item_id):
        candidates, method = match_name_to_entities(
            item.label_zh, item.color_words, entity_display
        )
        resolutions.append(
            NarrationResolution(
                item_id=item.item_id,
                entity_id=candidates[0] if method in ("name_unique", "name_color") else None,
                method=method,
                candidate_entity_ids=candidates,
            )
        )
    return resolutions

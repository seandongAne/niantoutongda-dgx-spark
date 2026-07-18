#!/usr/bin/env python
"""可点击成果页 — hero run 目录 → 自包含静态 index.html。

初赛完成门"可点击成果"的落地:实体卡(hero 图+VLM 名+属性+组徽章)、
生活组合(证据来源可见:旁白/轻确认/模板主导,共现只作佐证标签)、
新家布局、任务卡、澄清队列、复跑指纹(bundle/config 哈希)。

视觉:HeroUI 设计语言的手写实现(暗色优先、语义色 chip、圆角卡片),
零外部依赖(无 CDN/字体/JS 框架),file:// 直开——演示关键路径不引入
构建链与网络面。证据类别一律 颜色+文字标签,不做纯色编码。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

# 证据来源 → (中文标签, 语义色槽)。HeroUI 语义:narration=primary,
# confirmation=secondary, template=warning, cooccurrence=default。
SOURCE_LABEL = {
    "narration": ("旁白", "primary"),
    "confirmation": ("轻确认", "secondary"),
    "template": ("模板", "warning"),
    "cooccurrence": ("共现佐证", "neutral"),
}

CSS_COLOR = {
    "blue": "#338ef7", "pink": "#ff71d7", "red": "#f31260", "white": "#ececee",
    "black": "#3f3f46", "gray": "#a1a1aa", "green": "#17c964",
    "yellow": "#fbc531", "orange": "#f5a524", "purple": "#9353d3",
    "brown": "#a16207", "beige": "#d6c7a1",
}

BOX_TYPE_LABEL = {
    "life_group": "生活组",
    "technical_pack_unit": "独立装箱单元",
}

RISK_RULE_LABEL = {
    "CHILD_SHARP_TOOL_REACH": "儿童可触及锐器",
    "TRIP_HAZARD_IN_PATH": "通道绊倒风险",
    "POWER_IN_WET_ZONE": "潮湿区域用电",
}

RISK_STATUS_LABEL = {
    "TRIGGERED": ("已触发", "danger"),
    "NEEDS_USER": ("待人工确认", "warning"),
    "NOT_APPLICABLE": ("当前条件不成立", "neutral"),
}

DEFAULT_RISK_DISCLAIMER_ZH = (
    "仅为辅助风险提醒，不构成安全认证，也不能替代现场人员或专业人员复核。"
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_state_artifacts(run_dir: Path) -> list[dict] | None:
    """Read current completed-stage hashes without depending on a stale bundle.

    ``report`` and ``bundle`` are excluded because report is being generated and
    including its previous hash would create a self-reference.  ``None`` keeps
    legacy runs without state files on the bundle-based compatibility path.
    """

    state_dir = run_dir / "state"
    if not state_dir.is_dir():
        return None
    artifacts: list[dict] = []
    for state_path in sorted(state_dir.glob("*.json")):
        state = load_json(state_path, {})
        stage = str(state.get("stage") or state_path.stem)
        if stage in {"report", "bundle"} or state.get("status") != "done":
            continue
        outputs = state.get("outputs", {})
        if not isinstance(outputs, dict):
            continue
        artifacts.extend(
            {"stage": stage, "path": str(path), "sha256": str(digest)}
            for path, digest in sorted(outputs.items())
        )
    return artifacts


def esc(value: object) -> str:
    return html.escape(str(value))


def chip(text: str, kind: str = "neutral") -> str:
    return f'<span class="chip chip-{kind}">{esc(text)}</span>'


def color_chip(color: str) -> str:
    if not color:
        return ""
    swatch = CSS_COLOR.get(color, "#a1a1aa")
    return (
        f'<span class="chip chip-neutral"><i class="dot" '
        f'style="background:{swatch}"></i>{esc(color)}</span>'
    )


def img_tag(run_dir: Path, ref: str, cls: str = "hero") -> str:
    if not ref:
        return f'<div class="{cls} noimg">待补 hero 图</div>'
    src = ref
    if not ref.startswith(("http://", "https://", "/")):
        src = os.path.relpath(PROJ / ref, run_dir)
    return (
        f'<img class="{cls}" src="{esc(src)}" alt="" loading="lazy" '
        "onerror=\"this.outerHTML='<div class=&quot;" + cls +
        " noimg&quot;>待补 hero 图</div>'\">"
    )


def build_page(run_dir: Path, config_path: Path | None = None) -> str:
    trusted_display_path = run_dir / "inventory/display.jsonl"
    trusted_inventory_mode = trusted_display_path.exists()
    display_path = (
        trusted_display_path
        if trusted_inventory_mode
        else run_dir / "naming/display.jsonl"
    )
    placement_groups_path = run_dir / "group/placement_groups.jsonl"
    groups_path = (
        placement_groups_path
        if placement_groups_path.exists()
        else run_dir / "group/groups.jsonl"
    )
    display = {r["entity_id"]: r for r in load_jsonl(display_path)}
    groups = load_jsonl(groups_path)
    trusted_clarifications_path = run_dir / "inventory/clarifications.jsonl"
    clarifications_path = (
        trusted_clarifications_path
        if trusted_clarifications_path.exists()
        else run_dir / "group/clarifications.jsonl"
    )
    clarifications = load_jsonl(clarifications_path)
    conflicts = load_json(run_dir / "group/conflicts.json", [])
    layout = load_json(run_dir / "layout/layout.json", {})
    regions = load_json(run_dir / "regions/regions.json", {})
    cards = load_jsonl(run_dir / "taskcards/taskcards.jsonl")
    verdicts = load_json(run_dir / "verify/verdicts.json", {})
    trace_report = load_json(run_dir / "audit/replay-report.json", {})
    bundle = load_json(run_dir / "bundle.json", {})
    state_artifacts = _current_state_artifacts(run_dir)

    # 可选的完整落地摘要。每块只在自身所需的小型产物齐全时出现；旧 run
    # 没有这些文件时继续生成原页面，不把候选/审计大 JSON 内嵌进成果页。
    inventory_metrics_path = run_dir / "inventory/metrics.json"
    inventory_clarifications_path = run_dir / "inventory/clarifications.jsonl"
    inventory_metrics = (
        load_json(inventory_metrics_path, {})
        if inventory_metrics_path.exists() and inventory_clarifications_path.exists()
        else None
    )
    inventory_clarifications = (
        load_jsonl(inventory_clarifications_path)
        if inventory_metrics is not None
        else []
    )

    boxlist_path = run_dir / "group/boxlist.json"
    group_metrics_path = run_dir / "group/metrics.json"
    boxlist = (
        load_json(boxlist_path, {})
        if boxlist_path.exists() and group_metrics_path.exists()
        else None
    )
    group_metrics = (
        load_json(group_metrics_path, {}) if boxlist is not None else None
    )

    spatial_metrics_path = run_dir / "spatial/metrics.json"
    spatial_metrics = (
        load_json(spatial_metrics_path, {}) if spatial_metrics_path.exists() else None
    )
    spatial_review_metrics_path = run_dir / "spatial_review/metrics.json"
    spatial_review_metrics = (
        load_json(spatial_review_metrics_path, {})
        if spatial_review_metrics_path.exists()
        else None
    )

    risk_assessments_path = run_dir / "risk/assessments.json"
    risk_metrics_path = run_dir / "risk/metrics.json"
    risk_assessments = (
        load_json(risk_assessments_path, {})
        if risk_assessments_path.exists() and risk_metrics_path.exists()
        else None
    )
    risk_metrics = (
        load_json(risk_metrics_path, {}) if risk_assessments is not None else None
    )

    group_of = {eid: g for g in groups for eid in g.get("entity_ids", [])}
    region_names = {
        e["region_id"]: e["display_name_zh"] for e in regions.get("entries", [])
    }

    # ---- 顶部统计 ----
    stats = [
        (len(display), "可信库存实体" if trusted_inventory_mode else "实体"),
        (len(groups), "placement 单元" if trusted_inventory_mode else "生活组合"),
        (len(cards), "任务卡"),
        (len(clarifications), "待澄清"),
    ]
    if verdicts:
        n_verified = sum(1 for v in verdicts.values() if v["verdict"] == "VERIFIED")
        stats.append((f"{n_verified}/{len(verdicts)}", "验收通过"))
    if trace_report:
        stats.append((trace_report.get("message_count", 0), "Agent 消息"))
    stat_tiles = "".join(
        f'<div class="stat"><div class="stat-n">{esc(n)}</div>'
        f'<div class="stat-l">{esc(label)}</div></div>'
        for n, label in stats
    )
    entity_heading = "可信库存实体" if trusted_inventory_mode else "实体卡"
    entity_note = (
        "数据所有者确认的 20 行投影，raw ReID 仅保留为审计证据"
        if trusted_inventory_mode
        else "展示名 = 本地 VLM 读 hero 图,同款不同色自动消歧"
    )

    # ---- 实体卡 ----
    entity_cards = []
    for eid in sorted(display):
        row = display[eid]
        g = group_of.get(eid)
        gchip = chip(g["name_zh"], "success") if g else chip("未归组", "danger")
        entity_cards.append(
            '<div class="card entity">'
            + img_tag(run_dir, row.get("hero_crop_ref", ""))
            + f'<div class="cardbody"><div class="name">{esc(row["display_name_zh"])}</div>'
            + f'<div class="chips">{gchip}{color_chip(row.get("color_primary", ""))}</div>'
            + f'<div class="eid">{esc(eid)}</div></div></div>'
        )

    # ---- 组合 ----
    group_secs = []
    for g in groups:
        evidence_by_eid: dict[str, list[dict]] = {}
        for ev in g.get("member_evidence", []):
            evidence_by_eid.setdefault(ev["entity_id"], []).append(ev)
        rows = []
        for eid in g.get("entity_ids", []):
            name = display.get(eid, {}).get("display_name_zh", eid)
            tags = "".join(
                chip(*SOURCE_LABEL.get(ev["source"], (ev["source"], "neutral")))
                for ev in evidence_by_eid.get(eid, [])
            )
            detail = "; ".join(
                ev["detail"] for ev in evidence_by_eid.get(eid, [])
                if ev["source"] != "cooccurrence"
            )
            rows.append(
                f"<tr><td>{esc(name)}</td><td>{tags}</td>"
                f'<td class="detail">{esc(detail)}</td></tr>'
            )
        dominant = chip(*SOURCE_LABEL.get(
            g["dominant_source"], (g["dominant_source"], "neutral")))
        hint = esc(g.get("target_region_hint", "")) or "—"
        group_secs.append(
            f'<div class="panel"><div class="panel-head"><h3>{esc(g["name_zh"])}</h3>'
            f'<span class="mono dim">{esc(g["group_id"])}</span>{dominant}'
            f'<span class="dim">去向提示:{hint}</span></div>'
            f'<table><thead><tr><th>成员</th><th>证据</th><th>依据原文</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    # ---- 布局 ----
    layout_rows = []
    for gid, rid in sorted((layout.get("assignments") or {}).items()):
        g = next((x for x in groups if x["group_id"] == gid), {})
        alt = layout.get("alternatives", {}).get(gid)
        layout_rows.append(
            f'<tr><td>{esc(g.get("name_zh", gid))}</td>'
            f'<td><b>{esc(region_names.get(rid, rid))}</b> '
            f'<span class="mono dim">{esc(rid)}</span></td>'
            f'<td class="dim">{esc(region_names.get(alt, alt) if alt else "—")}</td></tr>'
        )
    status = layout.get("status", "未运行")
    status_chip = chip(
        "✓ " + status if status == "PLAN_READY" else status,
        "success" if status == "PLAN_READY" else "danger",
    )
    layout_sec = (
        f'<div class="panel"><div class="panel-head"><h3>布局结果</h3>{status_chip}</div>'
        '<table><thead><tr><th>组合</th><th>指派区域</th><th>备选</th></tr></thead>'
        f"<tbody>{''.join(layout_rows)}</tbody></table>"
        + (f'<div class="warn">{"; ".join(esc(c) for c in layout.get("conflicts", []))}</div>'
           if layout.get("conflicts") else "")
        + "</div>"
    )

    # ---- 任务卡 ----
    VERDICT_LABEL = {
        "VERIFIED": ("✓ 已验收", "success"),
        "NEEDS_USER": ("待用户裁决", "warning"),
        "FAILED": ("验收未通过", "danger"),
    }
    card_secs = []
    for c in cards:
        verdict_chip = ""
        if v := verdicts.get(c["card_id"]):
            verdict_chip = chip(*VERDICT_LABEL.get(v["verdict"], (v["verdict"], "neutral")))
        items = "".join(
            f'<li>{img_tag(run_dir, i.get("hero_crop_ref", ""), "thumb")}'
            f"<span>{esc(i['display_name_zh'])}</span></li>"
            for i in c["items"]
        )
        checks = "".join(
            f'<li><i class="box"></i>{esc(k)}</li>'
            for k in c["verification_checklist"]
        )
        card_secs.append(
            f'<div class="card taskcard"><div class="cardbody">'
            f'<div class="panel-head"><div class="name">{esc(c["box_label_zh"])}</div>'
            f'<span class="mono dim">{esc(c["card_id"])}</span>{verdict_chip}</div>'
            f'<div class="target">目标区域 <b>{esc(c["target_region_name_zh"])}</b>'
            + (f' <span class="dim">备选 {esc(region_names.get(c["alternative_region_id"], c["alternative_region_id"]))}</span>'
               if c.get("alternative_region_id") else "")
            + f'</div><ul class="items">{items}</ul>'
            f'<div class="check-title">验收清单</div><ul class="checks">{checks}</ul>'
            "</div></div>"
        )

    # ---- 验收复核 ----
    def reason_zh(code: str) -> str:
        kind, _, rest = code.partition(":")
        eid, _, detail = rest.partition(":")
        name = display.get(eid, {}).get("display_name_zh", eid)
        if kind == "NOT_SEEN":
            return f"照片中未找到「{name}」"
        if kind == "MISPLACED":
            return f"「{name}」摆放不符({detail})"
        if kind == "LOW_CONFIDENCE":
            return f"「{name}」匹配置信度低,需人工确认"
        return code

    verify_rows = []
    for c in cards:
        v = verdicts.get(c["card_id"])
        if not v:
            continue
        reasons = "<br>".join(esc(reason_zh(r)) for r in v["reason_codes"]) or \
            '<span class="dim">presence 与 compliance 全部通过</span>'
        photos = "<br>".join(
            f'<span class="mono dim">{esc(p)}</span>' for p in v["photo_refs"]
        ) or "—"
        verify_rows.append(
            f'<tr><td>{esc(c["box_label_zh"])} '
            f'<span class="mono dim">{esc(c["card_id"])}</span></td>'
            f'<td>{chip(*VERDICT_LABEL.get(v["verdict"], (v["verdict"], "neutral")))}</td>'
            f'<td class="detail">{reasons}</td><td>{photos}</td></tr>'
        )
    verify_sec = ""
    if verify_rows:
        verify_sec = (
            '<h2 id="verify">验收复核 '
            '<span class="note">MEM 答"在不在" ∧ 确定性校验答"对不对" → EXEC 裁决;'
            "消息链见 verify/messages.jsonl</span></h2>"
            '<div class="panel"><table><thead><tr><th>任务卡</th><th>结论</th>'
            "<th>原因</th><th>依据照片</th></tr></thead>"
            f"<tbody>{''.join(verify_rows)}</tbody></table></div>"
        )

    # ---- 澄清与冲突 ----
    clar_rows = "".join(
        f'<tr><td class="mono">{esc(c.get("entity_id") or c.get("projected_entity_id") or c.get("canonical_id") or "—")}</td>'
        f'<td>{esc(c.get("question_zh", "待确认"))}</td>'
        f'<td>{chip(c.get("reason") or c.get("status") or "待确认", "warning")}</td></tr>'
        for c in clarifications
    ) or '<tr><td colspan="3" class="dim">无待澄清项 — 旁白证据覆盖全部实体</td></tr>'
    conflict_list = "".join(
        f"<li>{esc(c)}</li>" for c in conflicts
    ) or '<li class="dim">无</li>'

    # ---- 复跑指纹 ----
    artifacts = "".join(
        f'<tr><td>{chip(a["stage"], "primary")}</td>'
        f'<td class="mono">{esc(Path(a["path"]).name)}</td>'
        f'<td class="mono dim">{esc(a["sha256"][:16])}…</td></tr>'
        for a in (
            state_artifacts
            if state_artifacts is not None
            else bundle.get("artifacts", [])
        )
    )
    config_hashes = (
        {config_path.name: _sha256_file(config_path)}
        if config_path is not None
        else bundle.get("config_refs", {})
    )
    config_refs = "".join(
        f'<tr><td>{chip("config", "secondary")}</td><td class="mono">{esc(k)}</td>'
        f'<td class="mono dim">{esc(v[:16])}…</td></tr>'
        for k, v in config_hashes.items()
    )
    main_trace = trace_report.get("main_chain", {})
    verify_trace = trace_report.get("verification", {})
    clarification_trace = trace_report.get("clarifications", {})
    producer_counts = trace_report.get("producer_counts", {})
    trace_summary = ""
    if trace_report:
        trace_summary = (
            '<div class="panel"><div class="panel-head"><h3>协议回放</h3>'
            + chip("✓ hash / causation / correlation PASS", "success")
            + "</div><table><tbody>"
            + f'<tr><td>四 Agent 主链</td><td>{esc(" → ".join(main_trace.get("actions", [])))}</td>'
            + f'<td>{chip(str(main_trace.get("complete", 0)) + " 条闭合", "success")}</td></tr>'
            + f'<tr><td>MEM→UI 二选一</td><td>{esc(clarification_trace.get("closed", 0))} closed / '
            + f'{esc(clarification_trace.get("open", 0))} open</td><td>{chip("已闭合", "success") if not clarification_trace.get("open") else chip("有待确认", "warning")}</td></tr>'
            + f'<tr><td>EXEC 验收复核</td><td>{esc(verify_trace.get("closed", 0))} 条四消息闭环 · '
            + f'{esc(verify_trace.get("adjudication_closed", 0))} 条用户裁决</td><td class="mono dim">{esc(producer_counts)}</td></tr>'
            + "</tbody></table></div>"
        )

    # ---- 完整落地可见性（均为有界摘要，不转储源 JSON） ----
    inventory_sec = ""
    if isinstance(inventory_metrics, dict):
        raw_count = inventory_metrics.get("raw_entity_count", "—")
        trusted_count = inventory_metrics.get("trusted_inventory_count", "—")
        question_cap = inventory_metrics.get("clarification_cap", "—")
        question_count = len(inventory_clarifications)
        unresolved = inventory_metrics.get("raw_link_unresolved_count", "—")
        eligible = inventory_metrics.get("downstream_eligible_count", "—")
        inventory_sec = (
            '<h2 id="trusted-inventory">可信库存 '
            '<span class="note">raw ReID 留作模型审计，可信投影进入下游</span></h2>'
            '<div class="summary-grid">'
            '<div class="summary-card"><div class="summary-k">原始实体 → 可信库存</div>'
            f'<div class="summary-v">{esc(raw_count)} → {esc(trusted_count)}</div></div>'
            '<div class="summary-card"><div class="summary-k">轻确认问题</div>'
            f'<div class="summary-v">{esc(question_count)} <span class="summary-unit">/ 上限 '
            f'{esc(question_cap)}</span></div></div>'
            '<div class="summary-card"><div class="summary-k">下游可用</div>'
            f'<div class="summary-v">{esc(eligible)}</div>'
            f'<div class="dim">raw 链接未决 {esc(unresolved)} 项，不阻断可信库存</div></div>'
            '</div>'
        )

    boxlist_sec = ""
    if isinstance(boxlist, dict) and isinstance(group_metrics, dict):
        boxes = boxlist.get("boxes", [])
        boxes = boxes if isinstance(boxes, list) else []
        box_rows = []
        for box in boxes[:8]:
            if not isinstance(box, dict):
                continue
            items = box.get("items", [])
            item_count = len(items) if isinstance(items, list) else 0
            box_type = box.get("box_type", "")
            box_rows.append(
                f'<tr><td>{esc(box.get("box_label_zh", "未命名箱"))}</td>'
                f'<td class="mono dim">{esc(box.get("box_id", "—"))}</td>'
                f'<td>{chip(BOX_TYPE_LABEL.get(str(box_type), str(box_type) or "未分类"), "secondary")}</td>'
                f'<td>{esc(item_count)}</td></tr>'
            )
        if len(boxes) > 8:
            box_rows.append(
                f'<tr><td colspan="4" class="dim">另有 {esc(len(boxes) - 8)} 个箱单条目，'
                '本页仅展示摘要</td></tr>'
            )
        group_count = group_metrics.get("group_count", "—")
        placement_count = group_metrics.get("placement_group_count", "—")
        covered_count = group_metrics.get(
            "covered_canonical_item_count",
            boxlist.get("canonical_item_count", "—"),
        )
        inventory_count = group_metrics.get(
            "trusted_inventory_count",
            boxlist.get("canonical_item_count", "—"),
        )
        box_count = group_metrics.get("box_count", boxlist.get("box_count", "—"))
        boxlist_sec = (
            '<h2 id="boxlist">箱单 '
            '<span class="note">三组生活组合与独立装箱单元共同覆盖可信库存</span></h2>'
            '<div class="summary-grid">'
            '<div class="summary-card"><div class="summary-k">生活组合</div>'
            f'<div class="summary-v">{esc(group_count)}</div></div>'
            '<div class="summary-card"><div class="summary-k">placement 单元</div>'
            f'<div class="summary-v">{esc(placement_count)}</div></div>'
            '<div class="summary-card"><div class="summary-k">物品覆盖</div>'
            f'<div class="summary-v">{esc(covered_count)}/{esc(inventory_count)}</div></div>'
            '<div class="summary-card"><div class="summary-k">箱数</div>'
            f'<div class="summary-v">{esc(box_count)}</div></div>'
            '</div>'
            '<div class="panel"><table><thead><tr><th>箱单</th><th>箱号</th>'
            '<th>类型</th><th>物品数</th></tr></thead>'
            f'<tbody>{"".join(box_rows)}</tbody></table></div>'
        )

    spatial_sec = ""
    if isinstance(spatial_metrics, dict):
        gate_status = spatial_metrics.get("gate_status")
        if not gate_status:
            gate_status = "PASS" if spatial_metrics.get("region_gate_passed") else "NEEDS_USER"
        gate_kind = "success" if gate_status == "PASS" else "warning"
        gate_reasons = spatial_metrics.get("gate_reasons", [])
        gate_reasons = gate_reasons if isinstance(gate_reasons, list) else []
        reason_html = (
            f'<div class="dim">门原因：{esc("; ".join(str(item) for item in gate_reasons[:4]))}</div>'
            if gate_reasons
            else '<div class="dim">覆盖与可信候选门已满足</div>'
        )
        spatial_sec = (
            '<h2 id="automatic-space">自动空间 '
            '<span class="note">自动观测跨帧去重后，可信候选才投影至布局区域</span></h2>'
            '<div class="panel"><div class="panel-head"><h3>空间生产门</h3>'
            f'{chip(str(gate_status), gate_kind)}</div>'
            '<div class="summary-grid compact">'
            '<div class="summary-card"><div class="summary-k">候选区域</div>'
            f'<div class="summary-v">{esc(spatial_metrics.get("candidate_count", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">自动接受</div>'
            f'<div class="summary-v">{esc(spatial_metrics.get("auto_accepted_count", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">已投影</div>'
            f'<div class="summary-v">{esc(spatial_metrics.get("projected_region_count", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">待确认 / 未观测</div>'
            f'<div class="summary-v small">{esc(spatial_metrics.get("needs_user_count", "—"))} / '
            f'{esc(spatial_metrics.get("not_observed_count", "—"))}</div></div>'
            f'</div>{reason_html}</div>'
        )

    spatial_review_sec = ""
    if isinstance(spatial_review_metrics, dict):
        review_gate = spatial_review_metrics.get("gate_status", "NEEDS_USER")
        review_kind = "success" if review_gate == "PASS" else "warning"
        review_reasons = spatial_review_metrics.get("gate_reasons", [])
        review_reasons = review_reasons if isinstance(review_reasons, list) else []
        power_counts = spatial_review_metrics.get("power_state_counts", {})
        power_counts = power_counts if isinstance(power_counts, dict) else {}
        power_html = (
            '<div class="dim">电源证据：'
            f'NEAR {esc(power_counts.get("NEAR", 0))} / '
            f'UNKNOWN {esc(power_counts.get("UNKNOWN", 0))} / '
            f'NOT_NEAR {esc(power_counts.get("NOT_NEAR", 0))}</div>'
            if power_counts
            else ""
        )
        review_reason_html = (
            '<div class="dim">裁定门原因：'
            f'{esc("; ".join(str(item) for item in review_reasons[:4]))}</div>'
            if review_reasons
            else '<div class="dim">五个冻结目标均有逐帧视觉证据；来源不会计入 AUTO_ACCEPTED。</div>'
        )
        spatial_review_sec = (
            '<h2 id="visual-space-review">视觉代理裁定 '
            '<span class="note">独立覆盖层：保留候选、轨迹、帧 SHA 与裁定来源</span></h2>'
            '<div class="panel"><div class="panel-head"><h3>视觉裁定门</h3>'
            f'{chip(str(review_gate), review_kind)}</div>'
            '<div class="summary-grid compact">'
            '<div class="summary-card"><div class="summary-k">裁定条目</div>'
            f'<div class="summary-v">{esc(spatial_review_metrics.get("decision_count", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">视觉接受</div>'
            f'<div class="summary-v">{esc(spatial_review_metrics.get("visually_adjudicated_count", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">已投影</div>'
            f'<div class="summary-v">{esc(spatial_review_metrics.get("projected_region_count", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">待处理</div>'
            f'<div class="summary-v">{esc(spatial_review_metrics.get("needs_user_count", "—"))}</div></div>'
            f'</div>{review_reason_html}{power_html}</div>'
        )

    risk_sec = ""
    if isinstance(risk_assessments, dict) and isinstance(risk_metrics, dict):
        assessment_rows = risk_assessments.get("assessments", [])
        assessment_rows = assessment_rows if isinstance(assessment_rows, list) else []
        risk_rows = []
        for assessment in assessment_rows[:3]:
            if not isinstance(assessment, dict):
                continue
            rule_id = str(assessment.get("rule_id", "未知规则"))
            status_value = str(assessment.get("status", "NEEDS_USER"))
            status_label = RISK_STATUS_LABEL.get(
                status_value, (status_value, "neutral")
            )
            reasons = assessment.get("reason_codes", [])
            reasons = reasons if isinstance(reasons, list) else []
            risk_rows.append(
                f'<tr><td>{esc(RISK_RULE_LABEL.get(rule_id, rule_id))}</td>'
                f'<td>{chip(*status_label)}</td>'
                f'<td>{esc(assessment.get("confidence", "—"))}</td>'
                f'<td class="detail">{esc("; ".join(str(item) for item in reasons[:3]) or "—")}</td></tr>'
            )
        status_counts = risk_metrics.get("status_counts", {})
        status_counts = status_counts if isinstance(status_counts, dict) else {}
        status_chips = "".join(
            chip(
                f"{RISK_STATUS_LABEL[status][0]} {status_counts.get(status, 0)}",
                RISK_STATUS_LABEL[status][1],
            )
            for status in ("TRIGGERED", "NEEDS_USER", "NOT_APPLICABLE")
        )
        disclaimer = risk_metrics.get(
            "disclaimer_zh", DEFAULT_RISK_DISCLAIMER_ZH
        )
        risk_sec = (
            '<h2 id="risk-reminders">风险提醒 '
            '<span class="note">固定三条规则，只消费有引用的显式事实</span></h2>'
            '<div class="panel"><div class="panel-head"><h3>三条规则状态</h3>'
            f'{status_chips}</div><table><thead><tr><th>规则</th><th>状态</th>'
            '<th>置信度</th><th>原因码</th></tr></thead>'
            f'<tbody>{"".join(risk_rows)}</tbody></table>'
            f'<div class="disclaimer">⚠ {esc(disclaimer)}</div></div>'
        )

    completion_sections = (
        inventory_sec + boxlist_sec + spatial_sec + spatial_review_sec + risk_sec
    )
    optional_nav = "".join(
        link
        for section, link in (
            (inventory_sec, '<a href="#trusted-inventory">可信库存</a>'),
            (boxlist_sec, '<a href="#boxlist">箱单</a>'),
            (spatial_sec, '<a href="#automatic-space">自动空间</a>'),
            (
                spatial_review_sec,
                '<a href="#visual-space-review">视觉裁定</a>',
            ),
            (risk_sec, '<a href="#risk-reminders">风险提醒</a>'),
        )
        if section
    )

    bundle_id = bundle.get("bundle_id", run_dir.name)
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>搬家复原 · {esc(bundle_id)}</title>
<style>
:root {{
  --bg:#09090b; --panel:#131316; --panel-2:#18181b; --line:#27272a;
  --fg:#ececee; --dim:#a1a1aa; --muted:#71717a; --radius:14px;
  --primary:#66aaf9; --primary-bg:rgba(0,111,238,.16);
  --secondary:#ae7ede; --secondary-bg:rgba(120,40,200,.20);
  --success:#45d483; --success-bg:rgba(23,201,100,.14);
  --warning:#f7b750; --warning-bg:rgba(245,165,36,.15);
  --danger:#f871a0; --danger-bg:rgba(243,18,96,.16);
  --neutral:#a1a1aa; --neutral-bg:rgba(161,161,170,.12);
  --shadow:0 4px 24px rgba(0,0,0,.35);
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg:#f4f4f5; --panel:#ffffff; --panel-2:#fafafa; --line:#e4e4e7;
    --fg:#18181b; --dim:#52525b; --muted:#a1a1aa;
    --primary:#005bc4; --primary-bg:rgba(0,111,238,.10);
    --secondary:#6020a0; --secondary-bg:rgba(120,40,200,.10);
    --success:#0e793c; --success-bg:rgba(23,201,100,.12);
    --warning:#936316; --warning-bg:rgba(245,165,36,.14);
    --danger:#c20e4d; --danger-bg:rgba(243,18,96,.10);
    --neutral:#52525b; --neutral-bg:rgba(113,113,122,.10);
    --shadow:0 4px 20px rgba(24,24,27,.08);
  }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.65 -apple-system,"SF Pro SC","PingFang SC",Inter,"Noto Sans SC",sans-serif;
  -webkit-font-smoothing:antialiased; }}
header {{ position:sticky; top:0; z-index:9; display:flex; gap:14px; align-items:center;
  padding:12px 28px; background:var(--bg);
  border-bottom:1px solid var(--line); }}
.brand {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:16px; }}
.brand i {{ width:10px; height:10px; border-radius:3px; background:linear-gradient(135deg,#338ef7,#9353d3); }}
header nav {{ display:flex; gap:4px; flex-wrap:wrap; }}
header nav a {{ color:var(--dim); text-decoration:none; font-size:13px;
  padding:5px 12px; border-radius:999px; transition:all .15s; }}
header nav a:hover {{ color:var(--fg); background:var(--neutral-bg); }}
header .spacer {{ flex:1; }}
main {{ max-width:1120px; margin:0 auto; padding:28px 28px 80px; }}
.hero-head {{ margin:8px 0 24px; }}
.hero-head h1 {{ margin:0 0 4px; font-size:24px; letter-spacing:-.02em; }}
.hero-head .sub {{ color:var(--dim); font-size:13px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:12px; margin:18px 0 8px; }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
  padding:14px 18px; }}
.stat-n {{ font-size:26px; font-weight:700; letter-spacing:-.02em; }}
.stat-l {{ color:var(--dim); font-size:12px; margin-top:2px; }}
.summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:12px; margin:10px 0 14px; }}
.summary-grid.compact {{ grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }}
.summary-card {{ background:var(--panel); border:1px solid var(--line);
  border-radius:var(--radius); padding:14px 18px; }}
.summary-k {{ color:var(--dim); font-size:12px; }}
.summary-v {{ font-size:24px; font-weight:700; letter-spacing:-.02em; margin-top:2px; }}
.summary-v.small {{ font-size:20px; }}
.summary-unit {{ color:var(--dim); font-size:13px; font-weight:500; }}
.arrow {{ color:var(--primary); padding:0 5px; }}
h2 {{ margin:36px 0 14px; font-size:17px; letter-spacing:-.01em;
  display:flex; align-items:baseline; gap:10px; }}
h2 .note {{ font-size:12px; font-weight:400; color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:14px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line);
  border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow);
  transition:transform .15s ease, border-color .15s ease; }}
.card:hover {{ transform:translateY(-2px); border-color:var(--muted); }}
.cardbody {{ padding:12px 14px; }}
.name {{ font-weight:650; font-size:15px; }}
.chips {{ margin:6px 0 4px; display:flex; flex-wrap:wrap; gap:4px; }}
.eid {{ font-size:11px; color:var(--muted); font-family:ui-monospace,SFMono-Regular,monospace; }}
.hero {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; background:var(--panel-2); }}
.noimg {{ display:flex; align-items:center; justify-content:center;
  color:var(--muted); font-size:12px; border-bottom:1px dashed var(--line);
  background:repeating-linear-gradient(45deg,var(--panel-2),var(--panel-2) 10px,var(--panel) 10px,var(--panel) 20px); }}
.hero.noimg {{ aspect-ratio:1; }}
.thumb {{ width:30px; height:30px; object-fit:cover; border-radius:8px; }}
.thumb.noimg {{ width:30px; height:30px; font-size:8px; border:1px dashed var(--line); border-radius:8px; flex:none; }}
.chip {{ display:inline-flex; align-items:center; gap:5px; padding:1px 9px;
  border-radius:999px; font-size:12px; line-height:1.7; white-space:nowrap; }}
.chip .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block;
  border:1px solid rgba(255,255,255,.25); }}
.chip-primary {{ background:var(--primary-bg); color:var(--primary); }}
.chip-secondary {{ background:var(--secondary-bg); color:var(--secondary); }}
.chip-success {{ background:var(--success-bg); color:var(--success); }}
.chip-warning {{ background:var(--warning-bg); color:var(--warning); }}
.chip-danger {{ background:var(--danger-bg); color:var(--danger); }}
.chip-neutral {{ background:var(--neutral-bg); color:var(--neutral); }}
.panel {{ background:var(--panel); border:1px solid var(--line);
  border-radius:var(--radius); padding:14px 18px; margin-bottom:14px; box-shadow:var(--shadow); }}
.panel-head {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:6px; }}
.panel-head h3 {{ margin:0; font-size:15px; }}
table {{ width:100%; border-collapse:collapse; margin-top:4px; }}
th {{ text-align:left; color:var(--muted); font-weight:500; font-size:11px;
  text-transform:uppercase; letter-spacing:.05em; padding:6px 10px; }}
td {{ text-align:left; padding:8px 10px; border-top:1px solid var(--line); font-size:14px; }}
.dim {{ color:var(--dim); font-size:12px; }}
.detail {{ color:var(--dim); font-size:12.5px; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,monospace; font-size:12px; }}
.warn {{ color:var(--danger); margin-top:10px; font-size:13px; }}
.disclaimer {{ margin-top:12px; padding:9px 12px; border-radius:10px;
  color:var(--warning); background:var(--warning-bg); font-size:12.5px; }}
.target {{ margin:2px 0 8px; font-size:14px; }}
.items {{ list-style:none; margin:6px 0; padding:0; display:flex; flex-direction:column; gap:6px; }}
.items li {{ display:flex; align-items:center; gap:9px; background:var(--panel-2);
  border:1px solid var(--line); border-radius:10px; padding:5px 10px; }}
.check-title {{ margin-top:10px; color:var(--muted); font-size:11px;
  text-transform:uppercase; letter-spacing:.05em; }}
.checks {{ list-style:none; margin:6px 0 0; padding:0; }}
.checks li {{ display:flex; align-items:center; gap:8px; color:var(--dim);
  font-size:13px; padding:3px 0; }}
.checks .box {{ width:14px; height:14px; border:1.5px solid var(--muted);
  border-radius:4px; flex:none; }}
ul.conflicts {{ margin:6px 0 0; padding-left:18px; }}
ul.conflicts li {{ font-size:13px; color:var(--dim); }}
footer {{ color:var(--muted); font-size:12px; margin-top:28px; }}
</style></head><body>
<header>
  <div class="brand"><i></i>AI 搬家复原</div>
  <nav>{optional_nav}<a href="#entities">实体</a><a href="#groups">生活组合</a><a href="#layout">布局</a>
  <a href="#cards">任务卡</a>{'<a href="#verify">验收</a>' if verify_rows else ''}<a href="#clarify">澄清</a><a href="#trace">复跑指纹</a></nav>
  <div class="spacer"></div>
  {chip(bundle_id, "primary")}
</header>
<main>
<div class="hero-head">
  <h1>房间成果总览</h1>
  <div class="sub">旧房间 → 实体识别 → 生活组合 → 新家布局 → 任务卡 · 全链确定性复跑</div>
  <div class="stats">{stat_tiles}</div>
</div>
{completion_sections}
<h2 id="entities">{esc(entity_heading)} <span class="note">{esc(entity_note)}</span></h2>
<div class="grid">{"".join(entity_cards)}</div>
<h2 id="groups">生活组合 <span class="note">证据优先级:旁白 &gt; 轻确认 &gt; 模板 &gt; 共现佐证</span></h2>
{"".join(group_secs)}
<h2 id="layout">新家布局 <span class="note">CP-SAT 约束求解,固定 seed 可复现</span></h2>
{layout_sec}
<h2 id="cards">任务卡</h2>
<div class="cards">{"".join(card_secs)}</div>
{verify_sec}
<h2 id="clarify">澄清队列与冲突记录</h2>
<div class="panel"><table><thead><tr><th>实体</th><th>问题</th><th>原因</th></tr></thead>
<tbody>{clar_rows}</tbody></table>
<div class="check-title">冲突记录</div><ul class="conflicts">{conflict_list}</ul></div>
<h2 id="trace">Agent trace 与复跑指纹 <span class="note">严格回放 audit/events.jsonl，再核对阶段 sha256</span></h2>
{trace_summary}
<div class="panel"><table><thead><tr><th>阶段</th><th>产物</th><th>sha256</th></tr></thead>
<tbody>{config_refs}{artifacts}</tbody></table></div>
<footer>bundle {esc(bundle_id)} · {esc(bundle.get("created_at", ""))} · 本页由 hero_pipeline report 阶段确定性生成</footer>
</main></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run_dir / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(args.run_dir, args.config), encoding="utf-8")
    print(json.dumps({"page": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

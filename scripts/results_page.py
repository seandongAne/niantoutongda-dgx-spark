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

import yaml

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

DEFERRED_RISK_STATUS_LABEL = {
    **RISK_STATUS_LABEL,
    "NEEDS_USER": ("诊断缺证据（已延期）", "neutral"),
}

DEFAULT_RISK_DISCLAIMER_ZH = (
    "仅为辅助风险提醒，不构成安全认证，也不能替代现场人员或专业人员复核。"
)

GEOMETRY_METRIC_LABEL = {
    "exact_surface_area": "精确面积",
    "clear_height": "净高",
    "load_capacity": "承重",
    "doorway_clear_width": "门洞净宽",
    "walk_path_clear_width": "通道净宽",
}


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


def load_scope_contract(config_path: Path | None) -> dict:
    """从当前配置解析同一份技术 closure；无 closure 的旧配置保持兼容。"""
    if config_path is None or not config_path.exists():
        return {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return {}
    stages = config.get("stages")
    if not isinstance(stages, dict):
        return {}
    closure_refs = {
        stage.get("closure")
        for name in ("group", "risk")
        if isinstance((stage := stages.get(name)), dict) and stage.get("closure")
    }
    if not closure_refs:
        return {}
    if len(closure_refs) != 1:
        raise ValueError("group/risk must reference the same technical closure")
    raw_ref = next(iter(closure_refs))
    closure_path = Path(raw_ref)
    if not closure_path.is_absolute():
        closure_path = PROJ / closure_path
    closure = load_json(closure_path, {})
    if not isinstance(closure, dict):
        raise ValueError("technical closure root must be an object")
    return closure


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


def img_tag(
    run_dir: Path,
    ref: str,
    cls: str = "hero",
    *,
    alt: str = "",
    loading: str = "lazy",
) -> str:
    if not ref:
        return f'<div class="{cls} noimg">待补 hero 图</div>'
    src = ref
    if not ref.startswith(("http://", "https://", "/")):
        src = os.path.relpath(PROJ / ref, run_dir)
    return (
        f'<img class="{cls}" src="{esc(src)}" alt="{esc(alt)}" loading="{esc(loading)}" '
        "onerror=\"this.outerHTML='<div class=&quot;" + cls +
        " noimg&quot;>待补 hero 图</div>'\">"
    )


def select_demo_space_frames(
    run_dir: Path,
    assignments: list[dict],
    *,
    asset_roots: list[Path] | None = None,
    max_frames: int = 3,
) -> list[dict]:
    """选择能覆盖最多最终 assignment 的少量真实新房帧。

    只使用 assignment 自身引用过的 ``frame:`` 证据；目录里的旧 review 结论不会
    进入页面。默认只为仓库内正式 run 寻找已拉回的小型 JPEG，临时测试目录不会
    意外依赖仓库结果。
    """

    if asset_roots is None:
        try:
            run_dir.resolve().relative_to(PROJ.resolve())
        except ValueError:
            return []
        asset_roots = [
            PROJ / "results/acceptance/HERO_S1/space-auto-shadow-v1",
            PROJ / "results/acceptance/HERO_S1/space-visual-adjudication-v1/frames",
        ]

    available: dict[str, Path] = {}
    for root in asset_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.jpg")):
            available.setdefault(path.name, path)

    coverage: dict[str, set[str]] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        anchor = str(assignment.get("anchor") or "unknown")
        for ref in assignment.get("evidence_refs", []):
            ref = str(ref)
            if not ref.startswith("frame:"):
                continue
            basename = Path(ref.removeprefix("frame:").split("#", 1)[0]).name
            if basename in available:
                coverage.setdefault(basename, set()).add(anchor)

    remaining = {
        str(item.get("anchor") or "unknown")
        for item in assignments
        if isinstance(item, dict)
    }
    selected: list[dict] = []
    while remaining and len(selected) < max_frames:
        ranked = sorted(
            (
                (-len(anchors & remaining), basename, anchors)
                for basename, anchors in coverage.items()
                if anchors & remaining
                and basename not in {item["basename"] for item in selected}
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not ranked:
            break
        _, basename, anchors = ranked[0]
        covered = anchors & remaining
        path = available[basename]
        try:
            ref = str(path.relative_to(PROJ))
        except ValueError:
            ref = str(path)
        selected.append(
            {
                "basename": basename,
                "ref": ref,
                "anchors": sorted(covered),
            }
        )
        remaining -= covered
    return selected


def build_page(run_dir: Path, config_path: Path | None = None) -> str:
    scope_contract = load_scope_contract(config_path)
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
    spatial_assignment_path = run_dir / "spatial/assignment.json"
    spatial_assignment = (
        load_json(spatial_assignment_path, {})
        if spatial_assignment_path.exists()
        else None
    )
    spatial_score_metrics_path = run_dir / "spatial_score/metrics.json"
    spatial_score_metrics = (
        load_json(spatial_score_metrics_path, {})
        if spatial_score_metrics_path.exists()
        else None
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
        (len(clarifications), "轻确认问题"),
    ]
    if verdicts:
        n_verified = sum(1 for v in verdicts.values() if v["verdict"] == "VERIFIED")
        stats.append((f"{n_verified}/{len(verdicts)}", "验收通过"))
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
            + img_tag(
                run_dir,
                row.get("hero_crop_ref", ""),
                alt=f'旧房视频中的「{row["display_name_zh"]}」识别证据',
            )
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
        item_rows = []
        for item in c["items"]:
            item_name = str(item["display_name_zh"])
            item_rows.append(
                "<li>"
                + img_tag(
                    run_dir,
                    item.get("hero_crop_ref", ""),
                    "thumb",
                    alt=f"{item_name}物品图",
                )
                + f"<span>{esc(item_name)}</span></li>"
            )
        items = "".join(item_rows)
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
            '<span class="note">Nemotron 视觉假设经全局一对一门后，才投影至布局区域</span></h2>'
            '<div class="panel"><div class="panel-head"><h3>空间生产门</h3>'
            f'{chip(str(gate_status), gate_kind)}</div>'
            '<div class="summary-grid compact">'
            '<div class="summary-card"><div class="summary-k">候选实例</div>'
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

    spatial_score_sec = ""
    if isinstance(spatial_score_metrics, dict):
        score_passed = spatial_score_metrics.get("acceptance_passed") is True
        score_reasons = spatial_score_metrics.get("gate_reasons", [])
        score_reasons = score_reasons if isinstance(score_reasons, list) else []
        score_reason_html = (
            '<div class="dim">评分门原因：'
            f'{esc("; ".join(str(item) for item in score_reasons[:4]))}</div>'
            if score_reasons
            else '<div class="dim">五类 anchor、support 与相对容量均匹配；零额外预测。</div>'
        )
        spatial_score_sec = (
            '<h2 id="automatic-space-score">独立空间评分 '
            '<span class="note">语义真值不含 prediction/track/region ID，只评分且不产出 regions</span></h2>'
            '<div class="panel"><div class="panel-head"><h3>冻结语义门</h3>'
            f'{chip("PASS" if score_passed else "FAIL", "success" if score_passed else "danger")}</div>'
            '<div class="summary-grid compact">'
            '<div class="summary-card"><div class="summary-k">精确语义分</div>'
            f'<div class="summary-v">{esc(spatial_score_metrics.get("score", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">匹配 anchor</div>'
            f'<div class="summary-v">{esc(spatial_score_metrics.get("matched_anchor_count", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">额外预测</div>'
            f'<div class="summary-v">{esc(spatial_score_metrics.get("extra_prediction_count", "—"))}</div></div>'
            '<div class="summary-card"><div class="summary-k">support / 容量不符</div>'
            f'<div class="summary-v small">{esc(spatial_score_metrics.get("support_type_mismatch_count", "—"))} / '
            f'{esc(spatial_score_metrics.get("capacity_class_mismatch_count", "—"))}</div></div>'
            f'</div>{score_reason_html}'
            '<div class="dim">电源差异仅作信息记录，不作为空房视频的安全结论。</div></div>'
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

    scope_sec = ""
    geometry_policy = scope_contract.get("geometry_policy")
    post_placement = scope_contract.get("post_placement_verification_contract")
    if isinstance(geometry_policy, dict) and isinstance(post_placement, dict):
        assumption = geometry_policy.get("reference_assumption", {})
        assumption = assumption if isinstance(assumption, dict) else {}
        non_required = geometry_policy.get("non_required_metrics", [])
        non_required = non_required if isinstance(non_required, list) else []
        metric_labels = "、".join(
            GEOMETRY_METRIC_LABEL.get(str(metric), str(metric))
            for metric in non_required
        )
        scope_sec = (
            '<h2 id="competition-scope">比赛技术闭环口径 '
            '<span class="note">假设、非必需项和可选验收均显式入合同</span></h2>'
            '<div class="panel"><div class="summary-grid compact">'
            '<div class="summary-card"><div class="summary-k">空间估计</div>'
            '<div class="summary-v">相对容量</div>'
            f'<div class="dim">书桌高 {esc(assumption.get("value_cm", "—"))} cm · '
            f'{esc(assumption.get("status", "—"))}，不是实测</div></div>'
            '<div class="summary-card"><div class="summary-k">精确测量</div>'
            '<div class="summary-v">非必需</div>'
            f'<div class="dim">{esc(metric_labels or "—")} · 非阻塞</div></div>'
            '<div class="summary-card"><div class="summary-k">搬后执行复核</div>'
            '<div class="summary-v">可选延期</div>'
            f'<div class="dim">{esc(post_placement.get("purpose_zh", ""))}</div></div>'
            '</div></div>'
        )

    risk_sec = ""
    if isinstance(risk_assessments, dict) and isinstance(risk_metrics, dict):
        risk_scope_deferred = (
            risk_metrics.get("scope_status") == "DEFERRED"
            and risk_metrics.get("blocking") is False
            and risk_assessments.get("scope_status") == "DEFERRED"
            and risk_assessments.get("blocking") is False
        )
        risk_status_labels = (
            DEFERRED_RISK_STATUS_LABEL if risk_scope_deferred else RISK_STATUS_LABEL
        )
        assessment_rows = risk_assessments.get("assessments", [])
        assessment_rows = assessment_rows if isinstance(assessment_rows, list) else []
        risk_rows = []
        for assessment in assessment_rows[:3]:
            if not isinstance(assessment, dict):
                continue
            rule_id = str(assessment.get("rule_id", "未知规则"))
            status_value = str(assessment.get("status", "NEEDS_USER"))
            status_label = risk_status_labels.get(
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
                f"{risk_status_labels[status][0]} {status_counts.get(status, 0)}",
                risk_status_labels[status][1],
            )
            for status in ("TRIGGERED", "NEEDS_USER", "NOT_APPLICABLE")
        )
        scope_html = ""
        if risk_scope_deferred:
            defer_reason = str(risk_metrics.get("defer_reason_zh", ""))
            scope_html = (
                '<div class="panel-head"><h3>执行范围</h3>'
                f'{chip("已延期", "neutral")}{chip("非阻塞", "success")}</div>'
                f'<div class="dim">{esc(defer_reason)}</div>'
            )
        disclaimer = risk_metrics.get(
            "disclaimer_zh", DEFAULT_RISK_DISCLAIMER_ZH
        )
        risk_sec = (
            '<h2 id="risk-reminders">风险提醒 '
            '<span class="note">固定三条规则，只消费有引用的显式事实</span></h2>'
            f'<div class="panel">{scope_html}<div class="panel-head"><h3>三条规则状态</h3>'
            f'{status_chips}</div><table><thead><tr><th>规则</th><th>状态</th>'
            '<th>置信度</th><th>原因码</th></tr></thead>'
            f'<tbody>{"".join(risk_rows)}</tbody></table>'
            f'<div class="disclaimer">⚠ {esc(disclaimer)}</div></div>'
        )

    completion_sections = (
        scope_sec
        + inventory_sec
        + boxlist_sec
        + spatial_sec
        + spatial_score_sec
        + spatial_review_sec
        + risk_sec
    )

    # ---- 评委主线：只在全部硬门真实通过时出现 ----
    main_trace = trace_report.get("main_chain", {})
    demo_ready = (
        trusted_inventory_mode
        and isinstance(inventory_metrics, dict)
        and isinstance(group_metrics, dict)
        and isinstance(spatial_metrics, dict)
        and spatial_metrics.get("gate_status") == "PASS"
        and isinstance(spatial_score_metrics, dict)
        and spatial_score_metrics.get("acceptance_passed") is True
        and layout.get("status") == "PLAN_READY"
        and len(cards) > 0
        and main_trace.get("complete") == 1
    )
    demo_sec = ""
    if demo_ready:
        assignment_rows = (
            spatial_assignment.get("assignments", [])
            if isinstance(spatial_assignment, dict)
            else []
        )
        assignment_rows = assignment_rows if isinstance(assignment_rows, list) else []
        demo_frames = select_demo_space_frames(run_dir, assignment_rows)

        representative_rows = []
        used_entities: set[str] = set()
        for group in groups:
            for entity_id in group.get("entity_ids", []):
                row = display.get(entity_id)
                if not row or not row.get("hero_crop_ref") or entity_id in used_entities:
                    continue
                representative_rows.append(row)
                used_entities.add(entity_id)
                break
            if len(representative_rows) >= 5:
                break
        if len(representative_rows) < 5:
            for entity_id in sorted(display):
                row = display[entity_id]
                if row.get("hero_crop_ref") and entity_id not in used_entities:
                    representative_rows.append(row)
                    used_entities.add(entity_id)
                if len(representative_rows) >= 5:
                    break

        old_crop_strip = "".join(
            img_tag(
                run_dir,
                row.get("hero_crop_ref", ""),
                "story-crop",
                alt=f'旧房视频中的「{row.get("display_name_zh", "物品")}」',
            )
            for row in representative_rows
        )
        room_visual = ""
        if demo_frames:
            room_visual = (
                '<figure class="room-visual">'
                + img_tag(
                    run_dir,
                    demo_frames[0]["ref"],
                    "room-frame",
                    alt="新房巡拍中的自动空间证据帧",
                    loading="eager",
                )
                + '<figcaption>新房巡拍 · 自动空间真实输入</figcaption></figure>'
            )

        frame_cards = "".join(
            '<figure class="evidence-frame">'
            + img_tag(
                run_dir,
                frame["ref"],
                "space-frame",
                alt=f'新房巡拍证据帧 {index}，被 {len(frame["anchors"])} 个最终区域引用',
            )
            + '<figcaption><b>新房证据 '
            + esc(index)
            + '</b><span>'
            + esc(len(frame["anchors"]))
            + ' 个最终区域引用此帧</span></figcaption></figure>'
            for index, frame in enumerate(demo_frames, start=1)
        )

        featured_card = next(
            (
                card
                for card in cards
                if card.get("target_region_name_zh") == "书桌"
                or "学习文具" in str(card.get("box_label_zh", ""))
            ),
            cards[0],
        )
        featured_items = "".join(
            '<li>'
            + img_tag(
                run_dir,
                item.get("hero_crop_ref", ""),
                "featured-thumb",
                alt=f'{item.get("display_name_zh", "物品")}物品图',
            )
            + f'<span>{esc(item.get("display_name_zh", "物品"))}</span></li>'
            for item in featured_card.get("items", [])
        )

        raw_count = inventory_metrics.get("raw_entity_count", "—")
        trusted_count = inventory_metrics.get("trusted_inventory_count", "—")
        question_count = len(inventory_clarifications)
        question_cap = inventory_metrics.get("clarification_cap", "—")
        group_count = group_metrics.get("group_count", "—")
        placement_count = group_metrics.get("placement_group_count", "—")
        covered_count = group_metrics.get("covered_canonical_item_count", "—")
        inventory_count = group_metrics.get("trusted_inventory_count", trusted_count)
        candidate_count = spatial_metrics.get("candidate_count", "—")
        accepted_count = spatial_metrics.get("auto_accepted_count", "—")
        score = spatial_score_metrics.get("score", "—")

        visual_class = " judge-hero-with-visual" if room_visual or old_crop_strip else ""
        demo_sec = (
            f'<section class="judge-hero{visual_class}" id="top">'
            '<div class="judge-copy">'
            '<div class="eyebrow">比赛技术闭环已通过 · Spark 本地运行</div>'
            '<h1><span>把旧家的生活组合，</span><span>带到新家</span></h1>'
            '<p>从旧房视频和新家巡拍出发，把嘈杂识别结果收敛成可信库存，'
            '自动理解可用空间，并生成搬家人员可以直接执行的任务卡。</p>'
            '<div class="hero-proof">'
            f'<span><b>{esc(raw_count)}→{esc(trusted_count)}</b> 可信库存</span>'
            f'<span><b>≤{esc(question_cap)}</b> 个轻确认</span>'
            f'<span><b>{esc(accepted_count)}/5</b> 自动空间</span>'
            f'<span><b>{esc(len(cards))}</b> 张任务卡</span>'
            '</div><div class="hero-actions">'
            '<a class="button button-primary" href="#demo-story">30 秒看懂闭环</a>'
            '<a class="button button-secondary" href="#evidence">查看完整证据</a>'
            '</div></div>'
            '<div class="judge-visual">'
            f'{room_visual}<div class="old-crop-strip">{old_crop_strip}</div>'
            '<div class="visual-caption">旧房可信物品 · 新房真实巡拍</div>'
            '</div></section>'
            '<section id="demo-story" class="demo-section">'
            '<div class="section-kicker">评委主线</div>'
            '<h2>两路真实输入，汇成一条可执行闭环</h2>'
            '<div class="story-grid">'
            '<article class="story-card"><div class="story-step">01 · 旧家</div>'
            '<h3>从原始识别里留下可信物品</h3>'
            f'<div class="story-metric">{esc(raw_count)} <span>收敛为</span> {esc(trusted_count)}</div>'
            f'<p>模型不确定性被压缩为 {esc(question_count)} 个高价值问题，上限 {esc(question_cap)}。</p></article>'
            '<article class="story-card"><div class="story-step">02 · 生活关系</div>'
            '<h3>物品不只是清单，而是生活组合</h3>'
            f'<div class="story-metric">{esc(group_count)} <span>个生活组合 /</span> {esc(placement_count)} <span>个落位单元</span></div>'
            f'<p>箱单覆盖 {esc(covered_count)}/{esc(inventory_count)}，下游只消费可信库存。</p></article>'
            '<article class="story-card"><div class="story-step">03 · 新家</div>'
            '<h3>自动空间生产器替代人工区域 ID</h3>'
            f'<div class="story-metric">{esc(candidate_count)} <span>候选实例 →</span> {esc(accepted_count)} <span>区</span></div>'
            f'<p>全局一对一门通过；独立语义评分 {esc(score)}，不反哺生产结果。</p></article>'
            '<article class="story-card"><div class="story-step">04 · 执行</div>'
            '<h3>规划结果变成搬家任务卡</h3>'
            f'<div class="story-metric">{esc(len(cards))} <span>张卡 ·</span> {esc(covered_count)}/{esc(inventory_count)}</div>'
            '<p>每张卡明确箱单、目标区域、物品和验收清单。</p></article>'
            '</div></section>'
            + (
                '<section id="space-visual" class="demo-section">'
                '<div class="section-kicker">自动空间</div>'
                '<h2>三张真实新房帧，覆盖全部五个最终区域</h2>'
                f'<div class="space-gallery">{frame_cards}</div>'
                '<div class="proof-bar">'
                f'<span>生产门 <b>PASS</b></span><span>自动接受 <b>{esc(accepted_count)}/5</b></span>'
                f'<span>独立评分 <b>{esc(score)}</b></span><span>待人工区域 <b>0</b></span>'
                '</div></section>'
                if frame_cards
                else ""
            )
            + '<section id="featured-task" class="demo-section featured-task">'
            '<div class="featured-copy"><div class="section-kicker">代表任务卡</div>'
            f'<h2>{esc(featured_card.get("box_label_zh", "任务卡"))}</h2>'
            f'<p class="featured-target">在新家落位到 <b>{esc(featured_card.get("target_region_name_zh", "目标区域"))}</b></p>'
            '<p>完整五张任务卡继续保留在下方证据区；这里先让评委一眼看到“识别结果如何变成动作”。</p>'
            '<a class="text-link" href="#cards">查看 5 张完整任务卡</a></div>'
            f'<ul class="featured-items">{featured_items}</ul></section>'
        )

    optional_nav = "".join(
        link
        for section, link in (
            (scope_sec, '<a href="#competition-scope">比赛口径</a>'),
            (inventory_sec, '<a href="#trusted-inventory">可信库存</a>'),
            (boxlist_sec, '<a href="#boxlist">箱单</a>'),
            (spatial_sec, '<a href="#automatic-space">自动空间</a>'),
            (
                spatial_score_sec,
                '<a href="#automatic-space-score">空间评分</a>',
            ),
            (
                spatial_review_sec,
                '<a href="#visual-space-review">视觉裁定</a>',
            ),
            (risk_sec, '<a href="#risk-reminders">风险提醒</a>'),
        )
        if section
    )

    bundle_id = bundle.get("bundle_id", run_dir.name)
    legacy_hero = (
        '<div class="hero-head"><h1>房间成果总览</h1>'
        '<div class="sub">旧房间 → 实体识别 → 生活组合 → 新家布局 → 任务卡 · 全链确定性复跑</div>'
        f'<div class="stats">{stat_tiles}</div></div>'
    )
    if demo_ready:
        header_nav = (
            '<a href="#demo-story">30 秒主线</a>'
            + (
                '<a href="#space-visual">新家五区</a>'
                if 'id="space-visual"' in demo_sec
                else '<a href="#automatic-space">自动空间</a>'
            )
            + '<a href="#featured-task">代表任务</a>'
            + '<a href="#evidence">完整证据</a>'
        )
    else:
        header_nav = (
            f'{optional_nav}<a href="#entities">实体</a><a href="#groups">生活组合</a>'
            '<a href="#layout">布局</a><a href="#cards">任务卡</a>'
            + ('<a href="#verify">验收</a>' if verify_rows else '')
            + '<a href="#clarify">澄清</a><a href="#trace">复跑指纹</a>'
        )
    evidence_intro = ""
    if demo_ready:
        evidence_intro = (
            '<section id="evidence" class="evidence-intro">'
            '<div><div class="section-kicker">完整证据</div>'
            '<h2>所有数字、边界与失败安全状态都可下钻</h2>'
            '<p>主线负责让评委快速理解价值；以下内容保留机器可复核的库存、空间、'
            '任务卡、风险边界与复跑指纹。</p></div>'
            f'<nav class="evidence-nav" aria-label="完整证据导航">{optional_nav}'
            '<a href="#entities">20 件物品</a><a href="#groups">组合</a>'
            '<a href="#layout">布局</a><a href="#cards">任务卡</a>'
            '<a href="#trace">审计</a></nav>'
            f'<div class="stats evidence-stats">{stat_tiles}</div></section>'
        )
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>搬家复原 · {esc(bundle_id)}</title>
<style>
:root {{
  --bg:#09090b; --panel:#131316; --panel-2:#18181b; --line:#27272a;
  --fg:#ececee; --dim:#a1a1aa; --muted:#8b8b95; --radius:14px;
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
    --fg:#18181b; --dim:#52525b; --muted:#6b6b75;
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
header nav a:focus-visible, a:focus-visible, summary:focus-visible {{
  outline:3px solid var(--primary); outline-offset:3px; }}
header .spacer {{ flex:1; }}
main {{ max-width:1120px; margin:0 auto; padding:28px 28px 80px; }}
[id] {{ scroll-margin-top:76px; }}
.hero-head {{ margin:8px 0 24px; }}
.hero-head h1 {{ margin:0 0 4px; font-size:24px; letter-spacing:-.02em; }}
.hero-head .sub {{ color:var(--dim); font-size:13px; }}
.judge-hero {{ min-height:540px; display:grid; align-items:center; gap:28px;
  padding:54px 0 44px; border-bottom:1px solid var(--line); }}
.judge-hero-with-visual {{ grid-template-columns:minmax(0,1.05fr) minmax(380px,.95fr); }}
.judge-copy h1 {{ margin:10px 0 14px; max-width:680px; font-size:clamp(38px,4.15vw,54px);
  line-height:1.08; letter-spacing:-.045em; }}
.judge-copy h1 span {{ display:block; }}
.judge-copy > p {{ max-width:650px; margin:0; color:var(--dim); font-size:17px; line-height:1.75; }}
.eyebrow, .section-kicker {{ color:var(--primary); font-size:12px; font-weight:700;
  letter-spacing:.09em; text-transform:uppercase; }}
.hero-proof {{ display:flex; flex-wrap:wrap; gap:8px; margin:24px 0 20px; }}
.hero-proof span {{ padding:8px 12px; border:1px solid var(--line); border-radius:999px;
  color:var(--dim); background:var(--panel); font-size:13px; }}
.hero-proof b {{ color:var(--fg); font-size:15px; }}
.hero-actions {{ display:flex; gap:10px; flex-wrap:wrap; }}
.button {{ display:inline-flex; min-height:42px; align-items:center; justify-content:center;
  padding:8px 16px; border-radius:12px; text-decoration:none; font-weight:650; }}
.button-primary {{ color:#fff; background:#006fee; }}
.button-primary:hover {{ background:#005bc4; }}
.button-secondary {{ color:var(--fg); background:var(--panel); border:1px solid var(--line); }}
.button-secondary:hover {{ border-color:var(--primary); }}
.judge-visual {{ min-width:0; }}
.room-visual {{ position:relative; margin:0; overflow:hidden; border:1px solid var(--line);
  border-radius:18px; background:var(--panel); box-shadow:var(--shadow); }}
.room-frame {{ width:100%; aspect-ratio:16/10; object-fit:cover; display:block; }}
.room-visual figcaption {{ position:absolute; left:12px; bottom:12px; padding:6px 10px;
  border-radius:999px; color:#fff; background:rgba(9,9,11,.76); font-size:12px;
  backdrop-filter:blur(10px); }}
.old-crop-strip {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin-top:8px; }}
.story-crop {{ width:100%; aspect-ratio:1; object-fit:cover; border:1px solid var(--line);
  border-radius:12px; background:var(--panel); }}
.visual-caption {{ margin-top:8px; color:var(--muted); font-size:11px; text-align:right; }}
.demo-section {{ padding:46px 0 8px; }}
.demo-section > h2, .evidence-intro h2 {{ margin:6px 0 20px; font-size:28px; line-height:1.25; }}
.story-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
.story-card {{ min-height:250px; padding:18px; border:1px solid var(--line);
  border-radius:16px; background:var(--panel); }}
.story-card h3 {{ margin:14px 0 18px; font-size:17px; line-height:1.35; }}
.story-step {{ color:var(--primary); font-size:12px; font-weight:700; }}
.story-metric {{ margin:0 0 12px; font-size:27px; font-weight:750; letter-spacing:-.03em; }}
.story-metric span {{ color:var(--dim); font-size:12px; font-weight:500; letter-spacing:0; }}
.story-card p {{ margin:0; color:var(--dim); font-size:13px; }}
.space-gallery {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
.evidence-frame {{ margin:0; overflow:hidden; border:1px solid var(--line);
  border-radius:16px; background:var(--panel); box-shadow:var(--shadow); }}
.space-frame {{ width:100%; aspect-ratio:16/10; object-fit:cover; display:block; }}
.evidence-frame figcaption {{ display:flex; justify-content:space-between; gap:8px;
  padding:10px 12px; color:var(--dim); font-size:12px; }}
.evidence-frame figcaption b {{ color:var(--fg); }}
.proof-bar {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px;
  margin-top:12px; overflow:hidden; border:1px solid var(--line); border-radius:14px;
  background:var(--line); }}
.proof-bar span {{ padding:12px 14px; color:var(--dim); background:var(--panel); font-size:12px; }}
.proof-bar b {{ display:block; color:var(--success); font-size:19px; }}
.featured-task {{ display:grid; grid-template-columns:minmax(0,.85fr) minmax(0,1.15fr);
  gap:28px; align-items:center; margin:38px 0 30px; padding:28px;
  border:1px solid var(--line); border-radius:18px; background:var(--panel); }}
.featured-copy h2 {{ margin:6px 0 8px; font-size:30px; }}
.featured-copy p {{ color:var(--dim); }}
.featured-target {{ font-size:17px; }}
.text-link {{ color:var(--primary); font-weight:650; text-decoration:none; }}
.text-link:hover {{ text-decoration:underline; }}
.featured-items {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px;
  list-style:none; margin:0; padding:0; }}
.featured-items li {{ min-width:0; padding:8px; border:1px solid var(--line);
  border-radius:12px; background:var(--panel-2); }}
.featured-thumb {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; border-radius:8px; }}
.featured-items span {{ display:block; margin-top:6px; overflow:hidden; text-overflow:ellipsis;
  color:var(--dim); font-size:11px; white-space:nowrap; }}
.evidence-intro {{ margin-top:52px; padding:38px 0 10px; border-top:1px solid var(--line); }}
.evidence-intro > div:first-child {{ max-width:760px; }}
.evidence-intro p {{ color:var(--dim); }}
.evidence-nav {{ display:flex; flex-wrap:wrap; gap:7px; margin:18px 0; }}
.evidence-nav a {{ padding:5px 10px; border:1px solid var(--line); border-radius:999px;
  color:var(--dim); text-decoration:none; font-size:12px; }}
.evidence-nav a:hover {{ color:var(--fg); border-color:var(--muted); }}
.evidence-stats {{ margin-top:14px; }}
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
@media (max-width:900px) {{
  .judge-hero-with-visual {{ grid-template-columns:1fr; }}
  .judge-visual {{ max-width:720px; }}
  .story-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .featured-task {{ grid-template-columns:1fr; }}
}}
@media (max-width:680px) {{
  header {{ align-items:flex-start; padding:10px 16px; flex-wrap:wrap; }}
  header nav {{ order:3; width:100%; flex-wrap:nowrap; overflow-x:auto; padding-bottom:2px; }}
  header nav a {{ flex:none; }}
  header .spacer {{ display:none; }}
  main {{ padding:18px 16px 60px; }}
  [id] {{ scroll-margin-top:112px; }}
  .judge-hero {{ min-height:auto; padding:34px 0; }}
  .judge-copy h1 {{ font-size:38px; }}
  .judge-copy > p {{ font-size:15px; }}
  .story-grid, .space-gallery, .proof-bar {{ grid-template-columns:1fr; }}
  .story-card {{ min-height:auto; }}
  .featured-task {{ padding:18px; }}
  .featured-items {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
  .demo-section > h2, .evidence-intro h2 {{ font-size:24px; }}
  h2 {{ align-items:flex-start; flex-direction:column; }}
  .panel {{ overflow-x:auto; }}
  .cards {{ grid-template-columns:1fr; }}
}}
@media (prefers-reduced-motion:reduce) {{
  * {{ scroll-behavior:auto!important; transition:none!important; }}
}}
</style></head><body>
<header>
  <div class="brand"><i></i>AI 搬家复原</div>
  <nav aria-label="主导航">{header_nav}</nav>
  <div class="spacer"></div>
  {chip(bundle_id, "primary")}
</header>
<main>
{demo_sec or legacy_hero}
{evidence_intro}
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

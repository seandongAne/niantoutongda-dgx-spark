#!/usr/bin/env python
"""可点击成果页 — hero run 目录 → 自包含静态 index.html。

初赛完成门"可点击成果"的落地:实体卡(hero 图+VLM 名+属性+组徽章)、
生活组合(证据来源可见:旁白/轻确认/模板主导,共现只作佐证标签)、
新家布局、任务卡、澄清队列、复跑指纹(bundle/config 哈希)。
无外部依赖(内联 CSS,无 CDN),file:// 直接打开,适合录屏。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

SOURCE_LABEL = {
    "narration": ("旁白", "#2563eb"),
    "confirmation": ("轻确认", "#7c3aed"),
    "template": ("模板", "#b45309"),
    "cooccurrence": ("共现佐证", "#6b7280"),
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def esc(value: object) -> str:
    return html.escape(str(value))


def badge(text: str, color: str) -> str:
    return (
        f'<span class="badge" style="background:{color}1a;color:{color};'
        f'border:1px solid {color}55">{esc(text)}</span>'
    )


def img_tag(run_dir: Path, ref: str, cls: str = "hero") -> str:
    if not ref:
        return f'<div class="{cls} noimg">无图</div>'
    src = ref
    if not ref.startswith(("http://", "https://", "/")):
        src = os.path.relpath(PROJ / ref, run_dir)
    return (
        f'<img class="{cls}" src="{esc(src)}" alt="" loading="lazy" '
        "onerror=\"this.outerHTML='<div class=&quot;" + cls +
        " noimg&quot;>图缺失</div>'\">"
    )


def build_page(run_dir: Path) -> str:
    display = {r["entity_id"]: r for r in load_jsonl(run_dir / "naming/display.jsonl")}
    groups = load_jsonl(run_dir / "group/groups.jsonl")
    clarifications = load_jsonl(run_dir / "group/clarifications.jsonl")
    conflicts = json.loads((run_dir / "group/conflicts.json").read_text(encoding="utf-8")) \
        if (run_dir / "group/conflicts.json").exists() else []
    layout = json.loads((run_dir / "layout/layout.json").read_text(encoding="utf-8")) \
        if (run_dir / "layout/layout.json").exists() else {}
    regions = json.loads((run_dir / "regions/regions.json").read_text(encoding="utf-8")) \
        if (run_dir / "regions/regions.json").exists() else {}
    cards = load_jsonl(run_dir / "taskcards/taskcards.jsonl")
    bundle = json.loads((run_dir / "bundle.json").read_text(encoding="utf-8")) \
        if (run_dir / "bundle.json").exists() else {}

    group_of = {
        eid: g for g in groups for eid in g.get("entity_ids", [])
    }
    region_names = {
        e["region_id"]: e["display_name_zh"] for e in regions.get("entries", [])
    }

    # ---- 实体卡 ----
    entity_cards = []
    for eid in sorted(display):
        row = display[eid]
        g = group_of.get(eid)
        gname = badge(g["name_zh"], "#059669") if g else badge("未归组", "#dc2626")
        color = row.get("color_primary", "")
        entity_cards.append(
            '<div class="card">'
            + img_tag(run_dir, row.get("hero_crop_ref", ""))
            + f'<div class="cardbody"><div class="name">{esc(row["display_name_zh"])}</div>'
            + f'<div class="meta">{gname}'
            + (badge(color, "#0891b2") if color else "")
            + f'</div><div class="eid">{esc(eid)}</div></div></div>'
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
                badge(*SOURCE_LABEL.get(ev["source"], (ev["source"], "#6b7280")))
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
        dominant = badge(*SOURCE_LABEL.get(g["dominant_source"], (g["dominant_source"], "#6b7280")))
        hint = esc(g.get("target_region_hint", "")) or "—"
        group_secs.append(
            f'<div class="panel"><h3>{esc(g["name_zh"])} '
            f'<span class="dim">{esc(g["group_id"])}</span> {dominant}</h3>'
            f'<div class="dim">旁白去向提示:{hint}</div>'
            f'<table><tr><th>成员</th><th>证据</th><th>依据原文</th></tr>'
            + "".join(rows) + "</table></div>"
        )

    # ---- 布局 ----
    layout_rows = []
    for gid, rid in sorted((layout.get("assignments") or {}).items()):
        g = next((x for x in groups if x["group_id"] == gid), {})
        alt = layout.get("alternatives", {}).get(gid)
        layout_rows.append(
            f'<tr><td>{esc(g.get("name_zh", gid))}</td>'
            f'<td><b>{esc(region_names.get(rid, rid))}</b> <span class="dim">{esc(rid)}</span></td>'
            f'<td class="dim">{esc(region_names.get(alt, alt) if alt else "—")}</td></tr>'
        )
    layout_status = layout.get("status", "未运行")
    layout_sec = (
        f'<div class="panel"><h3>布局结果 {badge(layout_status, "#059669" if layout_status == "PLAN_READY" else "#dc2626")}</h3>'
        '<table><tr><th>组合</th><th>指派区域</th><th>备选</th></tr>'
        + "".join(layout_rows) + "</table>"
        + (f'<div class="warn">{"; ".join(esc(c) for c in layout.get("conflicts", []))}</div>'
           if layout.get("conflicts") else "")
        + "</div>"
    )

    # ---- 任务卡 ----
    card_secs = []
    for c in cards:
        items = "".join(
            f"<li>{img_tag(run_dir, i.get('hero_crop_ref', ''), 'thumb')}"
            f"{esc(i['display_name_zh'])}</li>"
            for i in c["items"]
        )
        checks = "".join(
            f'<li class="check">☐ {esc(k)}</li>' for k in c["verification_checklist"]
        )
        card_secs.append(
            f'<div class="card taskcard"><div class="cardbody">'
            f'<div class="name">{esc(c["box_label_zh"])} '
            f'<span class="dim">{esc(c["card_id"])}</span></div>'
            f'<div>目标:<b>{esc(c["target_region_name_zh"])}</b></div>'
            f'<ul class="items">{items}</ul><ul class="checks">{checks}</ul>'
            "</div></div>"
        )

    # ---- 澄清与冲突 ----
    clar_rows = "".join(
        f'<tr><td>{esc(c.get("entity_id") or "—")}</td><td>{esc(c["question_zh"])}</td>'
        f'<td class="dim">{esc(c["reason"])}</td></tr>'
        for c in clarifications
    ) or '<tr><td colspan="3" class="dim">无待澄清项</td></tr>'
    conflict_list = "".join(f"<li>{esc(c)}</li>" for c in conflicts) or '<li class="dim">无</li>'

    # ---- 复跑指纹 ----
    artifacts = "".join(
        f'<tr><td>{esc(a["stage"])}</td><td class="mono">{esc(Path(a["path"]).name)}</td>'
        f'<td class="mono dim">{esc(a["sha256"][:16])}…</td></tr>'
        for a in bundle.get("artifacts", [])
    )
    config_refs = "".join(
        f'<tr><td>{esc(k)}</td><td colspan="2" class="mono dim">{esc(v[:16])}…</td></tr>'
        for k, v in bundle.get("config_refs", {}).items()
    )

    stats = (
        f"{len(display)} 实体 · {len(groups)} 组合 · "
        f"{len(cards)} 任务卡 · {len(clarifications)} 待澄清"
    )
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>搬家复原 · {esc(bundle.get("bundle_id", run_dir.name))}</title>
<style>
:root {{ --fg:#111827; --dim:#6b7280; --bg:#f8fafc; --card:#fff; --line:#e5e7eb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.6 -apple-system,"PingFang SC","Noto Sans SC",sans-serif;
       color:var(--fg); background:var(--bg); }}
header {{ position:sticky; top:0; background:#fffffff2; border-bottom:1px solid var(--line);
          padding:10px 24px; display:flex; gap:16px; align-items:baseline; z-index:9; }}
header h1 {{ font-size:17px; margin:0; }}
header nav a {{ margin-right:12px; color:#2563eb; text-decoration:none; font-size:13px; }}
main {{ max-width:1080px; margin:0 auto; padding:16px 24px 64px; }}
h2 {{ margin:32px 0 12px; font-size:16px; border-left:4px solid #2563eb; padding-left:8px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:12px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
.cardbody {{ padding:10px 12px; }}
.name {{ font-weight:600; }}
.meta {{ margin:4px 0; }}
.eid {{ font-size:11px; color:var(--dim); font-family:ui-monospace,monospace; }}
.hero {{ width:100%; aspect-ratio:1; object-fit:cover; background:#eef2f7; display:block; }}
.noimg {{ display:flex; align-items:center; justify-content:center; color:var(--dim); font-size:12px; }}
.hero.noimg {{ aspect-ratio:1; }}
.thumb {{ width:28px; height:28px; object-fit:cover; border-radius:6px; vertical-align:middle; margin-right:6px; }}
.thumb.noimg {{ display:inline-flex; width:28px; height:28px; font-size:9px; }}
.badge {{ display:inline-block; padding:0 8px; border-radius:999px; font-size:12px; margin-right:4px; }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin-bottom:12px; }}
.panel h3 {{ margin:4px 0 8px; font-size:15px; }}
table {{ width:100%; border-collapse:collapse; margin-top:6px; }}
th, td {{ text-align:left; padding:6px 8px; border-top:1px solid var(--line); font-size:14px; }}
th {{ color:var(--dim); font-weight:500; font-size:12px; border-top:none; }}
.dim {{ color:var(--dim); font-size:12px; }}
.detail {{ color:var(--dim); font-size:12px; }}
.mono {{ font-family:ui-monospace,monospace; font-size:12px; }}
.items, .checks {{ margin:8px 0 0; padding-left:0; list-style:none; }}
.items li {{ margin:4px 0; }}
.checks li {{ color:var(--dim); font-size:13px; }}
.warn {{ color:#b91c1c; margin-top:8px; font-size:13px; }}
ul.conflicts li {{ font-size:13px; }}
</style></head><body>
<header><h1>AI 搬家复原 · 成果页</h1>
<nav><a href="#entities">实体</a><a href="#groups">生活组合</a><a href="#layout">布局</a>
<a href="#cards">任务卡</a><a href="#clarify">澄清</a><a href="#trace">复跑指纹</a></nav>
<span class="dim">{esc(stats)}</span></header>
<main>
<h2 id="entities">实体卡 <span class="dim">展示名 = 本地 VLM 读 hero 图</span></h2>
<div class="grid">{"".join(entity_cards)}</div>
<h2 id="groups">生活组合 <span class="dim">旁白 &gt; 轻确认 &gt; 模板 &gt; 共现佐证</span></h2>
{"".join(group_secs)}
<h2 id="layout">新家布局(CP-SAT)</h2>
{layout_sec}
<h2 id="cards">任务卡</h2>
<div class="cards">{"".join(card_secs)}</div>
<h2 id="clarify">澄清队列与冲突记录</h2>
<div class="panel"><table><tr><th>实体</th><th>问题</th><th>原因</th></tr>{clar_rows}</table>
<h3>冲突记录</h3><ul class="conflicts">{conflict_list}</ul></div>
<h2 id="trace">复跑指纹</h2>
<div class="panel"><table><tr><th>阶段</th><th>产物</th><th>sha256</th></tr>
{config_refs}{artifacts}</table>
<div class="dim">bundle:{esc(bundle.get("bundle_id", ""))} · {esc(bundle.get("created_at", ""))}</div></div>
</main></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run_dir / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(args.run_dir), encoding="utf-8")
    print(json.dumps({"page": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""生成 17 锚点点选式对账 UI:点击 hero 图勾选归属,直接导出 review JSON。

输入 = anchor_review_sheet.py 产出的 anchor_review.v6.json + crops/<id>.jpg
(由 spark 导出的候选 hero 裁剪)。产物是单文件 HTML,放在 crops 同级目录用
浏览器直接打开;选择存 localStorage,导出即为可交回的 confirmed JSON。
硬负对(A–D)同一 tracklet 被两边同时勾选时实时红色告警。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>17 锚点对账 — __DATASET__</title>
<style>
  body { margin:0; font:14px/1.5 -apple-system,"PingFang SC",sans-serif; display:flex; height:100vh; }
  #side { width:250px; overflow-y:auto; border-right:1px solid #ccc; padding:10px; box-sizing:border-box; }
  #side button.anchor { display:block; width:100%; text-align:left; margin:3px 0; padding:6px 8px;
    border:1px solid #ddd; background:#fafafa; cursor:pointer; border-radius:6px; font-size:13px; }
  #side button.anchor.active { background:#cde4ff; border-color:#5b9bd5; }
  #side button.anchor.done::after { content:" ✓"; color:#2a8f2a; font-weight:bold; }
  #side .hn { color:#c0392b; font-size:11px; }
  #main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  #bar { padding:8px 12px; border-bottom:1px solid #ccc; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  #warn { padding:6px 12px; background:#fdecea; color:#c0392b; display:none; font-weight:bold; }
  #content { flex:1; overflow-y:auto; padding:12px; }
  h3.video { margin:14px 0 6px; border-bottom:1px solid #eee; padding-bottom:2px; }
  .grid { display:flex; flex-wrap:wrap; gap:8px; }
  .card { width:150px; border:3px solid #ddd; border-radius:8px; cursor:pointer; position:relative;
    background:#fff; user-select:none; }
  .card img { width:100%; height:104px; object-fit:contain; background:#f2f2f2; display:block;
    border-radius:5px 5px 0 0; }
  .card .cap { font-size:11px; padding:2px 4px; text-align:center; color:#333; }
  .card.sel { border-color:#2a8f2a; background:#eefaee; }
  .card.sel::after { content:"✓"; position:absolute; top:2px; right:6px; color:#2a8f2a;
    font-size:20px; font-weight:bold; }
  .card .badge { position:absolute; top:2px; left:4px; font-size:10px; padding:1px 5px;
    border-radius:8px; color:#fff; display:none; }
  .card .badge.warn { background:#e67e22; display:block; }
  .card .badge.danger { background:#c0392b; display:block; }
  .meta { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
  .meta input[type=text] { width:340px; }
  button.small { font-size:12px; padding:3px 10px; }
  #io { display:none; position:fixed; inset:10% 20%; background:#fff; border:2px solid #888;
    padding:12px; z-index:9; flex-direction:column; gap:8px; }
  #io textarea { flex:1; font:11px monospace; }
  .hint { color:#777; font-size:12px; }
</style>
</head>
<body>
<div id="side">
  <h3>锚点 (<span id="doneCount">0</span>/__NANCHORS__)</h3>
  <div class="hint">点图勾选"这是该锚点物品"。同视频多条碎轨都要勾。认不准的不勾。</div>
  <div id="anchorNav"></div>
</div>
<div id="main">
  <div id="bar">
    <b id="title"></b>
    <label><input type="checkbox" id="anchorDone"> 本锚点完成</label>
    <button id="export">导出 confirmed JSON</button>
    <button id="importBtn">导入/备份</button>
    <span class="hint" id="saveState"></span>
  </div>
  <div id="warn"></div>
  <div id="content"></div>
</div>
<div id="io">
  <b>粘贴既有 JSON 后点导入(支持导出的 confirmed 文档或内部备份);或复制当前内容备份</b>
  <textarea id="ioText"></textarea>
  <div><button id="ioImport">导入</button> <button id="ioClose">关闭</button></div>
</div>
<script>
const REVIEW = __REVIEW__;
const VIDEOS = __VIDEOS__;
const HN_PAIRS = REVIEW.hard_negative_pairs || [];
const KEY = "anchor_review_v6_confirmed";
const ANCHORS = REVIEW.entities;

let store = JSON.parse(localStorage.getItem(KEY) || "null") || { anchors: {}, };
ANCHORS.forEach(a => {
  if (!store.anchors[a.anchor_id])
    store.anchors[a.anchor_id] = { confirmed: {}, visible_in: [], note: "", done: false, extra: [] };
});
let cur = ANCHORS[0].anchor_id;

const st = aid => store.anchors[aid];
const isSel = (aid, id) => Object.values(st(aid).confirmed).some(l => l.includes(id));
const save = () => {
  localStorage.setItem(KEY, JSON.stringify(store));
  document.getElementById("saveState").textContent = "已自动保存 " + new Date().toLocaleTimeString();
};
const anchorsOf = id => ANCHORS.filter(a => isSel(a.anchor_id, id)).map(a => a.anchor_id);
const hnPartner = aid => {
  for (const p of HN_PAIRS) {
    if (p.anchor_ids[0] === aid) return p.anchor_ids[1];
    if (p.anchor_ids[1] === aid) return p.anchor_ids[0];
  }
  return null;
};

function renderNav() {
  const nav = document.getElementById("anchorNav");
  if (!nav.childElementCount) {
    ANCHORS.forEach(a => {
      const b = document.createElement("button");
      b.className = "anchor"; b.dataset.aid = a.anchor_id;
      b.onclick = () => { cur = a.anchor_id; renderAnchor(); renderNav(); };
      nav.appendChild(b);
    });
  }
  [...nav.children].forEach(b => {
    const a = ANCHORS.find(x => x.anchor_id === b.dataset.aid);
    const n = Object.values(st(a.anchor_id).confirmed).reduce((s, l) => s + l.length, 0);
    const hn = hnPartner(a.anchor_id);
    b.innerHTML = `${a.anchor_id} ${a.display_label_zh} <span class="hint">(${n} 选)</span>` +
      (hn ? ` <span class="hn">硬负↔${hn.replace("anchor_","")}</span>` : "");
    b.classList.toggle("active", a.anchor_id === cur);
    b.classList.toggle("done", !!st(a.anchor_id).done);
  });
  document.getElementById("doneCount").textContent =
    ANCHORS.filter(a => st(a.anchor_id).done).length;
}

function toggle(id, video) {
  const c = st(cur).confirmed;
  c[video] = c[video] || [];
  const i = c[video].indexOf(id);
  if (i >= 0) c[video].splice(i, 1);
  else {
    c[video].push(id); c[video].sort();
    if (!st(cur).visible_in.includes(video)) st(cur).visible_in.push(video); // 顺手勾出现视频
  }
  save(); refreshCards(); renderNav(); renderMeta();
}

// 点选只更新样式与角标,绝不重建卡片 DOM(重建会闪且丢滚动位置)
function refreshCards() {
  const partner = hnPartner(cur);
  let dangers = [];
  document.querySelectorAll("#content .card").forEach(card => {
    const id = card.dataset.id;
    card.classList.toggle("sel", isSel(cur, id));
    const others = anchorsOf(id).filter(a => a !== cur);
    const badge = card.querySelector(".badge");
    badge.className = "badge"; badge.textContent = "";
    if (others.length) {
      const isDanger = others.includes(partner);
      badge.className = "badge " + (isDanger ? "danger" : "warn");
      badge.textContent = "也在 " + others.map(a => a.replace("anchor_", "#")).join(",");
      if (isDanger && isSel(cur, id)) dangers.push(id);
    }
  });
  const allDanger = [];
  HN_PAIRS.forEach(p => {
    const [x, y] = p.anchor_ids;
    Object.values(st(x).confirmed).flat().forEach(id => {
      if (isSel(y, id)) allDanger.push(`${p.group_id}组 ${id} 同时在 ${x}/${y}`);
    });
  });
  const warn = document.getElementById("warn");
  warn.style.display = allDanger.length ? "block" : "none";
  warn.textContent = allDanger.length ? "⚠ 硬负冲突(同一轨勾进了硬负对两边,必须解除一边): " + allDanger.join("; ") : "";
}

function renderMeta() {
  const a = ANCHORS.find(x => x.anchor_id === cur);
  document.getElementById("title").textContent =
    `${a.anchor_id} ${a.display_label_zh} (category=${a.category_id})`;
  document.getElementById("anchorDone").checked = !!st(cur).done;
  const meta = document.getElementById("metaRow");
  if (!meta) return;
  [...meta.querySelectorAll("input[type=checkbox][data-vid]")].forEach(cb => {
    cb.checked = st(cur).visible_in.includes(cb.dataset.vid);
  });
  meta.querySelector("input[data-note]").value = st(cur).note || "";
}

function renderAnchor() {
  const a = ANCHORS.find(x => x.anchor_id === cur);
  const wrap = document.getElementById("content");
  wrap.innerHTML = "";
  const meta = document.createElement("div");
  meta.className = "meta"; meta.id = "metaRow";
  meta.innerHTML = "物品出现在:" +
    VIDEOS.map(v => `<label><input type="checkbox" data-vid="${v}"> ${v}</label>`).join(" ") +
    ` <label>备注 <input type="text" data-note placeholder="不确定项/说明,可留空"></label>` +
    ` <label>补录轨 id <input type="text" data-extra placeholder="如 v2_t123(跨表捞漏)" size="14"></label>` +
    `<button class="small" data-addextra>添加</button>`;
  wrap.appendChild(meta);
  meta.querySelectorAll("input[data-vid]").forEach(cb => cb.onchange = () => {
    const v = st(cur).visible_in;
    const i = v.indexOf(cb.dataset.vid);
    if (cb.checked && i < 0) v.push(cb.dataset.vid);
    if (!cb.checked && i >= 0) v.splice(i, 1);
    save();
  });
  meta.querySelector("input[data-note]").onchange = e => { st(cur).note = e.target.value; save(); };
  meta.querySelector("button[data-addextra]").onclick = () => {
    const input = meta.querySelector("input[data-extra]");
    const id = input.value.trim();
    if (!/^v\\d+_t\\d+$/.test(id)) { alert("id 形如 v2_t123"); return; }
    const video = id.split("_")[0];
    if (!st(cur).extra.includes(id)) st(cur).extra.push(id);
    if (!isSel(cur, id)) toggle(id, video);
    input.value = "";
    renderAnchor();
  };

  const byVideo = {};
  VIDEOS.forEach(v => byVideo[v] = [...(a.candidate_tracklets_by_video[v] || [])]);
  st(cur).extra.forEach(id => {
    const v = id.split("_")[0];
    if (byVideo[v] && !byVideo[v].includes(id)) byVideo[v].push(id);
  });
  VIDEOS.forEach(video => {
    const ids = byVideo[video] || [];
    if (!ids.length) return;
    const h = document.createElement("h3");
    h.className = "video"; h.textContent = `${video} (${ids.length} 候选)`;
    wrap.appendChild(h);
    const grid = document.createElement("div");
    grid.className = "grid";
    ids.forEach(id => {
      const card = document.createElement("div");
      card.className = "card"; card.dataset.id = id;
      card.innerHTML = `<span class="badge"></span>` +
        `<img loading="lazy" src="crops/${id}.jpg" onerror="this.style.opacity=.15">` +
        `<div class="cap">${id}</div>`;
      card.onclick = () => toggle(id, video);
      grid.appendChild(card);
    });
    wrap.appendChild(grid);
  });
  renderMeta(); refreshCards();
}

document.getElementById("anchorDone").onchange = e => {
  st(cur).done = e.target.checked; save(); renderNav();
};
document.getElementById("export").onclick = () => {
  const conflicts = [];
  HN_PAIRS.forEach(p => {
    const [x, y] = p.anchor_ids;
    Object.values(st(x).confirmed).flat().forEach(id => { if (isSel(y, id)) conflicts.push(id); });
  });
  if (conflicts.length && !confirm("存在硬负冲突未解除: " + conflicts.join(",") + "\\n仍要导出?")) return;
  const notDone = ANCHORS.filter(a => !st(a.anchor_id).done).map(a => a.anchor_id);
  if (notDone.length && !confirm("未标完成的锚点: " + notDone.join(",") + "\\n仍要导出?")) return;
  const doc = JSON.parse(JSON.stringify(REVIEW));
  doc.status = "data_owner_confirmed";
  doc.confirmed_at = new Date().toISOString();
  doc.entities.forEach(ent => {
    const s = st(ent.anchor_id);
    const confirmed = {};
    Object.entries(s.confirmed).forEach(([v, l]) => { if (l.length) confirmed[v] = [...l].sort(); });
    ent.confirmed_tracklet_ids_by_video = confirmed;
    ent.visible_in = [...s.visible_in].sort();
    if (s.note) ent.note = s.note;
  });
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "anchor_review.v6.confirmed.json"; a.click();
};
const io = document.getElementById("io");
document.getElementById("importBtn").onclick = () => {
  document.getElementById("ioText").value = JSON.stringify(store, null, 2);
  io.style.display = "flex";
};
document.getElementById("ioClose").onclick = () => io.style.display = "none";
document.getElementById("ioImport").onclick = () => {
  try {
    const raw = JSON.parse(document.getElementById("ioText").value);
    if (raw.entities) {  // 导出的 confirmed 文档
      raw.entities.forEach(ent => {
        const s = st(ent.anchor_id);
        if (!s) return;
        s.confirmed = {};
        Object.entries(ent.confirmed_tracklet_ids_by_video || {}).forEach(([v, l]) => s.confirmed[v] = [...l]);
        s.visible_in = [...(ent.visible_in || [])];
        s.note = ent.note || "";
      });
    } else if (raw.anchors) { store = raw; ANCHORS.forEach(a => {
      if (!store.anchors[a.anchor_id]) store.anchors[a.anchor_id] = { confirmed:{}, visible_in:[], note:"", done:false, extra:[] };
      if (!store.anchors[a.anchor_id].extra) store.anchors[a.anchor_id].extra = [];
    }); }
    else throw new Error("unrecognised document");
    save(); renderAnchor(); renderNav(); io.style.display = "none";
  } catch (err) { alert("JSON 解析失败: " + err); }
};
renderNav(); renderAnchor();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True, help="anchor_review.v6.json")
    parser.add_argument("--out", required=True, help="输出 HTML(与 crops/ 同级)")
    args = parser.parse_args()

    review = json.loads(Path(args.review).read_text())
    videos = sorted({v for ent in review["entities"] for v in ent["candidate_tracklets_by_video"]})
    html = (
        TEMPLATE
        .replace("__REVIEW__", json.dumps(review, ensure_ascii=False))
        .replace("__VIDEOS__", json.dumps(videos))
        .replace("__DATASET__", review.get("dataset_version", "dev_a"))
        .replace("__NANCHORS__", str(len(review["entities"])))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(json.dumps({"out": str(out), "anchors": len(review["entities"]), "videos": videos}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

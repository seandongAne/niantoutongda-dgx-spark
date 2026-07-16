#!/usr/bin/env python
"""生成离线 hardval 标注器:单文件 HTML,画框 + 导出 detection_eval GT 格式。

本地运行(帧图拉回后):
  python scripts/hardval_annotator.py \
    --manifest fixtures/dev_a/hardval/manifest.json \
    --vocab fixtures/dev_a/vocab.json \
    --anchors fixtures/dev_a/annotations/entities.template.json \
    --out fixtures/dev_a/hardval/annotate.html

用浏览器直接打开产物即可标注;数据自动存 localStorage,导出按钮生成
gt.json 下载。图片经相对路径加载,无任何网络依赖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>hardval 真值标注 — __DATASET__</title>
<style>
  body { margin:0; font:13px/1.4 -apple-system,sans-serif; display:flex; height:100vh; }
  #side { width:230px; overflow-y:auto; border-right:1px solid #ccc; padding:8px; box-sizing:border-box; }
  #side h3 { margin:4px 0; font-size:13px; }
  #frames button { display:block; width:100%; text-align:left; margin:2px 0; padding:4px 6px;
    border:1px solid #ddd; background:#fafafa; cursor:pointer; border-radius:4px; }
  #frames button.active { background:#cde4ff; border-color:#5b9bd5; }
  #frames button.done::after { content:" ✓"; color:#2a8f2a; font-weight:bold; }
  #main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  #bar { padding:6px 10px; border-bottom:1px solid #ccc; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  #stage { flex:1; overflow:auto; position:relative; background:#333; }
  canvas { display:block; cursor:crosshair; }
  #boxes { width:330px; overflow-y:auto; border-left:1px solid #ccc; padding:8px; box-sizing:border-box; }
  .box-row { border:1px solid #ddd; border-radius:4px; padding:6px; margin-bottom:6px; }
  .box-row.selected { border-color:#e67e22; background:#fff7ef; }
  .box-row label { display:block; margin:2px 0; }
  .box-row input[type=text], .box-row select { width:100%; box-sizing:border-box; }
  button.small { font-size:12px; padding:2px 8px; }
  #io { display:none; position:fixed; inset:10% 20%; background:#fff; border:2px solid #888;
    padding:12px; z-index:9; flex-direction:column; gap:8px; }
  #io textarea { flex:1; font:11px monospace; }
  .hint { color:#777; font-size:12px; }
</style>
</head>
<body>
<div id="side">
  <h3>帧列表 (<span id="doneCount">0</span>/__NFRAMES__)</h3>
  <div class="hint">拖拽画框;点行选中;Delete 删框;逐帧勾"完成"。</div>
  <div id="frames"></div>
</div>
<div id="main">
  <div id="bar">
    <b id="frameTitle"></b>
    <label><input type="checkbox" id="frameDone"> 本帧完成</label>
    <button id="export">导出 gt.json</button>
    <button id="importBtn">导入/查看 JSON</button>
    <span class="hint" id="saveState"></span>
  </div>
  <div id="stage"><canvas id="cv"></canvas></div>
</div>
<div id="boxes"><h3>本帧实例</h3><div id="boxList"></div></div>
<div id="io">
  <b>粘贴既有 gt.json 后点导入;或复制当前内容备份</b>
  <textarea id="ioText"></textarea>
  <div><button id="ioImport">导入</button> <button id="ioClose">关闭</button></div>
</div>
<script>
const MANIFEST = __MANIFEST__;
const CANONICALS = __CANONICALS__;
const ANCHORS = __ANCHORS__;
const DATASET = "__DATASET__";
const KEY = "hardval_gt_" + DATASET;

let store = JSON.parse(localStorage.getItem(KEY) || "null") || { frames: {}, done: {} };
let cur = 0, img = new Image(), scale = 1, sel = -1, drag = null;

const frameKey = f => f.video_id + "/" + f.frame_id;
const curFrame = () => MANIFEST.frames[cur];
const curBoxes = () => store.frames[frameKey(curFrame())] || [];
const setBoxes = b => { store.frames[frameKey(curFrame())] = b; save(); };

function save() {
  localStorage.setItem(KEY, JSON.stringify(store));
  document.getElementById("saveState").textContent = "已自动保存 " + new Date().toLocaleTimeString();
  renderSide();
}
function renderSide() {
  const wrap = document.getElementById("frames");
  if (!wrap.childElementCount) {
    MANIFEST.frames.forEach((f, i) => {
      const b = document.createElement("button");
      b.textContent = f.video_id + " · " + f.frame_id + " (" + f.selection_reason + ")";
      b.onclick = () => load(i);
      wrap.appendChild(b);
    });
  }
  [...wrap.children].forEach((b, i) => {
    const f = MANIFEST.frames[i];
    b.classList.toggle("active", i === cur);
    b.classList.toggle("done", !!store.done[frameKey(f)]);
  });
  document.getElementById("doneCount").textContent =
    MANIFEST.frames.filter(f => store.done[frameKey(f)]).length;
}
function load(i) {
  cur = i; sel = -1;
  const f = curFrame();
  document.getElementById("frameTitle").textContent = f.video_id + " / kf_" + f.frame_id + ".jpg";
  document.getElementById("frameDone").checked = !!store.done[frameKey(f)];
  img = new Image();
  img.onload = () => { drawCanvas(); renderBoxList(); };
  img.src = f.image;
  renderSide();
}
// 画布重绘与列表重建彻底分离:重建 DOM 会销毁刚点开的原生下拉,
// 因此选中/改字段只走 drawCanvas + refreshRowClasses,绝不重建列表。
function drawCanvas() {
  const cv = document.getElementById("cv"), stage = document.getElementById("stage");
  scale = Math.min(1, (stage.clientWidth - 4) / img.naturalWidth);
  cv.width = img.naturalWidth * scale; cv.height = img.naturalHeight * scale;
  const g = cv.getContext("2d");
  g.drawImage(img, 0, 0, cv.width, cv.height);
  curBoxes().forEach((b, i) => {
    g.lineWidth = i === sel ? 3 : 2;
    g.strokeStyle = i === sel ? "#e67e22" : (b.visible === false ? "#999" : "#3ddc84");
    g.strokeRect(b.bbox[0]*scale, b.bbox[1]*scale, (b.bbox[2]-b.bbox[0])*scale, (b.bbox[3]-b.bbox[1])*scale);
    g.font = "12px sans-serif"; g.fillStyle = g.strokeStyle;
    g.fillText(b.instance_id + ":" + b.canonical_id, b.bbox[0]*scale + 2, b.bbox[1]*scale + 12);
  });
  if (drag) {
    g.strokeStyle = "#ff5252"; g.lineWidth = 1.5;
    g.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0);
  }
}
function refreshRowClasses() {
  [...document.getElementById("boxList").children].forEach(
    (row, i) => row.classList.toggle("selected", i === sel));
}
function selectRow(i) { sel = i; refreshRowClasses(); drawCanvas(); }
function renderBoxList() {
  const wrap = document.getElementById("boxList");
  wrap.innerHTML = "";
  curBoxes().forEach((b, i) => {
    const div = document.createElement("div");
    div.className = "box-row" + (i === sel ? " selected" : "");
    div.innerHTML =
      '<label>instance_id <input type="text" list="anchorList" value="' + (b.instance_id||"") + '"></label>' +
      '<label>canonical <select>' +
        CANONICALS.map(c => '<option' + (c === b.canonical_id ? " selected" : "") + '>' + c + '</option>').join("") +
      '</select></label>' +
      '<label><input type="checkbox"' + (b.visible === false ? "" : " checked") + '> visible</label>' +
      '<button class="small">删除</button>';
    const [inst, canon, vis, del] =
      [div.querySelector("input[type=text]"), div.querySelector("select"),
       div.querySelector("input[type=checkbox]"), div.querySelector("button")];
    div.onclick = () => selectRow(i);
    inst.onchange = () => { b.instance_id = inst.value.trim(); setBoxes(curBoxes()); drawCanvas(); };
    canon.onchange = () => { b.canonical_id = canon.value; setBoxes(curBoxes()); drawCanvas(); };
    vis.onchange = () => { b.visible = vis.checked; setBoxes(curBoxes()); drawCanvas(); };
    del.onclick = e => { e.stopPropagation(); const bs = curBoxes(); bs.splice(i,1); setBoxes(bs); sel=-1; renderBoxList(); drawCanvas(); };
    wrap.appendChild(div);
  });
}
const cv = document.getElementById("cv");
cv.onmousedown = e => {
  const r = cv.getBoundingClientRect();
  drag = { x0: e.clientX - r.left, y0: e.clientY - r.top, x1: e.clientX - r.left, y1: e.clientY - r.top };
};
cv.onmousemove = e => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  drag.x1 = e.clientX - r.left; drag.y1 = e.clientY - r.top; drawCanvas();
};
cv.onmouseup = e => {
  if (!drag) return;
  const x0 = Math.min(drag.x0, drag.x1)/scale, y0 = Math.min(drag.y0, drag.y1)/scale;
  const x1 = Math.max(drag.x0, drag.x1)/scale, y1 = Math.max(drag.y0, drag.y1)/scale;
  drag = null;
  if ((x1-x0) > 4 && (y1-y0) > 4) {
    const bs = curBoxes();
    bs.push({ instance_id: "", canonical_id: CANONICALS[0], bbox: [x0,y0,x1,y1], visible: true });
    sel = bs.length - 1;
    setBoxes(bs);
    renderBoxList();
  }
  drawCanvas();
};
document.onkeydown = e => {
  if ((e.key === "Delete" || e.key === "Backspace") && sel >= 0
      && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") {
    const bs = curBoxes(); bs.splice(sel, 1); setBoxes(bs); sel = -1;
    renderBoxList(); drawCanvas(); e.preventDefault();
  }
};
document.getElementById("frameDone").onchange = e => {
  store.done[frameKey(curFrame())] = e.target.checked; save();
};
document.getElementById("export").onclick = () => {
  const missing = [];
  const frames = MANIFEST.frames.map(f => ({
    sequence_id: f.video_id,
    frame_id: f.frame_id,
    instances: (store.frames[frameKey(f)] || []).map(b => {
      if (!b.instance_id) missing.push(f.video_id + "/" + f.frame_id);
      return { instance_id: b.instance_id, canonical_id: b.canonical_id,
               bbox: b.bbox.map(v => Math.round(v*100)/100), visible: b.visible !== false };
    }),
  }));
  if (missing.length) { alert("存在空 instance_id:\\n" + [...new Set(missing)].join("\\n")); return; }
  const blob = new Blob([JSON.stringify({ dataset_id: DATASET, frames }, null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "gt.json"; a.click();
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
    if (raw.frames && Array.isArray(raw.frames)) {  // detection_eval GT 格式
      store = { frames: {}, done: {} };
      raw.frames.forEach(f => {
        store.frames[f.sequence_id + "/" + f.frame_id] = (f.instances || []).map(i => ({
          instance_id: i.instance_id, canonical_id: i.canonical_id,
          bbox: i.bbox, visible: i.visible !== false }));
      });
    } else { store = raw; }  // 内部备份格式
    save(); load(cur); io.style.display = "none";
  } catch (err) { alert("JSON 解析失败: " + err); }
};
const dl = document.createElement("datalist");
dl.id = "anchorList";
ANCHORS.forEach(a => { const o = document.createElement("option"); o.value = a; dl.appendChild(o); });
document.body.appendChild(dl);
window.onresize = drawCanvas;
renderSide(); load(0);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--anchors", help="entities.template.json,提供 instance_id 候选")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    vocab = json.loads(Path(args.vocab).read_text())
    canonicals = sorted({entry["canonical_id"] for entry in vocab["entries"]})
    anchors: list[str] = []
    if args.anchors:
        template = json.loads(Path(args.anchors).read_text())
        anchors = [row["anchor_id"] for row in template.get("entities", [])]

    html = (
        TEMPLATE
        .replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False))
        .replace("__CANONICALS__", json.dumps(canonicals, ensure_ascii=False))
        .replace("__ANCHORS__", json.dumps(anchors, ensure_ascii=False))
        .replace("__DATASET__", manifest["dataset_id"])
        .replace("__NFRAMES__", str(len(manifest["frames"])))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(json.dumps({"out": str(out), "frames": len(manifest["frames"]), "canonicals": len(canonicals)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

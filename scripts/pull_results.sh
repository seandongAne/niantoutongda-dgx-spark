#!/usr/bin/env bash
# 从远程拉回小体积产出:results/ 与 logs/(远程日志放 logs/remote/)。
# 纪律:赛方禁止 SSH 传大文件——预检本次增量,超过门限直接拒绝并指路
# scripts/pull_results_r2.sh(R2 回程通道)。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs/remote

MAX_MB=50
# 预检失败(断连常态)按网络纪律重试一次;仍失败则带原因 fail-closed,
# 不能让 set -e 在赋值管道里无声死掉、体积门成死代码。
stats=$(rsync -azn --stats spark:~/proj/results/ ./results/ 2>&1) \
  || stats=$(rsync -azn --stats spark:~/proj/results/ ./results/ 2>&1) \
  || { echo "pull_results: 体积预检 rsync 失败(已重试一次,疑似断连):" >&2
       printf '%s\n' "$stats" >&2; exit 2; }
delta_bytes=$(printf '%s\n' "$stats" | grep -i "transferred file size" \
  | grep -oE '[0-9][0-9,.]*' | head -1 | tr -d ',' || true)
delta_bytes=${delta_bytes:-0}
delta_mb=$((delta_bytes / 1024 / 1024))
if [ "$delta_mb" -gt "$MAX_MB" ]; then
  echo "本次 results/ 增量约 ${delta_mb}MB,超过 SSH 门限 ${MAX_MB}MB。" >&2
  echo "请改用 R2 回程通道拉大目录,例如:" >&2
  echo "  scripts/pull_results_r2.sh results/hero_s1/reid-w2 results/hero_s1/" >&2
  echo "小文件可用 rsync 排除大目录后单独拉取。" >&2
  exit 3
fi

rsync -az spark:~/proj/results/ ./results/
rsync -az spark:~/proj/logs/ ./logs/remote/
echo "pull_results: OK (增量 ~${delta_mb}MB)"

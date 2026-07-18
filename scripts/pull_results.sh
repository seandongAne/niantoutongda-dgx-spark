#!/usr/bin/env bash
# 从远程拉回小体积产出:results/ 与 logs/(远程日志放 logs/remote/)。
# 用法:
#   scripts/pull_results.sh                         # 全量小结果同步
#   scripts/pull_results.sh --files path [path ...] # 精确拉取 results/ 或 logs/ 文件
# 纪律:赛方禁止 SSH 传大文件——预检本次增量,超过门限直接拒绝并指路
# scripts/pull_results_r2.sh(R2 回程通道)。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs/remote

MAX_MB=50

if [ "${1:-}" = "--files" ]; then
  shift
  [ "$#" -gt 0 ] || { echo "usage: $0 --files path [path ...]" >&2; exit 2; }
  total_bytes=0
  for path in "$@"; do
    case "$path" in
      results/*|logs/*) ;;
      *) echo "pull_results: path must be under results/ or logs/: $path" >&2; exit 2 ;;
    esac
    case "$path" in
      *[!A-Za-z0-9._/-]*|*/../*|*/..)
        echo "pull_results: unsafe relative path: $path" >&2
        exit 2
        ;;
    esac
    size=$(ssh spark "stat -c %s -- ~/proj/$path") \
      || size=$(ssh spark "stat -c %s -- ~/proj/$path") \
      || { echo "pull_results: cannot stat remote file after retry: $path" >&2; exit 2; }
    case "$size" in
      ''|*[!0-9]*) echo "pull_results: invalid remote size for $path: $size" >&2; exit 2 ;;
    esac
    total_bytes=$((total_bytes + size))
  done
  total_mb=$((total_bytes / 1024 / 1024))
  if [ "$total_mb" -gt "$MAX_MB" ]; then
    echo "selected files total about ${total_mb}MB, above ${MAX_MB}MB SSH limit" >&2
    exit 3
  fi
  for path in "$@"; do
    case "$path" in
      results/*)
        mkdir -p "$(dirname "$path")"
        rsync -az "spark:~/proj/$path" "$path"
        ;;
      logs/*)
        relative=${path#logs/}
        destination="logs/remote/$relative"
        mkdir -p "$(dirname "$destination")"
        rsync -az "spark:~/proj/$path" "$destination"
        ;;
    esac
  done
  echo "pull_results: OK (selected files: $#, total ~${total_mb}MB)"
  exit 0
fi

# 预检失败(断连常态)按网络纪律重试一次;仍失败则带原因 fail-closed,
# 不能让 set -e 在赋值管道里无声死掉、体积门成死代码。
# 注意:macOS openrsync 的 dry-run --stats 恒报 0 transferred(2026-07-18 实锤,
# 门形同虚设),必须改用 --out-format 逐文件求和;目录行带尾 / 排除。
stats=$(rsync -azn --out-format='%l %n' spark:~/proj/results/ ./results/ 2>&1) \
  || stats=$(rsync -azn --out-format='%l %n' spark:~/proj/results/ ./results/ 2>&1) \
  || { echo "pull_results: 体积预检 rsync 失败(已重试一次,疑似断连):" >&2
       printf '%s\n' "$stats" >&2; exit 2; }
delta_bytes=$(printf '%s\n' "$stats" \
  | awk '$1 ~ /^[0-9]+$/ && $NF !~ /\/$/ {s+=$1} END {printf "%d", s}' || true)
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

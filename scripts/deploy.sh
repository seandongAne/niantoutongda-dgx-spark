#!/usr/bin/env bash
# 部署本地代码到远程 spark:~/proj/(单向覆盖,远程改动会被删除)
# 用法:
#   scripts/deploy.sh                         # 全量部署,保留历史行为
#   scripts/deploy.sh --files path [path ...] # 仅部署指定文件,不删除远端其他内容
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--files" ]; then
  shift
  [ "$#" -gt 0 ] || { echo "usage: $0 --files path [path ...]" >&2; exit 2; }
  for path in "$@"; do
    case "$path" in
      /*|../*|*/../*|*/..)
        echo "deploy: unsafe relative path: $path" >&2
        exit 2
        ;;
    esac
    [ -f "$path" ] || { echo "deploy: file not found: $path" >&2; exit 2; }
  done
  rsync -azR "$@" spark:~/proj/
  echo "deploy: OK -> spark:~/proj/ (selected files: $#)"
  exit 0
fi

# .git 不同步,远端 provenance 靠 COMMIT 文件;工作区有未提交改动时标记 -dirty
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if [ "$commit" != "unknown" ] && ! git diff --quiet HEAD 2>/dev/null; then
  commit="${commit}-dirty"
fi
echo "$commit" > COMMIT
rsync -az --delete --exclude .git --exclude logs --exclude results \
  --exclude .venv --exclude local-data --exclude .DS_Store --exclude __pycache__ \
  --exclude .env ./ spark:~/proj/
echo "deploy: OK -> spark:~/proj/ (commit=$commit)"

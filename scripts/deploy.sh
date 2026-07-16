#!/usr/bin/env bash
# 部署本地代码到远程 spark:~/proj/(单向覆盖,远程改动会被删除)
set -euo pipefail
cd "$(dirname "$0")/.."
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

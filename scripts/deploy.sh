#!/usr/bin/env bash
# 部署本地代码到远程 spark:~/proj/(单向覆盖,远程改动会被删除)
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --delete --exclude .git --exclude logs --exclude results \
  --exclude .venv --exclude local-data --exclude .DS_Store --exclude __pycache__ \
  --exclude .env ./ spark:~/proj/
echo "deploy: OK -> spark:~/proj/"

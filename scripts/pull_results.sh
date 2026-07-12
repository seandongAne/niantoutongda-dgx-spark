#!/usr/bin/env bash
# 从远程拉回小体积产出:results/ 与 logs/(远程日志放 logs/remote/)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs/remote
rsync -az spark:~/proj/results/ ./results/
rsync -az spark:~/proj/logs/ ./logs/remote/
echo "pull_results: OK"

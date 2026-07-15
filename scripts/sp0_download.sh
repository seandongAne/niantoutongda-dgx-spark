#!/usr/bin/env bash
# SP0 权重批量下载 — 在 spark 上通过 nohup 执行。幂等:modelscope 断点续传,重跑安全。
# 权重落在 ~/models(deploy.sh 的 rsync --delete 只打 ~/proj,不会误删权重)。
set -uo pipefail
# shellcheck disable=SC1090
source ~/venv/bin/activate
mkdir -p ~/models
LIST=~/proj/configs/download_list.txt
FAIL=0
while read -r mid; do
  [[ -z "$mid" || "$mid" == \#* ]] && continue
  dest=~/models/"${mid//\//__}"
  echo "== $(date -u +%H:%M:%S) downloading $mid -> $dest =="
  if modelscope download --model "$mid" --local_dir "$dest"; then
    echo "DOWNLOAD_OK $mid"
  else
    echo "DOWNLOAD_FAIL $mid"; FAIL=1
  fi
done < "$LIST"
df -h ~ | tail -1
[ "$FAIL" -eq 0 ] && echo "ALL_DOWNLOADS_OK" || echo "SOME_DOWNLOADS_FAILED"

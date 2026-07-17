#!/usr/bin/env bash
# 大体积产出回程通道:spark →(curl PUT 分块,无凭据)→ R2 Worker →(wrangler API)→ Mac。
#
# 背景:赛方禁止 SSH/rsync 传大文件(实测 ~17KB/s 亦不可用);素材日已验证
# Mac→R2→spark 方向,本脚本补反方向。凭据纪律:wrangler OAuth 只在 Mac;
# spark 只见"本次传输专用的一次性随机 token + Worker URL",传毕对象即删、
# token 即作废(disabled 重部署)。
#
# 用法:
#   scripts/pull_results_r2.sh <remote-path-under-~/proj> [local-dest-dir]
# 例:
#   scripts/pull_results_r2.sh results/hero_s1/reid-w2 results/hero_s1/
# 注意:>200MB 的传输请以后台方式发射本脚本并轮询输出,不要挂前台等。
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_PATH="${1:?usage: pull_results_r2.sh <remote-path-under-~/proj> [dest]}"
DEST="${2:-.}"
case "$REMOTE_PATH" in
  /*|*..*) echo "remote path 必须是 ~/proj 下的相对路径且不含 .." >&2; exit 2 ;;
esac

BUCKET=hero-s1-transfer
CHUNK_MB=64
UP_PAR=4
DOWN_PAR=6
WRANGLER="npx -y wrangler@4"
RELAY_CFG=infra/r2-relay/wrangler.toml

SESS="xfer-$(date +%Y%m%d-%H%M%S)-$RANDOM"
TOKEN="$(openssl rand -hex 24)"
WORK="$(mktemp -d)"
CHUNKS_FILE="$WORK/chunks.txt"
UPLOADED=0

cleanup() {
  status=$?
  set +e
  if [ "$UPLOADED" = 1 ] && [ -s "$CHUNKS_FILE" ]; then
    echo "[cleanup] 删除桶内会话对象 …"
    while read -r f; do
      $WRANGLER r2 object delete "$BUCKET/$SESS/$f" --remote >/dev/null 2>&1
    done < "$CHUNKS_FILE"
  fi
  echo "[cleanup] 作废上传 token …"
  $WRANGLER deploy --config "$RELAY_CFG" \
    --var "UPLOAD_TOKEN:disabled-$(openssl rand -hex 8)" >/dev/null 2>&1
  ssh spark "rm -rf /tmp/$SESS" >/dev/null 2>&1
  rm -rf "$WORK"
  exit $status
}
trap cleanup EXIT

echo "[1/6] 部署 Worker(一次性 token)…"
$WRANGLER deploy --config "$RELAY_CFG" --var "UPLOAD_TOKEN:$TOKEN" \
  > "$WORK/deploy.log" 2>&1
RELAY_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.workers\.dev' "$WORK/deploy.log" | head -1)
[ -n "$RELAY_URL" ] || { cat "$WORK/deploy.log" >&2; echo "无法解析 Worker URL" >&2; exit 1; }
echo "  relay: $RELAY_URL"

echo "[2/6] 节点连通性探测 …"
if ! ssh spark "curl -sf --max-time 25 '$RELAY_URL/health'" | grep -q ok; then
  echo "spark 无法访问 $RELAY_URL(workers.dev 可能被阻断)" >&2
  exit 1
fi

echo "[3/6] 节点打包分块($REMOTE_PATH,zstd -3,${CHUNK_MB}MiB/块)…"
ssh spark "set -e
  rm -rf /tmp/$SESS && mkdir -p /tmp/$SESS
  cd ~/proj && tar -cf - '$REMOTE_PATH' | zstd -T0 -3 -q -o /tmp/$SESS/payload.tzst
  cd /tmp/$SESS && split -b ${CHUNK_MB}m -d -a 3 payload.tzst c. && rm payload.tzst
  sha256sum c.* > SHA256SUMS && cat SHA256SUMS" > "$WORK/SHA256SUMS"
awk '{print $2}' "$WORK/SHA256SUMS" > "$CHUNKS_FILE"
n_chunks=$(wc -l < "$CHUNKS_FILE" | tr -d ' ')
[ "$n_chunks" -ge 1 ] || { echo "打包产出为空" >&2; exit 1; }
echo "  $n_chunks 块"

echo "[4/6] 节点并发上传(xargs -P $UP_PAR,curl 重试)…"
UPLOADED=1
ssh spark "cd /tmp/$SESS && ls c.* | xargs -P $UP_PAR -I{} \
  curl -sf --retry 5 --retry-all-errors --max-time 1800 \
    -T {} '$RELAY_URL/up/$TOKEN/$SESS/{}' -o /dev/null \
  && echo UPLOAD_OK" | grep -q UPLOAD_OK

echo "[5/6] Mac 并发拉回 + 校验重组 …"
xargs -P "$DOWN_PAR" -I{} sh -c \
  "npx -y wrangler@4 r2 object get '$BUCKET/$SESS/{}' --file '$WORK/{}' --remote >/dev/null 2>&1" \
  < "$CHUNKS_FILE"
(cd "$WORK" && shasum -a 256 -c SHA256SUMS --quiet)
mkdir -p "$DEST"
sort "$CHUNKS_FILE" | (cd "$WORK" && xargs cat) | zstd -d | tar -x -C "$DEST"

echo "[6/6] 校验通过,已解包到 $DEST/$(basename "$REMOTE_PATH")(清理由 trap 完成)"
echo "pull_results_r2: OK ($REMOTE_PATH, $n_chunks 块)"

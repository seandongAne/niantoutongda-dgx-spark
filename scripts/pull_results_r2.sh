#!/usr/bin/env bash
# 大体积产出回程通道 v2(预签名):spark →(curl PUT 预签名 URL)→ R2 →(curl GET)→ Mac。
#
# 背景:赛方禁止 SSH/rsync 传大文件;spark 侧 workers.dev 被 DNS 污染+SNI 阻断,
# S3 端点 r2.cloudflarestorage.com 实测可达,故弃 Worker 改预签名 URL。
# 凭据纪律:R2 S3 密钥只在 Mac 本地 .env;spark 只见"限时(默认 2h)单对象
# 预签名 URL"——无凭据落节点、无常驻端点、URL 到期自动失效。
#
# 用法:
#   scripts/pull_results_r2.sh <remote-path-under-~/proj> [local-dest-dir]
# 例:
#   scripts/pull_results_r2.sh results/hero_s1/reid-w2 results/hero_s1/
# 注意:>200MB 的传输请以后台方式发射本脚本并轮询输出,不要挂前台等。
#
# 脚本被硬杀后的残留(URL 会自行过期,无凭据风险):
#   - 桶内 xfer-<SESS>/ 分块:cleanup 尽力删;彻底兜底靠桶生命周期规则
#     xfer-ttl(xfer- 前缀 1 天自动过期,2026-07-18 已设置)
#   - spark ~/.xfer/<SESS>:下次运行的 6 小时时效清扫会带走
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE_PATH="${1:?usage: pull_results_r2.sh <remote-path-under-~/proj> [dest]}"
DEST="${2:-.}"
case "$REMOTE_PATH" in
  /*|*..*) echo "remote path 必须是 ~/proj 下的相对路径且不含 .." >&2; exit 2 ;;
esac

BUCKET=hero-s1-transfer
CHUNK_MB="${PULL_R2_CHUNK_MB:-64}"   # 测试可用小值强制多块路径
URL_TTL=7200
UP_PAR=4
DOWN_PAR=6
PRESIGN="python3 scripts/r2_presign.py"

SESS="xfer-$(date +%Y%m%d-%H%M%S)-$(openssl rand -hex 4)"
WORK="$(mktemp -d)"
CHUNKS_FILE="$WORK/chunks.txt"
UPLOADED=0

cleanup() {
  status=$?
  set +e
  if [ "$UPLOADED" = 1 ] && [ -s "$CHUNKS_FILE" ]; then
    echo "[cleanup] 删除桶内会话对象 …"
    LEFT=""
    while read -r f; do
      url=$(sed "s|^|$SESS/|" <<< "$f" | $PRESIGN delete --bucket "$BUCKET" --expires 600 2>/dev/null | head -1)
      curl -sf --max-time 60 -X DELETE "$url" -o /dev/null \
        || curl -sf --max-time 60 -X DELETE "$url" -o /dev/null \
        || LEFT="$LEFT $SESS/$f"
    done < "$CHUNKS_FILE"
    if [ -n "$LEFT" ]; then
      echo "⚠️ [cleanup] 以下对象删除失败,留待桶生命周期规则过期回收:$LEFT" >&2
    fi
  fi
  ssh spark "rm -rf ~/.xfer/$SESS" >/dev/null 2>&1
  rm -rf "$WORK"
  exit $status
}
trap cleanup EXIT

echo "[1/6] 凭据与节点连通性检查(会话 $SESS)…"
$PRESIGN get --bucket "$BUCKET" --key "probe" --expires 60 >/dev/null  # 凭据缺失即 fail
ACCOUNT_ID=$(grep -E '^R2_ACCOUNT_ID=' .env | head -1 | cut -d= -f2 | tr -d '"'"'" )
S3_HOST="$ACCOUNT_ID.r2.cloudflarestorage.com"
probe_code=$(ssh spark "curl -s --max-time 25 -o /dev/null -w '%{http_code}' https://$S3_HOST/" || echo 000)
case "$probe_code" in
  400|403) echo "  S3 端点可达($probe_code)" ;;
  *) echo "spark 无法访问 $S3_HOST(HTTP $probe_code)" >&2; exit 1 ;;
esac

echo "[2/6] 节点打包分块($REMOTE_PATH,zstd -3 流式,${CHUNK_MB}MiB/块,暂存 ~/.xfer)…"
ssh spark "set -e -o pipefail
  mkdir -p ~/.xfer
  find ~/.xfer -maxdepth 1 -name 'xfer-*' -mmin +360 -exec rm -rf {} + 2>/dev/null || true
  rm -rf ~/.xfer/$SESS && mkdir -p ~/.xfer/$SESS
  cd ~/proj && tar -cf - '$REMOTE_PATH' | zstd -T0 -3 -q \
    | (cd ~/.xfer/$SESS && split -b ${CHUNK_MB}m -d -a 3 - c.)
  cd ~/.xfer/$SESS && sha256sum c.* > SHA256SUMS && cat SHA256SUMS" > "$WORK/SHA256SUMS"
awk '{print $2}' "$WORK/SHA256SUMS" > "$CHUNKS_FILE"
n_chunks=$(wc -l < "$CHUNKS_FILE" | tr -d ' ')
[ "$n_chunks" -ge 1 ] || { echo "打包产出为空" >&2; exit 1; }
echo "  $n_chunks 块"

echo "[3/6] 生成预签名 PUT URL 并下发节点 …"
sed "s|^|$SESS/|" "$CHUNKS_FILE" | $PRESIGN put --bucket "$BUCKET" --expires "$URL_TTL" > "$WORK/put_urls.txt"
paste "$CHUNKS_FILE" "$WORK/put_urls.txt" | ssh spark "cat > ~/.xfer/$SESS/urls.txt"

echo "[4/6] 节点并发上传(xargs -P $UP_PAR,curl 重试)…"
UPLOADED=1
up_out=$(ssh spark "cd ~/.xfer/$SESS && tr '\t' '\n' < urls.txt | xargs -P $UP_PAR -n 2 \
  sh -c 'curl -sfS --retry 5 --max-time 1800 -T \"\$0\" \"\$1\" -o /dev/null' \
  && echo UPLOAD_OK:\$(wc -l < urls.txt)") || true
up_n=$(printf '%s\n' "$up_out" | grep -oE 'UPLOAD_OK: *[0-9]+' | grep -oE '[0-9]+' || true)
if [ "${up_n:-0}" -ne "$n_chunks" ]; then
  printf '%s\n' "$up_out" >&2
  echo "上传未确认($n_chunks 块,确认 ${up_n:-0})" >&2
  exit 1
fi

echo "[5/6] Mac 并发拉回(每块 3 次退避重试)+ 校验重组 …"
sed "s|^|$SESS/|" "$CHUNKS_FILE" | $PRESIGN get --bucket "$BUCKET" --expires "$URL_TTL" > "$WORK/get_urls.txt"
export WORK
paste "$CHUNKS_FILE" "$WORK/get_urls.txt" | tr '\t' '\n' | xargs -P "$DOWN_PAR" -n 2 sh -c '
  for i in 1 2 3; do
    curl -sfS --retry 3 --max-time 1800 -o "$WORK/$0" "$1" 2>"$WORK/$0.err" && exit 0
    sleep $((i*10))
  done
  echo "[down-fail] $0: $(tail -1 "$WORK/$0.err" 2>/dev/null)" >&2
  exit 1'
(cd "$WORK" && shasum -a 256 -c SHA256SUMS --quiet)
mkdir -p "$DEST"
sort "$CHUNKS_FILE" | (cd "$WORK" && xargs cat) | zstd -d | tar -x -C "$DEST"

echo "[6/6] 校验通过,已解包到 $DEST/$REMOTE_PATH(清理由 trap 完成)"
echo "pull_results_r2: OK ($REMOTE_PATH, $n_chunks 块)"

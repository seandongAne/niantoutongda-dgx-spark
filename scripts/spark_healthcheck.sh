#!/usr/bin/env bash
# spark 感染回弹自检 —— 每次会话首次要用 spark 前先跑此脚本。
# 背景:远程 spark 曾被 Koske 挖矿木马入侵(经裸奔的公网 Jupyter 进入),
#       2026-07-12 已清除并封堵入口。详见项目记忆 spark-compromised.md。
#
# 主办方端口映射规则(机器号 N):
#   公网 60NN -> 内网 22    (SSH)
#   公网 70NN -> 内网 7000  (应用)
#   公网 80NN -> 内网 8888  (Jupyter)
#   公网 90NN -> 内网 9000  (ComfyUI)
# 当前 spark-72:6072->22,7072->7000,8072->8888,9072->9000。
# NAT 后的公网端口不会出现在远端 ss 输出中;须检查内网 8888/9000 是否
# 监听在非 loopback 地址。无法仅靠监听状态可靠证明认证已开启,故门禁从严告警。
#
# 覆盖:已知矿工/投放器进程、Koske rootkit/隐藏 PID/文件、shell 钩子、
# 用户 cron、GNOME autostart/systemd user/系统级持久化、不安全 Jupyter unit、
# 已知矿池端口、敏感映射端口和异常 CPU 负载。
# 用法:./scripts/spark_healthcheck.sh    退出码:0=未发现已知 IOC 1=疑似回弹 2=自检未完成
set -uo pipefail

readonly SPARK_NUMBER=72
readonly PUBLIC_JUPYTER_PORT=$((8000 + SPARK_NUMBER))
readonly PUBLIC_COMFYUI_PORT=$((9000 + SPARK_NUMBER))
readonly RETRY_DELAY_SECONDS="${SPARK_HEALTHCHECK_RETRY_DELAY_SECONDS:-1}"

SSH=(
  ssh
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
  spark
)

remote_probe() {
  "${SSH[@]}" '
    fail() {
      printf "ERROR|%s\n" "$1"
      exit 3
    }

    for cmd in ps grep crontab ss awk; do
      command -v "$cmd" >/dev/null 2>&1 || fail "missing-command:$cmd"
    done

    processes=$(ps -eo comm=,args= 2>/dev/null) || fail "ps"
    drop=$(printf "%s\n" "$processes" | grep -cE "[p]anda_v14|[k]0ske6" || true)
    miner=$(printf "%s\n" "$processes" | grep -cE "^[[:space:]]*(xmrigARM|xmr)([[:space:]]|$)" || true)

    files=0
    for path in \
      "$HOME/.bashrc.koske" \
      "$HOME/koske" \
      "$HOME/xmr" \
      "$HOME/xmrig1" \
      "$HOME/build_a_claw_workshop-bundle/xmrig1" \
      "/dev/shm/hideproc.so" \
      "/dev/shm/.hiddenpid"; do
      if [ -e "$path" ] || [ -L "$path" ]; then
        files=$((files + 1))
      fi
    done

    hook=0
    for file in \
      "$HOME/.bashrc" \
      "$HOME/.bash_logout" \
      "$HOME/.profile" \
      "$HOME/.bash_profile" \
      "$HOME/.zshrc" \
      "$HOME/.zprofile" \
      "$HOME/.config/fish/config.fish"; do
      [ -e "$file" ] || continue
      [ -r "$file" ] || fail "unreadable-shell-hook"
      grep -qE "bashrc\.koske|[k]oske|[k]0ske|[p]anda_v14|hideproc|LD_PRELOAD" "$file"
      grep_status=$?
      case "$grep_status" in
        0) hook=$((hook + 1)) ;;
        1) ;;
        *) fail "grep-shell-hook" ;;
      esac
    done

    cron_text=$(crontab -l 2>&1)
    cron_status=$?
    case "$cron_status" in
      0)
        cron=$(printf "%s\n" "$cron_text" | grep -cE "[k]oske|[k]0ske|[p]anda_v14|xmr|hideproc|LD_PRELOAD" || true)
        ;;
      *)
        printf "%s\n" "$cron_text" | grep -qi "no crontab for" || fail "crontab"
        cron=0
        ;;
    esac

    persistence=0
    for dir in "$HOME/.config/autostart" "$HOME/.config/systemd/user"; do
      [ -d "$dir" ] || continue
      matches=$(grep -rlE "[k]oske|[k]0ske|[p]anda_v14|xmrig|kryptex|hideproc|LD_PRELOAD" "$dir" 2>/dev/null)
      grep_status=$?
      case "$grep_status" in
        0)
          count=$(printf "%s\n" "$matches" | awk "NF { n++ } END { print n + 0 }")
          persistence=$((persistence + count))
          ;;
        1) ;;
        *) fail "grep-persistence" ;;
      esac
    done

    system_persistence=0
    for file in \
      /etc/ld.so.preload \
      /etc/rc.local \
      /etc/systemd/system/shellkoske.service \
      /usr/lib/systemd/system/shellkoske.service; do
      [ -e "$file" ] || continue
      [ -r "$file" ] || fail "unreadable-system-persistence"
      grep -qE "[k]oske|[k]0ske|[p]anda_v14|xmrig|kryptex|hideproc|LD_PRELOAD" "$file"
      grep_status=$?
      case "$grep_status" in
        0) system_persistence=$((system_persistence + 1)) ;;
        1) ;;
        *) fail "grep-system-persistence" ;;
      esac
    done

    loaded_rootkit=0
    for maps in /proc/[0-9]*/maps; do
      grep -qF "/dev/shm/hideproc.so" "$maps" 2>/dev/null || continue
      loaded_rootkit=$((loaded_rootkit + 1))
    done

    unsafe_jupyter_unit=0
    jupyter_unit="$HOME/.config/systemd/user/jupyter-workshop.service"
    if [ -f "$jupyter_unit" ]; then
      [ -r "$jupyter_unit" ] || fail "unreadable-jupyter-unit"
      if grep -q -- "--ip=0.0.0.0" "$jupyter_unit" \
        && grep -q -- "--port=8888" "$jupyter_unit"; then
        unsafe_jupyter_unit=1
      fi
    fi

    listeners=$(ss -H -ltn 2>/dev/null) || fail "ss-listeners"
    count_exposed() {
      printf "%s\n" "$listeners" | awk -v port="$1" '\''
        $4 ~ (":" port "$") &&
        $4 != ("127.0.0.1:" port) &&
        $4 != ("[::1]:" port) { count++ }
        END { print count + 0 }
      '\''
    }
    jupyter_exposed=$(count_exposed 8888) || fail "parse-jupyter-listener"
    comfyui_exposed=$(count_exposed 9000) || fail "parse-comfyui-listener"

    connections=$(ss -H -tn 2>/dev/null) || fail "ss-connections"
    pool=$(printf "%s\n" "$connections" | grep -cE "(^|[[:space:]])[^[:space:]]*:7029([[:space:]]|$)" || true)

    [ -r /proc/loadavg ] || fail "loadavg-unreadable"
    IFS=" " read -r load _ </proc/loadavg || fail "loadavg-read"

    for value in "$drop" "$miner" "$files" "$hook" "$cron" "$persistence" \
      "$system_persistence" "$loaded_rootkit" "$unsafe_jupyter_unit" \
      "$jupyter_exposed" "$comfyui_exposed" "$pool"; do
      case "$value" in
        ""|*[!0-9]*) fail "invalid-count" ;;
      esac
    done
    case "$load" in
      ""|*[!0-9.]*) fail "invalid-load" ;;
    esac

    printf "OK|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n" \
      "$drop" "$miner" "$files" "$hook" "$cron" "$persistence" \
      "$system_persistence" "$loaded_rootkit" "$unsafe_jupyter_unit" \
      "$jupyter_exposed" "$comfyui_exposed" "$pool" "$load"
  '
}

OUT=""
ssh_status=255
for attempt in 1 2; do
  OUT=$(remote_probe)
  ssh_status=$?
  [ "$ssh_status" -eq 0 ] && break

  if [ "$ssh_status" -eq 255 ] && [ "$attempt" -eq 1 ]; then
    echo "⚠️  spark SSH 首次失败,按项目规则重试一次..." >&2
    sleep "$RETRY_DELAY_SECONDS"
    continue
  fi
  break
done

if [ "$ssh_status" -ne 0 ]; then
  if [ "$ssh_status" -eq 255 ]; then
    echo "⚠️  两次均连不上 spark(网络、SSH 或 OOM),自检未完成"
  else
    detail="${OUT#ERROR|}"
    [ "$detail" = "$OUT" ] && detail="remote-exit-${ssh_status}"
    echo "⚠️  spark 已连接,但远端检查失败(${detail}),自检未完成"
  fi
  exit 2
fi

case "$OUT" in
  *$'\n'*)
    echo "⚠️  spark 自检输出包含额外行,拒绝判定为干净"
    exit 2
    ;;
esac

IFS="|" read -r status drop miner files hook cron persistence \
  system_persistence loaded_rootkit unsafe_jupyter_unit \
  jupyter_exposed comfyui_exposed pool load extra <<<"$OUT"

if [ "$status" != "OK" ] || [ -n "${extra:-}" ]; then
  echo "⚠️  spark 自检输出格式异常,拒绝判定为干净"
  exit 2
fi

for value in "$drop" "$miner" "$files" "$hook" "$cron" "$persistence" \
  "$system_persistence" "$loaded_rootkit" "$unsafe_jupyter_unit" \
  "$jupyter_exposed" "$comfyui_exposed" "$pool"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "⚠️  spark 自检计数字段异常,拒绝判定为干净"
    exit 2
  fi
done
if [[ ! "$load" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "⚠️  spark 自检负载字段异常,拒绝判定为干净"
  exit 2
fi

problems=()
[ "$drop"              -gt 0 ] && problems+=("dropper 死循环进程×${drop}")
[ "$miner"             -gt 0 ] && problems+=("已知矿工进程×${miner}")
[ "$files"             -gt 0 ] && problems+=("矿工/Koske 文件×${files}")
[ "$hook"              -gt 0 ] && problems+=("shell 钩子×${hook}")
[ "$cron"              -gt 0 ] && problems+=("恶意 cron×${cron}")
[ "$persistence"       -gt 0 ] && problems+=("autostart/systemd user 持久化×${persistence}")
[ "$system_persistence" -gt 0 ] && problems+=("系统级 Koske 持久化×${system_persistence}")
[ "$loaded_rootkit"    -gt 0 ] && problems+=("已加载 hideproc rootkit×${loaded_rootkit}")
[ "$unsafe_jupyter_unit" -gt 0 ] && problems+=("休眠的不安全 Jupyter unit×${unsafe_jupyter_unit}")
[ "$jupyter_exposed"   -gt 0 ] && problems+=("Jupyter 内网 8888 非 loopback 监听×${jupyter_exposed}(公网 ${PUBLIC_JUPYTER_PORT})")
[ "$comfyui_exposed"   -gt 0 ] && problems+=("ComfyUI 内网 9000 非 loopback 监听×${comfyui_exposed}(公网 ${PUBLIC_COMFYUI_PORT})")
[ "$pool"              -gt 0 ] && problems+=("已知矿池端口 7029 连接×${pool}")

loadnote=""
awk "BEGIN { exit !(${load} > 40) }" 2>/dev/null \
  && loadnote="  ⚠ 高 CPU 负载 load=${load}(留意)"

if [ ${#problems[@]} -eq 0 ]; then
  echo "✅ SPARK CLEAN (未发现已知 IOC; load=${load}; 8888→${PUBLIC_JUPYTER_PORT},9000→${PUBLIC_COMFYUI_PORT} 未暴露)${loadnote}"
  exit 0
fi

echo "⚠️  疑似回弹/危险暴露: ${problems[*]}${loadnote}"
echo "→ 停下,向用户汇报;按项目记忆 spark-compromised.md 重新清理并考虑报障主办方。"
exit 1

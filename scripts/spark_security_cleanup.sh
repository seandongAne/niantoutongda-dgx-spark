#!/usr/bin/env bash
# Koske rootkit/root persistence audit and precise cleanup for spark-48f0.
# Run on Spark only, as root: sudo ./scripts/spark_security_cleanup.sh audit|clean|verify
set -Eeuo pipefail

readonly EXPECTED_HOST="spark-48f0"
readonly TARGET_USER="Developer"
readonly TARGET_HOME="/home/Developer"
readonly IOC_REGEX='[k]oske|[k]0ske|[p]anda_v14|xmrig|kryptex|hideproc|LD_PRELOAD'

KNOWN_PATHS=(
  "$TARGET_HOME/.bashrc.koske"
  "$TARGET_HOME/koske"
  "$TARGET_HOME/xmr"
  "$TARGET_HOME/xmrig1"
  "$TARGET_HOME/build_a_claw_workshop-bundle/xmrig1"
  "$TARGET_HOME/.config/autostart/scheduler.desktop"
  "/dev/shm/hideproc.so"
  "/dev/shm/.hiddenpid"
  "/etc/systemd/system/shellkoske.service"
  "/usr/lib/systemd/system/shellkoske.service"
)

die() {
  echo "ERROR: $*" >&2
  exit 2
}

section() {
  printf '\n[%s]\n' "$1"
}

assert_context() {
  [ "$(id -u)" -eq 0 ] || die "must run as root"
  [ "$(hostname)" = "$EXPECTED_HOST" ] || die "refusing unexpected host: $(hostname)"
  id "$TARGET_USER" >/dev/null 2>&1 || die "missing user: $TARGET_USER"
  [ -d "$TARGET_HOME" ] || die "missing home: $TARGET_HOME"
}

process_hits() {
  ps -eo pid=,user=,comm=,args= 2>/dev/null \
    | grep -Ei '[p]anda_v14|[k]0ske6|[x]mrigARM|[k]ryptex|^[[:space:]]*[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+xmr([[:space:]]|$)' \
    || true
}

loaded_rootkit_maps() {
  local maps
  for maps in /proc/[0-9]*/maps; do
    grep -qF '/dev/shm/hideproc.so' "$maps" 2>/dev/null || continue
    printf '%s\n' "$maps"
  done
}

known_path_hits() {
  local path
  for path in "${KNOWN_PATHS[@]}"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      stat -c '%A %U:%G %s %y %n' "$path" 2>/dev/null || printf '%s\n' "$path"
    fi
  done
}

named_ioc_hits() {
  find "$TARGET_HOME" /root /tmp /var/tmp /dev/shm /etc /usr/local /opt \
    -xdev \
    \( -path "$TARGET_HOME/proj" -o -path "$TARGET_HOME/.cache" \) -prune -o \
    \( -iname '*koske*' -o -iname '*k0ske*' -o -iname '*panda_v14*' \
       -o -iname '*xmrig*' -o -iname '*kryptex*' -o -iname '*hideproc*' \
       -o -name '.hiddenpid' \) \
    -printf '%y %m %u:%g %s %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null \
    || true
}

config_ioc_hits() {
  local paths=()
  local path
  for path in \
    /etc/ld.so.preload \
    /etc/rc.local \
    /etc/cron.d \
    /etc/cron.daily \
    /etc/systemd/system \
    /usr/lib/systemd/system \
    "$TARGET_HOME/.bashrc" \
    "$TARGET_HOME/.bash_logout" \
    "$TARGET_HOME/.profile" \
    "$TARGET_HOME/.bash_profile" \
    "$TARGET_HOME/.zshrc" \
    "$TARGET_HOME/.zprofile" \
    "$TARGET_HOME/.config/fish/config.fish" \
    "$TARGET_HOME/.config/autostart" \
    "$TARGET_HOME/.config/systemd/user" \
    /root/.bashrc \
    /root/.profile; do
    [ -e "$path" ] && paths+=("$path")
  done
  [ ${#paths[@]} -gt 0 ] || return 0
  grep -IlER "$IOC_REGEX" "${paths[@]}" 2>/dev/null || true
}

cron_ioc_hits() {
  local user text
  for user in "$TARGET_USER" root; do
    text=$(crontab -u "$user" -l 2>/dev/null || true)
    [ -n "$text" ] || continue
    printf '%s\n' "$text" | grep -Ei "$IOC_REGEX" \
      | sed "s/^/$user: /" || true
  done
}

unsafe_jupyter_unit() {
  local unit="$TARGET_HOME/.config/systemd/user/jupyter-workshop.service"
  [ -f "$unit" ] || return 1
  grep -q -- '--ip=0.0.0.0' "$unit" \
    && grep -q -- '--port=8888' "$unit"
}

audit() {
  local output
  section "IDENTITY"
  printf 'host=%s root_uid=%s target_user=%s utc=%s\n' \
    "$(hostname)" "$(id -u)" "$TARGET_USER" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uptime
  free -h

  section "KNOWN_PATHS"
  output=$(known_path_hits)
  [ -n "$output" ] && printf '%s\n' "$output" || echo "none"

  section "NAMED_IOCS"
  output=$(named_ioc_hits)
  [ -n "$output" ] && printf '%s\n' "$output" || echo "none"

  section "PROCESS_IOCS"
  output=$(process_hits)
  [ -n "$output" ] && printf '%s\n' "$output" || echo "none"

  section "LOADED_ROOTKIT_MAPS"
  output=$(loaded_rootkit_maps)
  [ -n "$output" ] && printf '%s\n' "$output" || echo "none"

  section "CONFIG_IOCS"
  output=$(config_ioc_hits)
  [ -n "$output" ] && printf '%s\n' "$output" || echo "none"

  section "CRON_IOCS"
  output=$(cron_ioc_hits)
  [ -n "$output" ] && printf '%s\n' "$output" || echo "none"

  section "JUPYTER_UNIT"
  if unsafe_jupyter_unit; then
    printf 'unsafe_unit=present enabled=%s active=%s\n' \
      "$(user_systemctl is-enabled jupyter-workshop.service 2>/dev/null || true)" \
      "$(user_systemctl is-active jupyter-workshop.service 2>/dev/null || true)"
  else
    echo "unsafe_unit=absent"
  fi

  section "DOCKER"
  if command -v docker >/dev/null 2>&1; then
    docker ps --no-trunc --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}' \
      2>/dev/null || echo "docker-query-failed"
  else
    echo "docker-not-installed"
  fi

  section "NETWORK"
  ss -lntup 2>/dev/null || true
  ss -H -tnp state established 2>/dev/null || true

  section "DNS"
  printf 'resolv_link=%s\n' "$(readlink -f /etc/resolv.conf 2>/dev/null || true)"
  lsattr /etc/resolv.conf 2>/dev/null || true
  grep -E '^[[:space:]]*nameserver[[:space:]]+' /etc/resolv.conf 2>/dev/null || true
}

user_systemctl() {
  local uid
  uid=$(id -u "$TARGET_USER")
  runuser -u "$TARGET_USER" -- env \
    XDG_RUNTIME_DIR="/run/user/$uid" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus" \
    systemctl --user "$@"
}

stop_loaded_rootkit_processes() {
  local maps pid
  while IFS= read -r maps; do
    [ -n "$maps" ] || continue
    pid=$(printf '%s\n' "$maps" | cut -d/ -f3)
    case "$pid" in
      ''|*[!0-9]*) die "invalid PID from maps: $maps" ;;
      1|2) die "rootkit loaded into critical PID $pid; reboot/reimage required" ;;
    esac
    echo "terminating process with hideproc loaded: pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$pid" 2>/dev/null || true
  done < <(loaded_rootkit_maps)
}

stop_known_processes() {
  local pid
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    [ "$pid" -gt 2 ] || die "refusing to terminate critical PID $pid"
    echo "terminating known IOC process: pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$pid" 2>/dev/null || true
  done < <(process_hits | awk '{print $1}')
}

remove_matching_user_persistence_files() {
  local dir file
  for dir in "$TARGET_HOME/.config/autostart" "$TARGET_HOME/.config/systemd/user"; do
    [ -d "$dir" ] || continue
    while IFS= read -r file; do
      [ -n "$file" ] || continue
      echo "removing IOC persistence file: $file"
      rm -f -- "$file"
    done < <(grep -IlER "$IOC_REGEX" "$dir" 2>/dev/null || true)
  done
}

clean_shell_hooks() {
  local file
  for file in \
    "$TARGET_HOME/.bashrc" \
    "$TARGET_HOME/.bash_logout" \
    "$TARGET_HOME/.profile" \
    "$TARGET_HOME/.bash_profile" \
    "$TARGET_HOME/.zshrc" \
    "$TARGET_HOME/.zprofile" \
    "$TARGET_HOME/.config/fish/config.fish"; do
    [ -f "$file" ] || continue
    grep -qE "$IOC_REGEX" "$file" 2>/dev/null || continue
    echo "removing IOC lines from: $file"
    sed -i -E "/$IOC_REGEX/Id" "$file"
    chown "$TARGET_USER:$TARGET_USER" "$file"
  done
}

clean_crontab() {
  local user current cleaned
  user="$1"
  current=$(mktemp)
  cleaned=$(mktemp)
  crontab -u "$user" -l >"$current" 2>/dev/null || true
  grep -Eiv "$IOC_REGEX" "$current" >"$cleaned" || true
  if [ -s "$cleaned" ]; then
    crontab -u "$user" "$cleaned"
  else
    crontab -u "$user" -r 2>/dev/null || true
  fi
  rm -f -- "$current" "$cleaned"
}

clean_system_file_lines() {
  local file tmp
  file="$1"
  [ -f "$file" ] || return 0
  grep -qEi "$IOC_REGEX" "$file" 2>/dev/null || return 0
  echo "removing IOC lines from system file: $file"
  tmp=$(mktemp)
  grep -Eiv "$IOC_REGEX" "$file" >"$tmp" || true
  if grep -qEv '^[[:space:]]*($|#)' "$tmp"; then
    cat "$tmp" >"$file"
  else
    rm -f -- "$file"
  fi
  rm -f -- "$tmp"
}

clean() {
  local path
  section "PRE_CLEAN_AUDIT"
  audit

  section "CLEAN_ACTIONS"
  stop_loaded_rootkit_processes
  stop_known_processes

  user_systemctl disable --now jupyter-workshop.service >/dev/null 2>&1 || true
  rm -f -- "$TARGET_HOME/.config/systemd/user/jupyter-workshop.service"
  user_systemctl daemon-reload >/dev/null 2>&1 || true

  for path in "${KNOWN_PATHS[@]}"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      echo "removing known IOC path: $path"
      rm -rf -- "$path"
    fi
  done

  remove_matching_user_persistence_files
  clean_shell_hooks
  clean_crontab "$TARGET_USER"
  clean_crontab root

  systemctl disable --now shellkoske.service >/dev/null 2>&1 || true
  rm -f -- /etc/systemd/system/shellkoske.service /usr/lib/systemd/system/shellkoske.service
  clean_system_file_lines /etc/rc.local
  clean_system_file_lines /etc/ld.so.preload
  systemctl daemon-reload >/dev/null 2>&1 || true

  section "POST_CLEAN_VERIFY"
  verify
}

verify() {
  local failures=0 output

  output=$(known_path_hits)
  if [ -n "$output" ]; then
    printf 'VERIFY_FAIL known paths:\n%s\n' "$output"
    failures=$((failures + 1))
  fi

  output=$(named_ioc_hits)
  if [ -n "$output" ]; then
    printf 'VERIFY_FAIL named IOCs:\n%s\n' "$output"
    failures=$((failures + 1))
  fi

  output=$(process_hits)
  if [ -n "$output" ]; then
    printf 'VERIFY_FAIL process IOCs:\n%s\n' "$output"
    failures=$((failures + 1))
  fi

  output=$(loaded_rootkit_maps)
  if [ -n "$output" ]; then
    printf 'VERIFY_FAIL loaded rootkit maps:\n%s\n' "$output"
    failures=$((failures + 1))
  fi

  output=$(config_ioc_hits)
  if [ -n "$output" ]; then
    printf 'VERIFY_FAIL config IOCs:\n%s\n' "$output"
    failures=$((failures + 1))
  fi

  output=$(cron_ioc_hits)
  if [ -n "$output" ]; then
    printf 'VERIFY_FAIL cron IOCs:\n%s\n' "$output"
    failures=$((failures + 1))
  fi

  if unsafe_jupyter_unit; then
    echo "VERIFY_FAIL unsafe Jupyter unit remains"
    failures=$((failures + 1))
  fi

  if ss -H -ltn 2>/dev/null | awk '$4 ~ /:(8888|9000)$/ && $4 !~ /^127\.0\.0\.1:/ && $4 !~ /^\[::1\]:/ { found=1 } END { exit !found }'; then
    echo "VERIFY_FAIL sensitive port exposed"
    failures=$((failures + 1))
  fi

  if [ "$failures" -eq 0 ]; then
    echo "✅ ROOT KOSKE SWEEP CLEAN"
    return 0
  fi

  echo "ROOT VERIFY FAILED: groups=$failures" >&2
  return 1
}

main() {
  assert_context
  case "${1:-}" in
    audit) audit ;;
    clean) clean ;;
    verify) verify ;;
    *) die "usage: $0 audit|clean|verify" ;;
  esac
}

main "$@"

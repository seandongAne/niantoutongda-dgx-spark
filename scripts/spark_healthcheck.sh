#!/usr/bin/env bash
# spark 感染回弹自检 —— 每次会话首次要用 spark 前先跑此脚本。
# 背景:远程 spark 曾被 Koske 挖矿木马入侵(经裸奔的公网 Jupyter 进入),
#       2026-07-12 已清除并封堵入口。详见项目记忆 spark-compromised.md。
# 用法:./scripts/spark_healthcheck.sh
# 退出码:0=干净  1=疑似回弹  2=连不上(自检未完成)
set -uo pipefail

OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=15 spark '
  miner=$(ps -eo comm= 2>/dev/null | grep -cxE "xmrigARM|xmr")
  files=$(ls -d ~/.bashrc.koske ~/xmr ~/xmrig1 ~/build_a_claw_workshop-bundle/xmrig1 2>/dev/null | wc -l)
  hook=$(grep -lE "bashrc\.koske" ~/.bashrc ~/.bash_logout ~/.profile ~/.bash_profile 2>/dev/null | wc -l)
  cron=$(crontab -l 2>/dev/null | grep -cE "k0ske|panda_v14|xmr")
  jup=$(ss -tln 2>/dev/null | grep -c "0.0.0.0:8888")
  pool=$(ss -tn 2>/dev/null | grep -cE "hashvault|kryptex|:7029")
  load=$(cut -d" " -f1 /proc/loadavg 2>/dev/null)
  printf "%s|%s|%s|%s|%s|%s|%s" "$miner" "$files" "$hook" "$cron" "$jup" "$pool" "$load"
') || { echo "⚠️  连不上 spark(网络或 SSH 问题),自检未完成"; exit 2; }

IFS="|" read -r miner files hook cron jup pool load <<<"$OUT"

problems=()
[ "${miner:-0}" -gt 0 ] && problems+=("矿工进程×${miner}")
[ "${files:-0}" -gt 0 ] && problems+=("矿工/koske 文件×${files}")
[ "${hook:-0}"  -gt 0 ] && problems+=("shell 钩子×${hook}")
[ "${cron:-0}"  -gt 0 ] && problems+=("恶意 cron×${cron}")
[ "${jup:-0}"   -gt 0 ] && problems+=("Jupyter 8888 又公网监听了")
[ "${pool:-0}"  -gt 0 ] && problems+=("矿池连接×${pool}")

# 高负载为软提示(挖矿会把 CPU 打满到 load~60;GPU 训练一般不会顶高 CPU load)
loadnote=""
awk "BEGIN{exit !(${load:-0} > 40)}" 2>/dev/null && loadnote="  ⚠ 高 CPU 负载 load=${load}(留意)"

if [ ${#problems[@]} -eq 0 ]; then
  echo "✅ SPARK CLEAN (load=${load:-?})${loadnote}"
  exit 0
else
  echo "⚠️  疑似回弹: ${problems[*]}${loadnote}"
  echo "→ 停下,向用户汇报;按项目记忆 spark-compromised.md 重新清理并考虑报障主办方。"
  exit 1
fi

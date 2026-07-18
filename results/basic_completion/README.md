# 基础完赛 · 参考代码跑通留档

> 2026-07-18(D6)执行。赛方口径:基础完赛只需跑通参考代码(OpenClaw + ComfyUI
> 超级英雄照片生成 workshop,链接页即讲义网页版,附件 `workshop-Copy1(1).ipynb`)。

## 结论

**参考代码全链路跑通,零单元格错误。** 讲义 0-5 章在 spark-72 上通过
`jupyter nbconvert --execute` 无头执行一遍(约 6 分钟),随后按 5.3 手动完成
chat UI 终极测试。

| 验证点 | 结果 |
|---|---|
| 第 0 章 env-check | 通过(端口空闲、GPU/磁盘正常) |
| 第 1 章 Ollama + qwen3.6:35b | 服务起于 127.0.0.1:11434,1.3 API 对话应答正常、thinking 关闭生效 |
| 第 2 章 ComfyUI (FLUX+PuLID) | 起于 127.0.0.1:8200,v0.18.1,system_stats 就绪 |
| 第 3 章 OpenClaw 2026.5.19 | gateway 起于 loopback:3030,onboard + 模型接 Ollama 成功 |
| 第 4 章 superhero skill | 四文件写入,`openclaw skills list` 显示 `✓ ready 🦸 superhero` |
| 第 5.1 章 CLI 端到端 | `MEDIA:...superhero_00001_.png` + "✅ skill 跑通" |
| 第 5.3 章 chat UI 终测 | 浏览器发「用 superhero skill 生成超级英雄照片…」,Agent 调 skill,回复"你的超级英雄照片来啦!"并内联渲染 `superhero_00002_.png` |

## 本目录文件

- `workshop-basicrun.ipynb` — 执行完的 notebook(含全部单元格输出,0 error)
- `superhero_00001_.png` — 5.1 CLI 路径生成图(1024×1024)
- `superhero_00002_.png` — 5.3 chat UI 对话路径生成图

## 与参考代码的偏差(全部为安全/健壮性适配,已留 .bak)

赛方参考按"可信局域网"假设放开访问;我方节点在公网 NAT + 曾被入侵的黑客松内网,
按项目安全纪律全部收紧为 loopback,功能路径未动:

1. notebook 3.1:`gateway.bind` `lan`→`loopback`;移除
   `dangerouslyDisableDeviceAuth`(改走标准设备配对,CLI approve)。
2. notebook 1.3:请求超时 60→300s(qwen3.6:35b 冷加载超 60s 会误伤)。
3. `scripts/comfyui-ctl.sh`:`--listen 0.0.0.0`→`127.0.0.1`。
4. `scripts/ollama-ctl.sh`:`OLLAMA_HOST` `0.0.0.0`→`127.0.0.1`。
5. `scripts/openclaw-ctl.sh`:还原 07-12 的 StepFun env 注入为原版
   (保证跑的是本地 Ollama 而非云端模型)。

远端运行副本:`spark:~/build_a_claw_workshop-bundle/workshop-basicrun.ipynb`;
执行日志:`spark:~/proj/logs/workshop_basicrun.log`。

## 复现 / 演示访问

```bash
# Mac 上开隧道后浏览器打开(token 在远端 openclaw.json 的 gateway.auth.token)
ssh -f -N -L 3030:127.0.0.1:3030 spark
# http://localhost:3030/#token=<token>
# 首次连接需在 spark 端批准设备:cd ~/build_a_claw_workshop-bundle && ./openclaw devices approve <request-id>

# 服务管理(均在 bundle 根目录)
bash scripts/ollama-ctl.sh  start|stop|status
bash scripts/comfyui-ctl.sh start|stop|status   # 生成前需 start,~35GB 常驻,用完即停
bash scripts/openclaw-ctl.sh start|stop|status
```

注:bundle 内另有多节点公网版讲义(OpenClaw:9000→公网 9072、ComfyUI:7000→公网
7072,`NODE_SUFFIX=72` 已配好)。若赛方要求公网可访问的活演示,需用户拍板后按该
版端口重配(公网暴露与安全纪律冲突,默认不开)。

## 运行时事件

- 内存峰值:qwen3.6:35b 运行态(34.5GB)与 ComfyUI(~30GB)同驻窗口,系统可用
  一度降至 4.9GB;窗口随 skill 的 ollama 自动卸载解除。
- 该窗口连带 OOM 了发射前已在跑的 `reid_task.py`(hero_s1 reid-final,
  04:12 起,未写出任何产出即消失,旧产出未受影响);workshop 收尾停 ComfyUI
  后已用原命令重启并确认在跑。教训:今后跑 workshop/演示前先确认无内存敏感
  任务共存,或错峰执行。

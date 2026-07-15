# DAY-03 · 2026-07-15

> 赛程日：D3
>
> 当日主责：选题切换与评分对齐
>
> 状态：进行中

## 今日目标

- 记录选题从海事离线 AI 中枢切换为 AI 搬家复原 Agent（队员一致投票）。
- 将搬家方案的 v0.2 两份文档按赛事官方六项评审标准审计，并落为 v0.3 修订。
- 对应评分项：全部六项（对齐矩阵本身），重点补齐平台适配性（双生态）、演示效果（成品视频）与赛事征文（十日谈）三个此前完全缺席的载体。

## 关键证据或截图

- 完成结果：`docs/AI搬家复原Agent_产品与技术设计_v0.3.md`、`docs/AI搬家复原Agent_七日落地切片_v0.3.md`；v0.2 两份加取代标记留档。
- Commit：`3dbc4a7`（v0.2 原始文档为 `db8025f`）。
- v0.3 相对 v0.2 的实质变化：
  1. 新增 §4.4 赛事评分对齐矩阵（沿用海事线 v0.3 方法），六项标准全部有得分载体、锚点与不可裁剪标记；
  2. 补 Stepfun 双生态载体：SP0 扩为双生态探针，新增 A1 语音能力切片（Step-Audio 2 mini 旁白理解 → TTS-3B 播报 → 文本润色三级降级阶梯，0.5 天时间盒，不阻塞主链）；
  3. V1 从"90 秒演示"改为 3～4 分钟成品视频分镜（90 秒主闭环 + 断网高光 + trace 动画 + 调优对比），Day 6 冻结分镜；
  4. 新增 E1 十日谈切片与每日 DoD 检查项；
  5. 新增 EXEC→MEM `VerificationCheckRequest` 验收复核反向回路，协同 trace 单命令回放进入 S7/S8 验收（"智能体融合 25%"的协同证据从单向流水线升级为有反向请求的协同）；
  6. 七日窗口锚定日历：Day 1 = 07-15（D3）… Day 7 = 07-21（D9），07-22（D10）只留复验与提交。
- 平台事实澄清（Sean 确认）：赛方节点为 24/7 SSH 资格，持续到 2026-07-22 赛程结束；唯一已知约束是跨境网络波动。已补记入 `CLAUDE.md` / `AGENTS.md` 远程机器信息。
- A1 降级阶梯第三级修正（commit `6653b7d`）：原"Stepfun 开源文本模型润色"不成立——Stepfun 开源文本旗舰为 321B 级 MoE，128GB 统一内存不可行；第三级改为 Step-Audio 2 mini 文本输出模式生成任务卡朗读稿，并区分"质量不达标走阶梯"与"加载失败走同生态换型号 + P1 上报"两条路径。教训：降级阶梯里的每一级都要单独过桌面预审，兜底选项不能只因为"听起来最保守"就免检。
- 未验证或降级边界：v0.3 仍是纯文档；G0/SP0/A1 均未启动，Step-Audio 系列在本队 Spark 上未实测（可行性证据沿用海事线核实结论）；不得将对齐矩阵描述为已实现能力。

- **两日主链冲刺启动（Sean 拍板）**：技术先行，48 小时内目标 = 主链端到端（视频→实体→组合→区域→CP-SAT→任务卡→trace），真模型、在 Spark 上；冻结任务 B 验收、SF1 预注册、三次复跑等日历型证据流程仍按 v0.3 七日计划走。
- SP0 桌面预审（commit `3497366`）：5 个 ModelScope ID 经公开 API 核验 200（`IDEA-Research/grounding-dino-base`、`facebook/dinov2-base`、`stepfun-ai/Step-Audio-2-mini`、`Step-Audio-Tokenizer`、`Step-Audio-TTS-3B`）；两个猜测 ID 404 排除（`AI-ModelScope/grounding-dino-base`、`AI-ModelScope/dinov2-base`）。
- S0 契约核心落地（同 commit）：`backend/schemas/core.py`（10 契约对象 + 枚举，extra=forbid，未知 schema_version 拒绝）、`backend/tools/solver/layout_solver.py`（CP-SAT 硬约束 + 整数软目标 + 替代区域 + INFEASIBLE 冲突解释 + PLANNER_ERROR 区分，seed/单 worker 确定性）、`backend/tools/audit/store.py`（追加式 JSONL）；本地 pytest **14/14 全绿**。
- Spark 侧：会话首检 `✅ SPARK CLEAN`；节点为裸环境（无 torch/modelscope），`spark_bootstrap.sh min` 完成（venv + modelscope CLI）；`sp0_download.sh`（5 模型 → `~/models/`）与 `torch(cu130)+deps` 安装已 nohup 发射（logs：`~/proj/logs/sp0_download.log`、`bootstrap_torch_deps.log`），发射后 20 秒复核确认两者均在真实推进。
- NVIDIA 模型机会核实（Sean 追问触发）：`nvidia/` 命名空间在 ModelScope 全 404，真实宿主为 **`nv-community/` 镜像 org**；`NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`（→ vlm_attributes 槽位）与 `NVIDIA-Nemotron-Nano-9B-v2`（→ 任务文案润色槽位）均 API 核验 200 并已入队下载（`logs/nvidia_download.log`）；NVFP4-QAD 变体核验存在，记为量化/L2 升级路径不下载。models.yaml 两槽位已从 TBD 落到实 ID。
- 未验证或降级边界（本段新增）：所有模型尚未跑过任何推理（探针 L1/L2 明日）；NVIDIA VLM 在两日主链内仍先以规则+检测类别兜底，VL 探针通过后接入；TensorRT L2 与 TAO 微调为 NVIDIA 软件栈侧的补充载体。

## 失败与教训

- 选题切换时，海事文档里已经做过的官方评分对齐（v0.2 评分对齐补丁、双生态载体、成品视频分镜、E1）没有随选题迁移，搬家 v0.2 六项标准里三项载体完全缺席——**评分对齐矩阵必须是任何新选题文档的第一节检查项，选题可以换，赛事标准不换**。
- 搬家选题在"行业落地价值"叙事上先天弱于 ToB 深场景，已在矩阵解读中明示：靠工程深度（第 2 项）和完整性（第 3 项）补分，访谈证据必须做实。
- 十日谈汇编稿（`docs/十日谈_念头通达.md`）的章节骨架仍是海事叙事线，需随新选题改写章节注释——留待 D4 处理，不阻塞今日交付。

## 明日计划

- Day 1（切片日历）已在今日开始：G0 数据拍摄与真值冻结、SP0 双生态探针、S0 骨架三线并行；负责人按手册 §2 分工。
- SP0 必须在明日内完成 Step-Audio 2 mini 启动探针，否则 A1 降级阶梯提前评估。
- 若 G0/SP0 明日收工仍未 PASS，按手册阻断规则优先补数据/换型号，不得越过依赖开始 S1/S2。

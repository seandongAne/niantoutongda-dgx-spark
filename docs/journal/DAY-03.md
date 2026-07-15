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
- 未验证或降级边界（本段新增）：所有模型尚未跑过任何推理（探针 L1/L2 明日）；NVIDIA VLM 在两日主链内仍先以规则+检测类别兜底，VL 探针通过后接入；TensorRT 部署为 SF1-L2 补充载体（口径不称 TAO）。
- **外部评审三 P0 全部核实成立并当日修复**（commit 见当日）：
  1. P0-1 `VERIFIED` 只证明"出现"不证明"摆对" → 验收拆为 presence（MEM）∧ compliance（SPACE）双结论，EXEC 汇总 `VerificationVerdict`，"出现但放错 = FAILED(MISPLACED)"入契约与测试；
  2. P0-2 请求/结果同一消息自相矛盾 → 验收复核改为五条独立不可变消息（Request / PresenceResult / ComplianceResult / Verdict / UserAdjudication），全部携带 correlation_id、causation_id、producer、payload_hash，覆盖不全或串号 = 协议错误直接拒绝；`backend/tools/verification/verdict.py` + 9 个新测试，本地 22/22 绿；
  3. P0-3 单 venv 装不下两条官方依赖线（Step-Audio 钉 `transformers==4.49.0`，Nemotron VL 要 `>4.53,<4.54`，均经官方仓库/模型卡核实）→ 三套隔离环境 `~/venv`（视觉主链）/`~/envs/stepaudio`/`~/envs/nemotron_vl`，`spark_bootstrap.sh env <name>` + `configs/env_*.txt` 锁定；mamba-ssm aarch64 现场编译失败的兜底 = NGC 容器。
- 同批 P1 采纳：SP0 拆 **SP0-core（阻塞 S1）/ SP0-score（Day 2 收口，只阻塞演示与 A1）**；评分矩阵从"全部不可裁剪"改为**必须得分载体 / 有条件增强 / 明确可舍弃**三档；NVIDIA 主链锚定为 Nemotron VL 属性抽取（必经产出方，规则兜底 = 降级路径须 P1 上报），TTS 与 9B 文案降为增强档；口径红线：社区 checkpoint 转 TensorRT 只许说"TensorRT 部署"，"TAO"字样做了才准写。
- 教训：**"全部不可裁剪"等于没有优先级**——评分对齐矩阵不仅要枚举载体，还要给出出问题时的裁决顺序；以及验收语义要从"照片里有"追问到"放对了没有"，单 Agent 重跑自己的能力不构成验收。

- **S1/S2 主链代码落地（当晚第二批）**：`backend/pipeline/` 四模块——关键帧采样（均匀降采样+静止段去重）、确定性贪心 IoU 追踪器（同输入必同输出，有专门测试）、Grounding DINO / DINOv2 封装（延迟导入，本地不需要 torch）、`ingest_video` 编排（视频→Observation/Tracklet 契约对象+Top-K 证据裁剪+三阶段审计事件）。检测器/嵌入器按 Protocol 注入：本地用合成视频+fake 检测器测通端到端（含确定性双跑一致），Spark 上接真模型。本地 **32/32 测试绿**。
- SP0-core 探针脚本 `scripts/sp0_core_probe.py` 就绪并已部署：CUDA matmul 生死门 → 检测/嵌入真模型加载+前向+计时+峰值显存 → CP-SAT 双态冒烟，落 `results/acceptance/sp0/<run_id>/metrics.json`；节点上已挂 watcher（deps 装完自动跑探针）。无真实图片时探针只证明"加载+前向"，真实输出探针等 G0 素材重跑——不冒充质量证据。
- torch 安装仍在跟跨境网络缠斗（CUDA 库单包 200~700MB,超时自动续传中）；vision 双模型权重已完整落盘,Step-Audio 2 mini 下载中。

- **SP0-core 首轮探针(watcher 自动触发,run `sp0core_20260715_2027`)**:3/4 过——CUDA matmul(GB10 识别正常,4096² 0.366s)、DINOv2 嵌入前向(768 维,self-cosine=1.0)、CP-SAT 双态(PLAN_READY/NEW_SPACE_INCOMPATIBLE)全部真机通过;**检测失败**:Grounding DINO 前向触发 nvrtc JIT 内核编译,cu128 工具链不认 GB10 新算力架构(`invalid value for --gpu-architecture`)。根因回溯:cu130 首装死于 `error: incomplete-download`(跨境断流),自动降级到 cu128 埋下此雷。已用 `pip --resume-retries 10` 重装 cu130(断点续传)并链式自动重跑探针。教训:**降级路径成功 ≠ 主路径不再需要——cu128 能跑基础算子但 JIT 必炸,凡"自动 fallback 成功"都要回头问一句主路径当初为什么失败**。
- **阶跃 Step Plan 接入(Sean 提供控制台截图,配额 2000M 远超预期)**:落地 `scripts/stepfun_api.py` 本地客户端(stdlib-only,models/chat+audio 子命令,stderr 报 usage)+ `docs/STEPFUN_API_PLAYBOOK.md`(四条 dev-time 用途按冲刺价值排序:A1 prompt 预热 > G0 预标注/词表工厂 > LLM-judge > 文案润色;五条红线:密钥仅本地、演示主链不接云、云输出仅候选、影像出境先脱敏、批量任务记 usage)。`.env` 进 .gitignore 且 deploy.sh 排除——密钥双重隔离于 git 与节点。待 key 到手先跑 `models` 拿权威清单。

## 失败与教训

- 选题切换时，海事文档里已经做过的官方评分对齐（v0.2 评分对齐补丁、双生态载体、成品视频分镜、E1）没有随选题迁移，搬家 v0.2 六项标准里三项载体完全缺席——**评分对齐矩阵必须是任何新选题文档的第一节检查项，选题可以换，赛事标准不换**。
- 搬家选题在"行业落地价值"叙事上先天弱于 ToB 深场景，已在矩阵解读中明示：靠工程深度（第 2 项）和完整性（第 3 项）补分，访谈证据必须做实。
- 十日谈汇编稿（`docs/十日谈_念头通达.md`）的章节骨架仍是海事叙事线，需随新选题改写章节注释——留待 D4 处理，不阻塞今日交付。

## 明日计划

- Day 1（切片日历）已在今日开始：G0 数据拍摄与真值冻结、SP0 双生态探针、S0 骨架三线并行；负责人按手册 §2 分工。
- SP0 必须在明日内完成 Step-Audio 2 mini 启动探针，否则 A1 降级阶梯提前评估。
- 若 G0/SP0 明日收工仍未 PASS，按手册阻断规则优先补数据/换型号，不得越过依赖开始 S1/S2。

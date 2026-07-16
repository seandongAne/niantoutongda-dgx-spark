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
- **SP0-core 终局:PASS(当晚 23 点前)**。cu130 断点续传成功(torch 2.13+cu130),nvrtc JIT 错误随之消失;最后一雷是 transformers 新版把 `post_process_grounded_object_detection` 的 `box_threshold` 改名 `threshold`(文本标签迁 `text_labels`),一行修复后全绿:检测在合成图上真检出 "red box"(score 0.758,框坐标与画的矩形吻合)、嵌入 33ms/图、CP-SAT 双态、`failures: []`。**S1 依赖解除**。合成图口径不变:这是加载+前向+基础语义探针,真实场景质量探针等 G0 素材重跑。
- **阶跃 Step Plan 接入(Sean 提供控制台截图,配额 2000M 远超预期)**:落地 `scripts/stepfun_api.py` 本地客户端(stdlib-only,models/chat+audio 子命令,stderr 报 usage)+ `docs/STEPFUN_API_PLAYBOOK.md`(四条 dev-time 用途按冲刺价值排序:A1 prompt 预热 > G0 预标注/词表工厂 > LLM-judge > 文案润色;五条红线:密钥仅本地、演示主链不接云、云输出仅候选、影像出境先脱敏、批量任务记 usage)。`.env` 进 .gitignore 且 deploy.sh 排除——密钥双重隔离于 git 与节点。待 key 到手先跑 `models` 拿权威清单。

- **G0 试机片过真实管线（当晚，Sean 拍摄 35s 卧室片段）**：67 关键帧、检测 0.40s/帧、大件与停留物全部成轨（bed 25 帧/12.5s 跨度）；72% 单帧碎片经标签分析确认为杂物闪烁（box×29）而非追踪失败——**碎片率对乱房间是误导性指标，真验收口径应为"每件停留锚点是否成长轨"**。宽松参数（iou 0.2/miss 4）缝合断轨且碎片 -25%，已定为 ingest 默认（真实批次后随 S3 阈值冻结）。`scripts/g0_clip_check.py` 成为素材质检门。房间分配拍板：A=Sean 房间（仅开发，素材不进任何提交物），B=队友房间（冻结+成片，需记录展示授权），另加 R1 鲁棒性扫描（队友房间 2–3 个，零标注，只验流程）。

- **任务 A 首段真实视频质检（82s，17 件锚点清单就位）**：管线吞吐正常（164 关键帧、0.4s/帧），大件+杂物全成轨；但按清单对账 8/17 锚点缺席。逐帧目检定位两类根因：①词表语境错位——格架式玩具柜不是西方 "cabinet"，换 "toy storage organizer" 即救回；玫红水壶实为保温杯型（补 "tumbler"）；②拍摄距离——白色立柜怼脸拍导致物体大于画面（检测器无框可画），小件（水壶/夜灯/摄像头/台灯）疑似离太远。词表 v2 冻结进 `fixtures/dev_a/anchor_items.md`；距离法则写入拍摄指导（大件退后整件入画、小件凑近占画面 1/4+）。**流水线质检协议第一次真实运转就拦下了问题——如果三段拍完才检查,返工成本×3**。

- **任务 A 第一段重拍验收:16/17 锚点成轨(正式合格)**。三层根因逐一击破:①拍摄距离(白色立柜退后入画即成);②词表措辞(luggage/mini fridge/table lamp/laundry bag/tumbler 分别取胜);③**GDINO 得分稀释陷阱——15 类词表一次喂,小类得分被稀释到阈值下**(同一只水壶:15 类下碎片、3 类下 0.56 分)→ `detect.py` 改为 ≤6 类/批分批检测 + 跨批 IoU 去重,这是管线级修复,写进代码注释。唯一未决:夜灯(疑似发光球灯开灯过曝成白团,零命中)——待 Sean 确认实物,拍摄时关灯再验证。逐帧目检时顺带发现手工盒包装印有儿童照片(商品印刷),已提醒 Sean 后续镜头规避。教训:**"检测不到"要拆成距离/措辞/词表规模三个独立变量逐个排除,一次只动一个**。

- **任务 A 旧屋素材收官:4 段视频、17/17 锚点全部物理确认可检出**。二三段+定向补拍段(弱势锚点逐件近距停留)依次过质检;夜灯实物为白色柱体+贴纸(非猜测的发光球),检测词 smart speaker + cylinder lamp 双词兜底;摄像头双词 security camera/baby monitor。词表演进 v3→v5 沉淀出原则:**GDINO 词表是召回网不是分类器**——同批词竞争 token 归因,易混概念(行李箱/脏衣袋两度互吞)必须分批,每批 ≤4 词;"cylinder lamp" 单独成批宁滥勿缺,身份精度交给 S3。正式 S1/S2 已产出 348 tracklets(v3 词表),v5 全量重跑 nohup 中(4 段,配置 dev-a-vocab5)。S3 跨视频匹配为下一战役。

- **codex 提案裁决(教师-学生循环 + TAO 微调 GDINO)**:诊断照单全收(它抓到 `detect.py` 跨批 NMS 无类别去重会误删堆叠实体的真缺陷),处方砍一半——**TAO 微调设三条件门**(推理修复后仍不达标 ∧ 主链闭环 ∧ ≤Day 5;评分矩阵属"明确可舍弃",单房间微调打冻结验收 = 过拟合撞枪口),Mask/SAM、R1 伪标签同缓。采纳六项(canonical_id 词表编译 / alias-group NMS / prompt 自动搜索 / 停留段瓦片化 / 30–50 帧困难验证集 / 视觉教师+数据工厂分阶段损失)打包为 **S2.5 批次,一次性重跑 ingest**,专档文档 `docs/AI搬家复原Agent_S2.5_检测推理修复批次.md`。教训:**外部方案先过评分矩阵与泛化风险,再谈技术美感——"必须先修推理路径"这句 codex 自己说对了,结论却排了微调,取舍要自己做**。
- **v5 全量 ingest 收官**:v1 191 / v2 165 / v3 142 / v4 123 tracklets。G0 量化线达标(17 锚点每个 ≥2/3 旧段、≥12 词全三段);弱点入册当 S2.5 靶子:security camera 全程零轨迹、table lamp v1 缺、luggage v3 全灭、storage box 过火(71 条/段)、复合标签伪影 5 种。
- **Nemotron VL 本地环境:编译失败 → 当夜修复(未动用 NGC 兜底)**。三个根因逐一击破:①节点缺 python3.12-dev 且无 sudo → `apt-get download + dpkg -x` 用户态解包,经 CPATH 注入(父目录必须在内——Ubuntu pyconfig.h 是桩,相对引用 `<aarch64-linux-gnu/python3.12/pyconfig.h>`);②aarch64 上 torch cpp_extension 默认 Jetson 架构表(sm_53 起,CUDA 13 已删)→ 钉 `TORCH_CUDA_ARCH_LIST=12.1`;③mamba-ssm 2.2.5 setup.py **硬编码**同款架构表且不读环境变量,外加 CCCL 3.x 删除了 `cub::CTA_SYNC`/`cub::LaneId` → sdist 双补丁(`scripts/patch_mamba_gb10.py`,幂等)。causal_conv1d 1.6.2.post1 无架构问题只需①。验收:selective_scan fwd+**bwd**(bwd 正是补丁内核)+ 完整 Mamba block GB10 真机冒烟全过,transformers 4.53.3 合规(>4.53,<4.54)。教训:**"兜底预案存在"不等于"该走兜底"——三个报错(`Python.h missing`/`compute_53`/`cub has no member`)每个都指向可修的具体原因,先读报错再谈降级**。
- 会话勘误:本轮对话误开在 Brief-CC 仓库目录下(文件操作全走绝对路径,无实际影响);仓库根 `.env` 已补建(空 `STEPFUN_API_KEY=`,gitignore+deploy 双排除),Sean 拿到 key 直接填入即可。

## 失败与教训

- 选题切换时，海事文档里已经做过的官方评分对齐（v0.2 评分对齐补丁、双生态载体、成品视频分镜、E1）没有随选题迁移，搬家 v0.2 六项标准里三项载体完全缺席——**评分对齐矩阵必须是任何新选题文档的第一节检查项，选题可以换，赛事标准不换**。
- 搬家选题在"行业落地价值"叙事上先天弱于 ToB 深场景，已在矩阵解读中明示：靠工程深度（第 2 项）和完整性（第 3 项）补分，访谈证据必须做实。
- 十日谈汇编稿（`docs/十日谈_念头通达.md`）的章节骨架仍是海事叙事线，需随新选题改写章节注释——留待 D4 处理，不阻塞今日交付。

## 明日计划

- Day 1（切片日历）已在今日开始：G0 数据拍摄与真值冻结、SP0 双生态探针、S0 骨架三线并行；负责人按手册 §2 分工。
- SP0 必须在明日内完成 Step-Audio 2 mini 启动探针，否则 A1 降级阶梯提前评估。
- 若 G0/SP0 明日收工仍未 PASS，按手册阻断规则优先补数据/换型号，不得越过依赖开始 S1/S2。

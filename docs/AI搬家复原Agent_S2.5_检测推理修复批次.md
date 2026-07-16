# S2.5 检测推理修复批次 — codex 提案裁决落地

> 2026-07-15 定稿。来源:外部 codex 提案(教师-学生循环 + TAO 微调)的取舍裁决。
> 结论一句话:**诊断照单全收,处方砍一半——先修推理路径,微调设门。**
> 执行窗口:Day 4(2026-07-16),与 S3 并行不冲突(S3 用 v5 现有数据先跑)。

## 0. 批次红线

- **一次性重跑**:S2.5-1~4 全部落地后,任务 A 四段视频只重跑 **一次** ingest
  (`--config-version dev-a-vocab6`),产出 v6 基线。禁止逐项修完逐项重跑
  (每轮 4 段 ≈ 20min GPU + 一轮人工对账,拆开跑 = 白烧一天)。
- **口径红线不变**:没真跑 TAO,任何材料不得出现"TAO"字样(正确措辞 =
  "TensorRT 部署")。砍缓 TAO 微调的附带好处就是天然合规。
- 任务 B 素材/答案永不参与本批次任何环节(词表选择、打分、验证集)。

## 1. 采纳项(Day 4 全做)

| # | 项目 | 落点 | 验收 |
|---|---|---|---|
| S2.5-1 | **canonical_id 词表编译** — 三层解耦:用户语言(夜灯)/ 检测 prompt(smart speaker, cylinder lamp)/ 内部 ID(anchor_06)。schema: `canonical_id / display_label_zh / detection_prompts[] / confusable_group / notes` | 新建 `fixtures/dev_a/vocab.json`;`ingest_task.py` 增加 `--vocab` 入口,由编译器展开成批次化 prompt 序列(易混概念自动分批);`anchor_items.md` 保留为人读版 | ingest 不再手拼逗号词表;同一 canonical_id 的多 prompt 命中在产物里归并到同一标签 |
| S2.5-2 | **alias-group-aware NMS**(codex 抓到的真缺陷) — 现在 `detect.py` 跨批去重是无类别 IoU>0.8,会把两个紧贴/堆叠的**不同**实体删掉一个。改为:只在同一 `confusable_group`(= 同一 canonical 概念的别名词,如 security camera↔baby monitor、smart speaker↔cylinder lamp)内去重;跨概念高重叠框一律双留 | `backend/pipeline/detect.py` `detect()`;alias 分组信息从 vocab.json 传入 | 单测:两个不同概念框 IoU=0.9 → 保留 2;同概念别名框 IoU=0.9 → 保留高分 1 |
| S2.5-3 | **prompt 自动搜索** — 候选描述词(可含 Step 3.x 生成的同义词,key 到手后)在任务 A 真值上自动打分选优,替代手工五版试错。评分 = 锚点召回 − λ·每帧误报 − μ·碎轨率 | 新脚本 `scripts/prompt_search.py`,复用 g0_clip_check 的轨迹统计;真值 = anchor_items.md 17 锚点人工对账 | 对 v5 已知弱词(security camera / table lamp / luggage / storage box)各给出 ≥1 个得分更高的替代词或确认现词最优 |
| S2.5-4 | **停留段瓦片化多尺度检测** — 相机静止段(= 用户 2s 停留,即用户指定重点)对关键帧加做 2×2 重叠瓦片 + 杂乱区选择性 3×3,低阈值小物路径;瓦片坐标映射回全图后按 S2.5-2 规则去重。非停留段维持整帧单尺度控制成本 | `backend/pipeline/detect.py` 或新 `tiled.py`;静止段判定可复用 keyframes 的帧差信号 | 困难验证集(S2.5-5)上小物/堆叠召回相对 v5 提升;整段处理时间 ≤2× v5 |
| S2.5-5 | **人工困难验证集** — 从任务 A 四段抽 30–50 关键帧,专挑小物体/遮挡/堆叠场景,人工框真值。指标只留三个核心:**锚点召回 / 碎轨率 / 每帧误报**(codex 列的 8 个不全建) | `fixtures/dev_a/hardval/`(帧引用 + 真值 json);评测脚本并入 prompt_search 或独立 `scripts/hardval_eval.py` | v5 与 v6 各出一行三指标,v6 至少两项不劣、小物召回提升 |
| S2.5-6 | **视觉教师 + 数据工厂**(STEPFUN_API_KEY 已到手,07-15 链路验证通过) — Step 3.x flash 读脱敏 crop/contact-sheet 出结构化属性标签;Step Image Edit 做保身份增强(光照/模糊/遮挡/背景/尺度),train-only。供 SF1-L1 投影头。**损失分阶段上**:先纯真实基线 → +0.3 合成 → +0.2 属性,每阶段一行消融,不许一次全上(归因会糊) | `scripts/stepfun_api.py` 扩展;产物进 local-data(不进 git);manifest 版本化 + hash | 每阶段消融行齐全;合成图人工抽检通过率记录在案 |
| S2.5-7 | **GDINO 帧批处理**(2026-07-15 深夜拍板并入) — `detect.py` 现在逐帧推理(0.4s/帧是管线真瓶颈,不是 VLM);改为多关键帧攒批一次前向,GB10 128G 统一内存放得下大 batch。与 S2.5-4 瓦片化改**同一个文件**,同批落地避免两次动刀 | `backend/pipeline/detect.py`(与 S2.5-4 合并施工);批大小从显存实测定,fake 检测器单测保行为等价 | 单帧结果与逐帧路径 bit-等价(同阈值同框);整段 ingest 墙钟时间可见下降,计入 v6 对比行 |
| S2.5-8 | **hero crop 评分**(codex 提速提案 #3 的 ingest 侧) — tracklet 选代表帧从"检测分最高"升级为综合评分:面积 × 清晰度(拉普拉斯方差)× 完整度(截断/遮挡惩罚,框贴画面边缘 = 截断);Top-3 证据帧照旧留全(UI/审计/嵌入用),但**标出 hero** 供 S5 单调用 | `backend/pipeline/ingest_task.py` 选帧逻辑;评分函数独立可测 | 17 锚点的 hero 帧人工抽检:清晰、完整、无截断的比例显著高于 v5 的"最高检测分"帧 |

## 2. 设门项(大概率赛后)

| 项目 | 门条件 | 理由 |
|---|---|---|
| **TAO 微调 Grounding DINO** | 三条件**同时**满足才开工:① S2.5-5 困难验证集证明推理修复后仍不达标;② 主链(S3–S7)已闭环;③ 不晚于 Day 5 | ① 评分矩阵里属"明确可舍弃"档,"Spark 单机训练"载体已由 SF1-L1 投影头(必须档)承接;② 微调数据只能来自任务 A(Sean 房间),任务 B 是队友房间——单房间微调打冻结验收 = 过拟合撞枪口;③ aarch64 上 TAO 容器/ODVG 格式/配置 ≥1–2 天纯管道工时,Day 4–6 排的是主链 |
| Mask GDINO / SAM 实例分割 | 同上门;且仅当堆叠问题在瓦片化后仍存在 | codex 自己也排第二阶段 |
| **R1 按失败类型触发的主动学习储备** | ① 主链 S3–S7 闭环;② R1 额外授权素材存在且真实失败可复现;③ SF1-L1 真实基线已训练;④ 已预留不送云、不训练的 R1 challenge 子集 | 多余额度只作失败驱动储备,不设消耗 KPI;沿用 S2.5-6 contact-sheet 工具与人工复核,按下节路由,一轮无增益即停 |
| GDINO TensorRT 化 | **维持 SF1-L2 原排期,不提前**(2026-07-15 拍板) | 帧批处理(S2.5-7)先吃掉最大一口吞吐;TensorRT 引擎构建/校验是独立工程量,提前挤占 Day 4 主链;口径红线同上——没跑就不写 |

### 2.1 R1 主动学习闭环(过门后才执行)

R1 保持"低标注鲁棒性扫描"定位:不做全量框标注;只有真实失败触发时,才对最小失败
切片作人工确认。每轮固定为:

1. **发现与归档**:登记 `failure_type / source_hash / pipeline_version / reproduction`;
   每批只处理一种失败,不把不同根因混在同一轮。
2. **隐私门**:仅从额外授权的 R1 素材导出脱敏 crop/contact-sheet,逐批由 Sean
   拍板;任务 B 与预留 R1 challenge 子集都不送云、不训练。
3. **候选工厂**:Step 3.7 Flash 生成词汇/属性/难例候选,Step Image Edit 仅对已确认
   crop 做保身份定向增强;云输出不自动成为标签。
4. **人工校对与故障路由**:

   | `failure_type` | 训练/修复去向 | 禁止的错误归因 |
   |---|---|---|
   | `vocab_mismatch` / `class_miss` | 候选送 S2.5-3,在任务 A 真值上选词 | 不能因教师说得像就直接改冻结词表 |
   | `identity_mismatch` | 人工确认的 hard positive/negative 对送 SF1-L1 | 不把同类不同实例误作正样本 |
   | `blur` / `lighting` / `partial_occlusion` | 定向增强 train-only,按真实→合成→属性做消融 | 不用随机海量扩图稀释真实分布 |
   | `small_object_miss` / `stacked_suppression` | 先回 S2.5-2/4;仍失败才过检测器升级门 | 物体没有检测框时,投影头训练不能冒充修复 |

5. **双门验收**:目标失败指标改善且任务 A hardval 不回退;同时预留 R1 challenge
   同方向改善。通过才开下一小批,否则回退并停止该失败类型。
6. **证据**:manifest 记录来源 hash、模型/prompt/配置、候选/接受/拒绝数、token
   用量和前后指标;合成/伪标签仍只进 train split。

该闭环是赛程余量项,不阻塞 S2.5 完成定义,也不得抢占 S3–S7 主链。

## 3. 安全门(升格为标准作业)

- 任务 B 永不参与训练/选词/停止决策;
- 云模型(StepFun)只提议、不自批;进训练集的框必须人工确认;
- 合成/伪标签数据 train-only,不进验证集;
- R1 训练池与 R1 challenge 物理分离;challenge 不送云、不参与训练/选词;
- 每轮数据/词表/配置 manifest 版本化 + hash;指标无提升即回退;
- 演示主链永不接云 API(不变)。

## 4. v5 基线(2026-07-15 夜,对比基准)

`dev-a-vocab5`,threshold 0.28,批次=4。四段:v1 191 / v2 165 / v3 142 / v4 123 tracklets。

**G0 量化线达标**:17 锚点每个 ≥2/3 旧段可见;≥12 词全三段可见。

**已知弱点(= S2.5 的靶子)**:

| 弱点 | 证据 | 归属修复项 |
|---|---|---|
| `security camera` 全程零轨迹(摄像头全靠 baby monitor) | 矩阵无该行 | S2.5-3 |
| `table lamp` v1 缺席(0/1/1/0,散落 `lamp` 碎片) | 矩阵 | S2.5-3 |
| `luggage` v3 全灭(3/2/0/2,两只行李箱都丢) | 矩阵 | S2.5-3/4 |
| `storage box` 过火(71/57/38/25,成了杂物召回网) | 矩阵 | S2.5-3(词太宽)+ S2.5-5(误报计入) |
| 复合标签伪影(`luggage mini fridge` 等 5 种) | 矩阵 | S2.5-1(归并)+ S2.5-2 |
| 堆叠/紧贴实体可能被 NMS 误删 | 代码审查(codex) | S2.5-2 |
| 小物整帧单尺度饿死 | 结构性 | S2.5-4 |

完整矩阵在节点 `~/proj/logs/ingest_a_v5.log`;产物 `~/proj/local-data/ingest_a_v5/`。

## 5. 执行顺序与完成定义

1. S2.5-1 词表编译 → S2.5-2 NMS 修复(带单测)→ S2.5-5 困难验证集标注(人工,可并行)
2. S2.5-3 prompt 搜索(本地词库 + 教师生成候选,key 已通)
3. S2.5-4 瓦片化 + S2.5-7 帧批处理(同文件同批施工)+ S2.5-8 hero 评分
4. **一次性重跑** `dev-a-vocab6` → 三指标对比 v5 → S3 在 v6 数据上复验
5. S2.5-6 数据工厂异步启动,不阻塞 1–4
6. R1 主动学习仅在 §2 四门同时通过后按失败类型启动;它不属于本批次阻塞项

**完成定义**:v6 三指标不劣于 v5 且小物/堆叠召回可见提升;S3 硬负样本对
(4 组同类不同实例)在 v6 上的判定不回退;journal 记录消融行。

> Nemotron VL 本地环境注:mamba-ssm/causal_conv1d 编译失败已于当夜**修复**
> (三根因:python3.12-dev 缺失无 sudo → 用户态解包 deb + CPATH;torch 默认
> Jetson 架构表 → 钉 TORCH_CUDA_ARCH_LIST=12.1;mamba-ssm 硬编码 sm_53..87 +
> CCCL 3 删除 CTA_SYNC/LaneId → sdist 补丁 `scripts/patch_mamba_gb10.py`)。
> selective_scan fwd+bwd + Mamba block 已在 GB10 真机冒烟通过,SP0-score
> 探针解锁;NGC 容器兜底**未动用**。

## 6. Day 4 执行记录（2026-07-16）

### 6.1 已落地

- S2.5-1/2：`fixtures/dev_a/vocab.json` + canonical/category 分层编译；同
  canonical 别名去重、跨概念重叠保留，单测覆盖。
- S2.5-3/5 的**评测器与口径**已冻结：IoU=0.5 最大基数一对一匹配，输出
  锚点召回/碎轨率/FP 每帧；任务 B dataset id 明确拒绝。真实 30–50 帧人工框
  GT 尚未交付，因此 prompt 排名与 v5/v6 三指标仍不得计算。
- S2.5-4/7/8：停留/低运动帧瓦片检测、GDINO 帧 batch、hero crop 综合评分
  已进入 ingest 主路。hero 作为 Tracklet `attributes.hero_ref/hero_score` 输出，
  Top-3 证据仍保留。
- 诊断对比工具 `scripts/compare_ingest_runs.py` 只输出产物数量、canonical 覆盖、
  hero 覆盖和墙钟；它把 hardval 指标显式标为未评测，禁止用 tracklet 数冒充碎轨率。

### 6.2 GB10 真机定参

| image batch | 相对逐帧 | 结论 |
|---:|---:|---|
| 2 | 1.158× | 冻结主路 |
| 4 | 1.112× | 无额外收益 |
| 8 | 0.969× | 回退，禁用 |

batch 与逐帧的标签/框结构一致；GPU 浮点差最大为 `score=0.000671`、
`box=0.0041px`，在冻结的 `1e-3 / 0.5px` 决策容差内，但不宣传为逐 bit
相等。原 tile 探针将单帧 18 个整帧框放大到 147 个；加入“切片边缘截断剔除、
仅补全图面积 ≤12% 的小物、每 canonical 每帧最多 3 个”后降为 38 个，整帧
18 框不丢。

### 6.3 真实素材的 stationary 降级

首个 v6 尝试完成 v1 后日志暴露 `tiled_kf=0`，立即停止并以
`ingest_a_v6_aborted_stationary0` 留档，未继续烧完四段。根因不是程序没调用，
而是四段手持素材不存在可靠的连续 2 秒绝对静止：2fps 全局光流探针在四段只找到
0/0/1/0 个严格低运动段。

修复后规则为：严格停留存在则优先；否则明确记录
`adaptive_low_motion_fallback`，每段从最低运动 10% 取帧，时间 NMS 间隔 2s，
最多 12 帧。四段无模型复验实际选中 9/11/6/12 帧，共 38 帧，按 tile 真机墙钟
实测整批为 v5 的 1.328×，守住 ≤2× 成本门。冻结参数见
`configs/ingest_dev_a_v6.yaml`。

### 6.4 v6 与 S3 v6 最终诊断

`dev-a-vocab6` 已完成四视频一次性重跑并拉回本地，验收证据提交于
`542382e`：

| 诊断量（非 hardval 指标） | v5 | v6 | 变化 |
|---|---:|---:|---:|
| keyframes | 614 | 614 | 0 |
| observations | 3919 | 6563 | +2644（+67.5%） |
| tracklets | 621 | 945 | +324（+52.2%） |
| hero 覆盖 | 0/621 | 945/945 | 100% |
| 四段墙钟 | 1249s | 1659s | 1.328× |

v6 使用 batch=2，四段 tile 帧为 9/11/6/12，守住 S2.5-4 的 ≤2× 成本门。
弱词的 canonical tracklet 出现量增加，其中 `security_camera` 由 4/5/5/8 增至
9/11/10/11，`table_lamp` 由 0/1/1/0 增至 7/7/8/9，`luggage` 由
3/2/0/2 增至 11/11/6/10。**这些是无真值候选诊断，不是召回率。**
`storage_box` 仍高达 83/64/38/34，unknown tracklet 也由 20 增至 33，表明候选
膨胀与复合 prompt 噪声尚未解决。

S3 v6 三次复跑 hash 均为
`700c121094068dd96ab8de4d2256013ba8ee6887fcef4b4d9504987af71cc613`：

| S3 诊断量 | v5 | v6 |
|---|---:|---:|
| tracklets / entities | 621 / 564 | 945 / 871 |
| 自动链接 | 62 | 83 |
| 澄清请求 | 544 | 921 |
| matched / suspected / new entities | 45 / 374 / 145 | 59 / 617 / 195 |

确定性与数据流已闭环，但新增候选把澄清量放大 69.3%。因此下一步不是继续降低
检测阈值，而是先做低质量候选门控、同视频短轨缝合，并用 hero 属性增强 S3 证据。
完整机器真值仍缺失，故 `g2_evaluated=false`，联系表也不能冒充四组正式 PASS。

另一个关键诊断是：v6 的 7811 条 S3 candidate 中 `attribute` 分量恒为 1.0、
`context` 恒为 0.5，而 `configs/reid_dev_a_v1.yaml` 中两者权重均为 0。即 hero
引用虽然达到 945/945，但属性尚未真正进入 S3 判别；必须先修正 missing/unknown
的默认相似度语义并产生有方差的颜色/材质/文字属性，再在有真值条件下调权重。

### 6.5 尚未完成，不得提前宣称

- S2.5-5 真实人工框 hardval 与其三指标；
- S2.5-3 在该真值上的候选词胜负；
- S2.5-6 家庭 crop 的云视觉教师/增强批次（逐批影像出境批准尚未取得，故未上传）；
- 17-anchor hero 的清晰/完整/无截断人工质量抽检；
- data owner 对 S3 四组硬负 tracklet 映射的正式确认，以及 17-anchor 完整 G2 真值。

因此原批次“v6 三指标不劣、小物召回提升、四组硬负不回退”的完整完成定义仍未满足；
当前完成的是工程实现、真机成本门、可复跑诊断链与阻塞项固化。

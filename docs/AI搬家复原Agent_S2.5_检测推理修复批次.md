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
| S2.5-6 | **视觉教师 + 数据工厂**(等 STEPFUN_API_KEY) — Step 3.x flash 读脱敏 crop/contact-sheet 出结构化属性标签;Step Image Edit 做保身份增强(光照/模糊/遮挡/背景/尺度),train-only。供 SF1-L1 投影头。**损失分阶段上**:先纯真实基线 → +0.3 合成 → +0.2 属性,每阶段一行消融,不许一次全上(归因会糊) | `scripts/stepfun_api.py` 扩展;产物进 local-data(不进 git);manifest 版本化 + hash | 每阶段消融行齐全;合成图人工抽检通过率记录在案 |

## 2. 设门项(大概率赛后)

| 项目 | 门条件 | 理由 |
|---|---|---|
| **TAO 微调 Grounding DINO** | 三条件**同时**满足才开工:① S2.5-5 困难验证集证明推理修复后仍不达标;② 主链(S3–S7)已闭环;③ 不晚于 Day 5 | ① 评分矩阵里属"明确可舍弃"档,"Spark 单机训练"载体已由 SF1-L1 投影头(必须档)承接;② 微调数据只能来自任务 A(Sean 房间),任务 B 是队友房间——单房间微调打冻结验收 = 过拟合撞枪口;③ aarch64 上 TAO 容器/ODVG 格式/配置 ≥1–2 天纯管道工时,Day 4–6 排的是主链 |
| Mask GDINO / SAM 实例分割 | 同上门;且仅当堆叠问题在瓦片化后仍存在 | codex 自己也排第二阶段 |
| R1 伪标签难例挖掘 | R1 素材存在 + 基线投影头训完 | 设计本身好(传递一致性 + 人工复核);contact-sheet 对比工具先做(与 S2.5-6 同一工具) |

## 3. 安全门(升格为标准作业)

- 任务 B 永不参与训练/选词/停止决策;
- 云模型(StepFun)只提议、不自批;进训练集的框必须人工确认;
- 合成/伪标签数据 train-only,不进验证集;
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
2. S2.5-3 prompt 搜索(先本地词库,key 到手后加教师生成候选)
3. S2.5-4 瓦片化
4. **一次性重跑** `dev-a-vocab6` → 三指标对比 v5 → S3 在 v6 数据上复验
5. S2.5-6 数据工厂随 key 到手异步启动,不阻塞 1–4

**完成定义**:v6 三指标不劣于 v5 且小物/堆叠召回可见提升;S3 硬负样本对
(4 组同类不同实例)在 v6 上的判定不回退;journal 记录消融行。

> Nemotron VL 本地环境注:mamba-ssm 与 causal_conv1d 在节点双双编译失败
> (2026-07-15 夜),按预案走 **NGC PyTorch 容器**兜底;若容器路径也超时,
> 属性抽取的执行者可切 StepFun 云 VLM(仅开发期工具用途,演示主链不接云)。

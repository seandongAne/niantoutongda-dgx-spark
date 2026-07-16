# Stepfun Step Plan 2000M token 使用手册

> **2026-07-16 初赛范围覆盖：**停止网络/生成图片规模化校准与 R1 主动学习；
> 不再建设 household-pack 或 challenge 数据池。本文中对应章节只作历史设计保留，
> 当前执行以 `docs/初赛范围冻结_2026-07-16.md` 为准。
>
> 赛方提供的阶跃 API 配额(约 2000M Credit)。接入信息来自开发平台控制台;
> 客户端 = `scripts/stepfun_api.py`(仅本地运行)。

## 接入事实(2026-07-15 控制台截图)

- Base URL:`https://api.stepfun.com/step_plan/v1`
- 双协议:Chat Completions(OpenAI 兼容 `/chat/completions`)+ Messages(Claude 兼容 `/messages`)
- 已验证用途覆盖:step-3.7-flash、stepaudio-2.5-*、step-image-edit-2;**权威清单以
  `python scripts/stepfun_api.py models` 实调结果为准,不凭截图猜**。
- 密钥:仓库根 `.env` 的 `STEPFUN_API_KEY`(已被 .gitignore 与 deploy.sh 双重排除)。

## 红线(先于一切用途)

1. **密钥只存本地 Mac**——永不进 git、永不上 Spark 节点(节点零凭据纪律)。
2. **演示主链不调用云 API**——断网高光与"本地优先"是我们自己的评分叙事。
   云用途全部是 dev-time。
3. **云端输出只是候选/评审意见**:预标注须人工校对后才能进真值;judge 结论
   只进分析报告,不改数据;不得让云输出自动写入任何契约对象。
4. **涉家庭影像/语音出境前必须脱敏**(G0 纪律),且逐批由 Sean 拍板。
5. 每个批量任务记录 token 用量(客户端 stderr 已带 usage 行)。

## 用途台账（含已停止项）

| # | 用途 | 模型 | 时机 | 说明 |
|---|---|---|---|---|
| 1 | **A1 prompt 预热** | stepaudio-2.5-* | 立即~Day 5 | 本地 Step-Audio 2 mini 环境就绪前,先在云上把"拍摄旁白→物品标签候选"的 prompt 与输出 JSON 协议调通、冻结;权重就绪后平移到本地。A1 时间盒只有 0.5 天,预热直接决定成败 |
| 2 | **G0 预标注 + 词表工厂** | step-3.7-flash | G0~S2.5-3 | 纯文本零隐私成本:教师生成 Grounding DINO 英文检测词与同义词候选,直接送入 S2.5-3,在任务 A 人工真值上自动打分选优,替代手工多版试错 |
| 3 | **停止：S2.5-6 规模化视觉教师 + 数据工厂** | step-3.7-flash + step-image-edit-2 | 初赛不执行 | 不采集或生成大规模图片，不建立训练数据工厂；已有设计只作赛后备忘 |
| 4 | **停止：R1 主动学习** | step-3.7-flash + step-image-edit-2 | 初赛不执行 | 不建立 R1 数据池、challenge、云端增强或循环训练 |
| 5 | **LLM-judge** | step-3.7-flash | S3 起 | 跨视频匹配错误分析批处理、任务卡可读性评审;judge prompt 入库,结论只进 results/ 分析报告,不改数据 |
| 6 | **文案与十日谈润色** | step-3.7-flash | D9 | 十日谈汇编、演示旁白稿;低优先 |

> 用途②③的 household-pack 产品化扩展已由 2026-07-16 后续范围裁决停止；
> `docs/家居视觉校准工厂_设计.md` 仅保留为赛后备忘。已经生成的纯文本候选可在
> 不开启新数据工程的前提下复用，但不得反向扩成初赛关键路径。

## R1 主动学习储备（已停止，历史设计）

以下内容不再执行，仅用于解释曾经的安全护栏。除非 Sean 重新明确授权，任何 session
不得据此创建 R1 素材、challenge、训练批次或云任务。

R1 不是"把剩余额度跑掉"的自动训练任务,而是失败驱动的储备通道。必须同时满足:

1. 失败来自额外授权的 R1 鲁棒性素材,可稳定复现并登记 `failure_type`;
   **任务 B 始终冻结**,不得进入 R1 训练、选词或停止决策。
2. 只上传 Sean 逐批批准的脱敏 crop/contact-sheet,不上传原始家庭视频;云端只生成
   候选,进入训练集前逐条人工确认。
3. 每批只处理一种失败类型,在 manifest 记录来源 hash、模型与 prompt 版本、候选/
   接受/拒绝数量及 token 用量;预先留出不送云、不训练的真实 R1 challenge 子集。

| 真实失败类型 | 阶跃产物 | 正确去向 |
|---|---|---|
| 词汇错位、类别漏检 | 英文同义词/外观描述候选 | S2.5-3 prompt 搜索;由任务 A 真值打分,不得直接改冻结词表 |
| 已检出但跨视频认错实例 | 属性候选、hard positive/negative crop 对 | SF1-L1 投影头 train-only 数据 |
| 模糊、光照、部分遮挡导致嵌入不稳 | 保身份定向增强 | SF1-L1 分阶段训练;真实图基线→+合成→+属性逐行消融 |
| 小物完全漏检、堆叠框被吞 | 失败归因与定向增强候选 | 先回 S2.5-2/4;仍失败才触发检测器升级门。**投影头不能修复根本没有框的物体** |

一轮只允许改变一个数据或配置变量。只有目标失败指标改善、任务 A hardval 不回退、
R1 challenge 同方向改善时才开下一批;任一不满足就回退并停止该失败类型的额度投入。
多余额度只作为这一通道的储备,不设消耗率 KPI。

Messages API(Claude 兼容)意味着也能给本地 agent CLI 供模型——评估过,
对写代码的边际价值低(主力开发另有分工),暂不启用;若出现大量机械性
杂活(批量翻译、格式迁移)可重开。

## 加分角度

dev-time 使用阶跃 API 本身就是 Stepfun 生态投入的额外证据——每次批量任务
在当天十日谈记一笔(用途 + token 量),与本地 Step-Audio 推理形成
"云上调优、端上部署"的完整叙事。

## 第一步(key 到手后)

```bash
python scripts/stepfun_api.py models          # 拿权威模型清单
python scripts/stepfun_api.py chat --model step-3.5-flash --prompt "ping"
```

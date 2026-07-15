# Stepfun Step Plan 2000M token 使用手册

> 赛方提供的阶跃 API 配额(约 2000M Credit)。接入信息来自开发平台控制台;
> 客户端 = `scripts/stepfun_api.py`(仅本地运行)。

## 接入事实(2026-07-15 控制台截图)

- Base URL:`https://api.stepfun.com/step_plan/v1`
- 双协议:Chat Completions(OpenAI 兼容 `/chat/completions`)+ Messages(Claude 兼容 `/messages`)
- 可用模型:step-3.5 文本系 + step-audio-2.5 语音系等;**权威清单以
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

## 四条用途(按冲刺价值排序)

| # | 用途 | 模型 | 时机 | 说明 |
|---|---|---|---|---|
| 1 | **A1 prompt 预热** | step-audio-2.5 | 立即~Day 5 | 本地 stepaudio 环境就绪前,先在 API 上把"拍摄旁白→物品标签候选"的 prompt 与输出 JSON 协议调通、冻结;权重就绪后平移到本地 Step-Audio 2 mini。A1 时间盒只有 0.5 天,预热直接决定成败 |
| 2 | **G0 预标注 + 词表工厂** | step-3.5-flash(+VL 若在清单) | G0 拍摄日 | 纯文本零隐私成本:生成 Grounding DINO 开放词汇英文检测词表(家居物品);若有 VL 模型:对脱敏关键帧出标签候选,人工只做校对;与 Nemotron VL 输出对拍 = 免费第二意见质检 |
| 3 | **LLM-judge** | step-3.5-flash | S3 起 | 匹配错误分析批处理、任务卡可读性评审;judge prompt 入库,结论进 results/ 分析报告 |
| 4 | **文案与征文润色** | step-3.5-flash | D9 | 十日谈汇编、演示旁白稿;低优先 |

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

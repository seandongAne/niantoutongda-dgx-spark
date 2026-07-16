# Nemotron 在 DGX Spark 上的服务路线与实测(2026-07-16 凌晨)

> 触发:Sean 质疑 mamba-ssm/causal_conv1d 是否本不必要(NVIDIA 应有原生适配),
> 并提议 NVFP4 量化。当前结论:**NVFP4/vLLM 是已上线主路,
> transformers/BF16 是已验证 fallback;vLLM 容器路线不需要那两个包**。

## 我们用的模型

- 主路:`nv-community/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`
  (`vlm_attributes`,vLLM 容器,已在 Spark 实测通过)。
- 已验证 fallback:`nv-community/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`
  (transformers + `nemotron_vl` venv)。
- 已裁撤独立 9B `task_copy`:结构化任务卡仍是确定性主路;如需纯文本润色,
  复用同一 12B 服务且不得修改事实字段。
- 12B v2 VL 是 **NemotronH 混合 Mamba-Transformer** 架构 —— 这正是
  BF16/transformers fallback 需要 mamba-ssm/causal_conv1d 的原因
  (模型卡官方要求),不是我们配置错误。

## 两条路线

| | transformers/BF16 fallback | vLLM/NVFP4 主路 |
|---|---|---|
| mamba-ssm/causal_conv1d | 需要(已在 GB10 原生编译通过,`patch_mamba_gb10.py`) | **不需要**(vLLM 内置 mamba2 内核) |
| 量化 | BF16(~24GB 权重) | **NVFP4-QAD**(~4 分之一体积,Blackwell 原生 FP4) |
| 接入方式 | nemotron_vl venv 内 python | OpenAI 兼容 HTTP :8000(主 venv 直接 curl) |
| 运行载体 | `~/envs/nemotron_vl`(已就绪) | 容器 `vllm-nvfp4`,仅绑定 `127.0.0.1:8000` |
| 状态 | **已验证 fallback**:fwd+bwd+Mamba block 与图文质量通过 | **已上线主路**:真实图文 25.4 tok/s,单 tile 已生效 |

## NVFP4 结论

- 官方 QAD(量化感知蒸馏)checkpoint 存在且 **ModelScope 已核验 200**:
  `nv-community/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`
  (架构字段 NemotronH_Nano_VL_V2;HF 侧同物异名 `Nemotron-Nano-VL-12B-V2-FP4-QAD`)。
- 社区 playbook(NVIDIA 论坛置顶)给出的起始配方:
  `vllm serve <model> --quantization modelopt_fp4 --max-model-len 24000
  --gpu-memory-utilization 0.3 --port 8000`;项目已在
  `nvcr.io/nvidia/vllm:26.06-py3` 无补丁跑通,并按真实 S5 工况收紧为
  max-model-len 4096、单 tile 等参数(完整实测见下)。
- "全部选 NVFP4"的现实边界:只有 Nemotron 槽位有官方 NVFP4;GDINO/DINOv2
  体量小走 TensorRT FP16 即可(SF1-L2 口径),Step-Audio 属阶跃生态无此选项。

## 节点当前服务状态

- docker 28.3.3 已装;`Developer` 已加入 docker 组并经全新 SSH 登录验证,
  client/server 均为 28.3.3。
- `vllm-nvfp4` 已常驻,服务只绑定 `127.0.0.1:8000`;权重位于
  `~/models/nv-community__NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`。
- 后续 S5 客户端直接走 OpenAI 兼容接口;服务端主配方不再待跑。

## BF16 实测(2026-07-15 深夜,`scripts/bench_nemotron_bf16.py`,判据:decode ≥20 tok/s 则不迁移)

| 工况 | prefill(温机) | decode(温机) |
|---|---|---|
| 文本-only(33 tok prompt) | 0.13 s | **7.8 tok/s** |
| 图文(真实 S5 证据 crop,1323 tok prompt) | 0.72 s | **1.7 tok/s** |

加载 211 s;峰值显存 27.7 GB;首轮含 triton JIT 预热(~5 s,一次性)。
输出质量合格:对 ingest 证据 crop 正确输出了逐物品"类别/颜色/材质/文字标记"
(样例:"Toy Washing Machine: Category: Toy, Color: White with pink...")。

**判据击穿 → 迁移 vLLM+NVFP4 成立。** 真实 S5 工况(图文)1.7 tok/s 意味着
128-token 属性抽取一次 ~75 s;百次级别的批量任务在 transformers 路线上不可用。
图文比纯文本慢 4.6×(1323 tok 上下文的注意力层 KV 开销 + HF generate 低效),
vLLM 的连续批处理恰好是这类多 crop 批量工况的对症药。

### 运行时坑位(已固化进 bootstrap / bench 脚本)

- ModelScope 镜像目录的嵌套 `auto_map` 指向 HF 原站(LLM 骨干 + RADIO 视觉塔
  的 .py 代码文件)→ 节点不可达;解法 `HF_ENDPOINT=https://hf-mirror.com`
  (只拉 KB 级代码文件,一次性缓存;权重仍走 ModelScope,不违反纪律)。
- NemotronH 推理时 triton JIT 现场编译 `cuda_utils.c`,同样吃 Python.h →
  CPATH 需在**运行时**在场;已用 `.pth` import 行注入 venv(sitecustomize.py
  方案不可用:Debian 在 /usr/lib/python3.12 有同名文件且路径序在前,会遮蔽)。
- processor 输出的 `num_patches` 键必须过滤后再喂 `generate`(README 官方
  示例只传 input_ids/attention_mask/pixel_values 三键,是有原因的)。

## vLLM + NVFP4 实测(2026-07-16 凌晨,`scripts/vllm_smoke.py`,同工况对照)

容器 `nvcr.io/nvidia/vllm:26.06-py3`,权重 NVFP4-QAD 9.9GB。服务端参数
(codex 基线配方,已验证可跑):`--trust-remote-code --quantization
modelopt_fp4 --max-model-len 4096 --limit-mm-per-prompt '{"image":1,"video":0}'
--max-num-seqs 8 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.30`。

| 工况 | BF16/transformers(decode) | NVFP4/vLLM 默认 tile(含 prefill) | NVFP4/vLLM 单 tile(含 prefill) |
|---|---|---|---|
| 文本-only | 7.8 tok/s | **26.5 tok/s** | 26.2 tok/s(同工况,正常波动) |
| 图文(真实 S5 证据 crop) | 1.7 tok/s | **25.4 tok/s**(prompt **1322** tok) | **26.0 tok/s**(prompt **298** tok) |

注意口径:vLLM 列是 completion/总耗时(**含 prefill**),比 BF16 的纯 decode
口径更保守——真实差距只会更大。图文工况 **~15×**,128-token 属性抽取从
~75 s 降到 ~5 s。输出质量复核合格(同一张证据 crop,正确产出
"Toy Washing Machine / Category / Color / Material" 结构)。

**单 tile A/B 的正确解读**:视觉前缀 1280→256 tok(5 tile→1,codex 的
tile 账精确命中:1322=5×256+42 文本,298=256+42)。单请求 128-token 输出
工况下 decode 占主导,tok/s 只从 25.4→26.0(墙钟 5.05→4.92s)——**收益不在
单请求 tok/s,在批量工况**:每次调用省 ~1024 tok prefill 计算 + KV cache
足迹缩 4.4×(并发席位更多),S5 上百 crop 批量时才兑现。质量本样张无回退
(玩具洗衣机属性照抽),但小字/标记类 crop 在 512² 单 tile 下可能丢细节——
条件升级策略(见下方裁决 #3)正是为此留的门。

### vLLM 侧坑位

- 第一枪失败:NVFP4-QAD 目录含 custom code → 必须 `--trust-remote-code`。
- codex 给的 `--mm-processor-kwargs '{"max_num_tiles":1,"use_thumbnail":false}'`
  **一半是错的**:vLLM 26.06 内建 `NanoNemotronVLProcessor` 只接受
  `max_num_tiles` 作 init kwarg,`use_thumbnail` 从模型 config.json 读、
  传了直接 TypeError 拒启。且源码 `if use_thumbnail and patches.shape[0] > 1`
  —— tile=1 时 thumbnail 自动不加,只传 `{"max_num_tiles":1}` 即达到
  单 tile 目标(256 视觉 token)。

## codex 提速提案裁决(2026-07-15 深夜,五项采纳四项半)

| # | 提案 | 裁决 | 落点 |
|---|---|---|---|
| 1 | 单 tile 视觉输入(`--mm-processor-kwargs`) | **采纳,已 A/B 验证**(上表)。服务端参数修正:只传 `{"max_num_tiles":1}` — `use_thumbnail` 不是 init kwarg,传了 TypeError 拒启;源码 tile=1 时 thumbnail 自动不加。bbox 外扩 8–12% + letterbox 512²(防长条 crop 直接 resize 变形)归 **S5 客户端 crop 制备** | 服务端已生效;客户端部分进 S5 设计 |
| 2 | 输出 128→32 token JSON schema(vLLM structured outputs + `"unknown"` 合法 + temperature 0) | **采纳,归 S5**。128-token 开放式描述是迁移决策的探针工况,不是生产路径;生产 prompt = 已知类别候选 → 只补属性槽位。冒烟脚本保持 128 是为与 BF16 同工况可比,别改 | S5 实现 |
| 3 | hero crop 单调用 + 条件升级(unknown/跨视频冲突/需读字 → 才跑第二张或多 tile) | **采纳**。Top-3 证据帧是 UI/审计/嵌入用的,不是三倍 VLM 预算;12× 调用削减账成立。hero 综合评分(面积×清晰度×截断惩罚)是 ingest 侧小改 → **进 S2.5-8**(反正要重跑);升级策略归 S5 | S2.5-8 + S5 设计 |
| 4 | 服务端基线参数 + 有界异步并发扫(1/2/4/8/16) | **采纳,服务端已生效**(max-model-len 4096 / max-num-seqs 8 / batched-tokens 8192 / gpu-util 0.30 / limit-mm image:1——原 24000 max-model-len 确实是照抄社区 playbook 的浪费)。并发扫等 S5 客户端存在后做 | 服务端已生效;并发扫归 S5 |
| 5 | 调用削减(只喂 17 锚点 Top-K)+ 持久缓存(`crop_sha256+prompt_schema_version+model_revision`)+ 验收指标重构(整批墙钟/P50-P95/JSON 合法率/峰值内存,替代单请求 tok/s) | **采纳,归 S5 设计**。单请求 tok/s 的历史使命(迁移拍板)已完成 | S5 设计 |
| — | 它建议的 A/B 顺序(基线→单tile→schema→hero→并发) | **半保留**:作测量顺序对,作工程顺序不对——#2/#3/#5 是 S5 实现内容,S5 排在 S3 跨视频匹配之后,提速优化不许插队到 S3 前面。今晚只做到基线+单tile 对照为止 | 本文档 |
| — | "暂别折腾 speculative decoding / prefix cache / 激进显存" | **同意**,与我方判断一致 | — |

## 已拍板并生效的决策

1. vLLM + NVFP4-QAD 立为 S5 属性抽取的**服务路线**——**已实测成立**
   (上表,图文 ~15×);已修好的 transformers+BF16 env 降为**已验证兜底**。
   服务常驻配置 = codex 基线参数 + `{"max_num_tiles":1}`,升级工况
   (需读字/冲突复核)由 S5 客户端按 #3 策略**发局部放大的第二张 crop**
   (mm-processor-kwargs 是服务级参数,升级不靠换服务配置)。
2. VL 模型纯文本 prompt 也能答 → **已裁撤 9B-v2 task_copy 槽位**。
   任务卡主路保持结构化渲染;可选润色复用同一 12B 服务,少一份权重和
   一次探针,且生成失败不影响任务执行。
3. 评审口径红利:"NVFP4-QAD 跑在 GB10 Blackwell 原生 FP4 tensor core 上,
   走 NVIDIA 官方 DGX Spark playbook 路线" —— 平台适配 15% 的直球叙事。

## 参考

- NVIDIA 官方 playbooks: github.com/NVIDIA/dgx-spark-playbooks (nvidia/vllm)
- 社区 Nemotron v2 VL 配方: github.com/raphaelamorim/spark-playbooks
  (run-nemotron-v2-VL),NVIDIA 论坛帖 350349
- vLLM 官方博客 2026-06-01 "vLLM on the DGX Spark"(cu130-nightly,
  统一内存调参:低 --max-num-seqs、留显存余量、先热身再压测)
- build.nvidia.com/spark/nemotron

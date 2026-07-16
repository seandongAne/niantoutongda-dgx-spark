# Nemotron 在 DGX Spark 上的官方服务路线调研(2026-07-15 夜)

> 触发:Sean 质疑 mamba-ssm/causal_conv1d 是否本不必要(NVIDIA 应有原生适配),
> 并提议 NVFP4 量化。结论:**两条路线并存,vLLM 容器路线不需要那两个包**。

## 我们用的模型

- `nv-community/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`(vlm_attributes 主链载体)
- `nv-community/NVIDIA-Nemotron-Nano-9B-v2`(task_copy,增强档)
- 12B v2 VL 是 **NemotronH 混合 Mamba-Transformer** 架构 —— 这正是
  transformers 路线需要 mamba-ssm/causal_conv1d 的原因(模型卡官方要求),
  不是我们配置错误。

## 两条路线

| | transformers 路线(已修好) | vLLM 容器路线(官方 playbook) |
|---|---|---|
| mamba-ssm/causal_conv1d | 需要(已在 GB10 原生编译通过,`patch_mamba_gb10.py`) | **不需要**(vLLM 内置 mamba2 内核) |
| 量化 | BF16(~24GB 权重) | **NVFP4-QAD**(~4 分之一体积,Blackwell 原生 FP4) |
| 接入方式 | nemotron_vl venv 内 python | OpenAI 兼容 HTTP :8000(主 venv 直接 curl) |
| 前置条件 | 无(已就绪) | docker 权限(见下)+ 容器拉取 ~20GB |
| 状态 | fwd+bwd+Mamba block 冒烟通过 | 待跑通(Day 4 时间盒) |

## NVFP4 结论

- 官方 QAD(量化感知蒸馏)checkpoint 存在且 **ModelScope 已核验 200**:
  `nv-community/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`
  (架构字段 NemotronH_Nano_VL_V2;HF 侧同物异名 `Nemotron-Nano-VL-12B-V2-FP4-QAD`)。
- 社区 playbook(NVIDIA 论坛置顶)实跑配方:
  `vllm serve <model> --quantization modelopt_fp4 --max-model-len 24000
  --gpu-memory-utilization 0.3 --port 8000`;基底镜像 nvidia/vllm:25.10 时代
  需打 Nemotron v2 VL 支持补丁,26.x / `vllm/vllm-openai:cu130-nightly`
  预期已含上游支持 —— **首次 serve 必须实测验证**。
- "全部选 NVFP4"的现实边界:只有 Nemotron 槽位有官方 NVFP4;GDINO/DINOv2
  体量小走 TensorRT FP16 即可(SF1-L2 口径),Step-Audio 属阶跃生态无此选项。

## 节点现状与堵点

- docker 28.3.3 已装,但 docker 组只有 xsuper;**Developer 在 sudo 组**
  (`sudo -n` 失败只是因为要密码)→ 解锁 = Sean 在节点上执行
  `sudo usermod -aG docker Developer` 后重新登录。无 podman,无 rootless 前置。
- 磁盘 2.5T 空闲,128G 统一内存,容器拉取无资源压力(跨境慢,夜里 nohup 拉)。

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

## 决策建议(待 Sean 拍板)

1. vLLM + NVFP4-QAD 立为 S5 属性抽取的**服务路线**;已修好的
   transformers+BF16 env 降为**已验证兜底**(零沉没成本,今晚已冒烟)。
2. 顺手简化:VL 模型纯文本 prompt 也能答 → **裁撤 9B-v2 task_copy 槽位**,
   一个模型双职,少一份权重/一次探针(models.yaml v0.3 一并改)。
3. 评审口径红利:"NVFP4-QAD 跑在 GB10 Blackwell 原生 FP4 tensor core 上,
   走 NVIDIA 官方 DGX Spark playbook 路线" —— 平台适配 15% 的直球叙事。

## 参考

- NVIDIA 官方 playbooks: github.com/NVIDIA/dgx-spark-playbooks (nvidia/vllm)
- 社区 Nemotron v2 VL 配方: github.com/raphaelamorim/spark-playbooks
  (run-nemotron-v2-VL),NVIDIA 论坛帖 350349
- vLLM 官方博客 2026-06-01 "vLLM on the DGX Spark"(cu130-nightly,
  统一内存调参:低 --max-num-seqs、留显存余量、先热身再压测)
- build.nvidia.com/spark/nemotron

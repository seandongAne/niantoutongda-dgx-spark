# 语义重排与推理优化对比（2026-07-18，复核版）

结论：TopK 的确会把很小的 proposal 漂移放大成巨大的 query-slot raw tensor
差异，但本轮进一步证明，原始 `4.17 / 0.984` 剪刀差还混入了未封存的 cuDNN TF32
策略。对当前两张冻结图，ONNX Runtime 的检测集合与 PyTorch 严格等价；真正仍会丢失
检测的是现有 TensorRT engine。当前可直接保留的性能收益只有 `torch.compile` FP32
的 `1.170×`，BF16/FP16 虽更快但检测门失败。Torch-TensorRT 已推进到 99.98% dry-run
覆盖，实际 engine build 在 26.06 nightly 栈的 BERT embedding `aten.add` converter
处触发符号 shape 错误，尚无可报告的混合执行延迟。

上述全部结果只覆盖 batch 2、两张冻结图和一个 prompt，是诊断证据，不是独立冻结集
验收，也不能替代 hero R@1 或全量检测回归。

## 本轮性能与正确性

| 路线 | P50 / 名义加速 | 检测集合门 | 当前结论 |
|---|---:|---|---|
| eager FP32 | `680.78 ms / 1.000×` | `[5,5]`，PASS | 冻结基线 |
| eager FP16 | `497.98 ms / 1.366×` | 检测数 `[5,5]`，但标签/顺序结构 FAIL | 不启用 |
| eager BF16 | `502.19 ms / 1.356×` | `[4,5]`，FAIL | 不启用 |
| `torch.compile` FP32 | `581.83 ms / 1.170×` | `[5,5]`，score `≤7.79e-5`、box `≤0.00037 px`，PASS | 当前唯一可保留候选；首次 lazy invocation `59.1 s`，仍有 10 个 graph break |
| `torch.compile` BF16 | `312.84 ms / 2.176×` | `[4,5]`，FAIL | 性能上界，不启用 |
| ONNX Runtime CPU FP32 | `7.80 s` 量级 | `[5,5]→[5,5]`；10/10 exact-label 一一 IoU 匹配，strict diagnostic PASS | 图在当前检测决策口径下等价；不是性能候选 |
| 现有 TensorRT FP32 no-TF32 | `647.9 ms / 1.05×` | `[5,5]→[2,3]`，FAIL | engine 仍不可用 |
| 现有 TensorRT FP16 | `277.2 ms / 2.45×` | `[5,5]→[1,3]`，FAIL | 无效输出下的速度上界 |
| Torch-TensorRT hybrid | 未形成 engine | 容器 eager no-TF32 集合门 PASS；export bit-exact | dry-run `4520/4521`、规划两段 TRT partition、99.98%；实际 FP32 build 在 `aten.add` converter 失败，未测 FP16 |

GPU steady-state forward 使用 CUDA Event；ORT CPU 与 `torch.compile` 首次 lazy
invocation 使用 wall-clock。两种口径均不含图像预处理、H2D、后处理与并发服务。
首次 invocation 包含本次运行的 lazy compilation，不把它描述为跨环境通用的
“绝对冷编译时间”。

## TopK 与精度策略的复核

- 当前 host PyTorch 默认状态为 matmul TF32 `false`、cuDNN TF32 `true`、float32
  matmul precision `highest`；显式复现该状态时，与旧冻结 oracle 的 logits/boxes
  逐元素 bit-exact。旧 oracle 没有记录这些开关，因此后续冻结合同必须把它们写入
  manifest，不能再靠框架默认值。
- 同一 host 从默认状态改为双 TF32 `false` 后，20,906 个 proposal 的 TopK 集合在
  两张图上仍均为 900/900 相同，但同 rank 只剩 837/900、840/900。raw 最终 boxes
  最大偏差 `0.98414`；按 encoder proposal ID 对齐后仅约 `2.0e-4`，检测集合严格 PASS。
- NGC 26.06 与 host 都关闭 TF32 后，TopK 两批分别只有 4/0 个 rank 槽变化，集合均
  900/900 相同，检测集合严格 PASS。两边都使用所谓“PyTorch default”时，不同 cuDNN
  build 让 TopK 仅约 223/224 个槽同 rank，并替换 4/3 个 proposal；这说明默认精度
  策略会把运行栈差异带进 oracle。
- 因此 raw query-slot 等价应保留为诊断和安全报告，但不能单独承担检测语义验收。
  当前诊断门使用 exact label partition + maximum-total-IoU 一一匹配，并同时限制
  IoU、score 和像素框差；真正切主链仍需更大的独立冻结检测集。

## ONNX 边界结果

- 原 ONNX 与插桩 ONNX 在相同 ORT provider/optimization 下，最终 logits/boxes
  bit-exact，排除了“加输出本身扰动图”的解释。
- 图中只有一个 TopK，`axis=1`、`K=900`、`largest=1`、`sorted=1`；另有 48 个
  GridSample，覆盖 encoder/decoder 各 6 层、每层 4 次。GridSample 目前只是 sentinel，
  没有证据把它写成根因。
- 相对 host no-TF32，proposal score 最大偏差 `7.44e-5`、coord logits 最大偏差
  `2.11e-4`；TopK 第一张集合 900/900，第二张 899/900。raw final boxes 最大偏差
  `0.6098`，按 proposal ID 对齐后降至 `7.69e-6`；但对齐后的 final logits 最大
  偏差仍为 `1.3666`。TopK 主导槽位与 boxes 的剪刀差，但没有消除 decoder logits
  漂移，不能据此把所有差异归给 TopK。
- 正式 AutoProcessor 后处理得到 `[5,5]→[5,5]`。no-TF32 对比的最差 IoU
  `0.9999983`、score 差 `2.74e-4`、框差 `0.00025 px`；相对旧默认精度 oracle 的
  最差 IoU `0.9999797`、score 差 `3.63e-4`、框差 `0.0030 px`。两路均过当前 strict
  diagnostic 门，所以不能再把 ORT raw 剪刀差写成检测数退化。

## 分区收益上限

冻结 FP32 profile 的 P50 构成为：文本 backbone `3.83 ms / 0.56%`、视觉 backbone
`214.39 ms / 31.36%`、encoder `400.95 ms / 58.65%`、decoder `49.08 ms / 7.18%`，
其余约 `14.94 ms / 2.18%`。hook 后整机 P50 只增加约 0.18%，并保持输出 bit-exact。

如果只把视觉 backbone 加速 3 倍，Amdahl 上限约为
`1 / (1 - 0.3136 + 0.3136 / 3) = 1.264×`，还没扣除分区和数据交换开销。因此
backbone-only TensorRT 可作为低风险备选，但不是主要性能押注；58.65% 的 encoder
才决定上限，也正是 deformable attention / GridSample 所在区域。

## Torch-TensorRT 实际阻断点

- v4 默认分解：容器 eager no-TF32 对 host no-TF32 的 raw 槽位最大偏差仍为
  logits `0.928`、boxes `0.626`，但 10 个检测全部严格集合匹配；导出兼容改写与
  `torch.export` 对 same-process eager 均 bit-exact。
- dry-run 报告 4,521 个算子中 4,520 个 converter-supported，只把一个
  `aten.all.dim` 留在 PyTorch，规划 3,280 + 1,240 算子的两段 TRT partition；
  该阶段尚未形成实际 engine。
- 真正 build 在 BERT embeddings 的 `embeddings + position_embeddings`，即
  `torch.ops.aten.add.Tensor`，进入 elementwise converter 后触发
  `ValueError: __len__() should return >= 0`。experimental decompositions 没有改变
  该节点；指定 module FQN 的回退请求也未生效，dry-run 与失败节点均未变化，原因
  尚未定位。
- 这属于 Torch-TensorRT build/converter 问题，不是本轮检测精度问题。下一次应优先
  用正式稳定的匹配版本复现并提交最小 bug reproducer，或显式把 `aten.add.Tensor`
  留在 PyTorch后重新查看覆盖/engine 数；没有 engine 前不写性能数字。官方设置参考
  [Torch-TensorRT CompilationSettings](https://docs.pytorch.org/TensorRT/user_guide/compilation/compilation_settings.html)。

## 语义重排与评估边界

- 实例级语义签名重排仍只保留 transductive diagnostic：dev R@1
  `0.4961→0.6378`，已经打开的 58-query proxy 为 `46→51` 个 Top-1 命中。
- 既有 learned projection 的训练标签覆盖 proxy 502/502 tracklet，旧 selection 也
  暴露过全体转写覆盖率；该 partition 已消耗，不能继续调转写器后复用它宣称提升。
- 优化转写层仍值得用于清理复合标签和稳定 schema；若要显著推高正式 R@1，仍需要
  实例级语义/视觉重排，以及未参与本轮任何选择的新 `holdout_b`。

## 下一步顺序

1. 先将显式 precision policy、输入/模型/processor/scorer 哈希和检测集合匹配器一起
   封入新冻结合同，并扩展到未参与调参的检测集。
2. 主链候选先保留 `torch.compile` FP32，做端到端吞吐、并发和更大冻结集回归；继续
   分析 10 个 `Tensor.item()` graph break 能否安全消除。
3. Torch-TensorRT 先对 `aten.add` converter 形成最小复现；只有 FP32 engine 的集合门
   PASS 后才打开 FP16/BF16 和性能报告。
4. GridSample/MSDeformAttn 继续做逐层 sentinel 对比；只有出现 proposal-ID 对齐后仍
   扩大的第一处边界，才进入插件或自定义 converter。
5. TF-TRT 继续旁路；没有 native TensorFlow FP32 与 PyTorch FP32 对齐前，不创建
   性能数字或新依赖。

## 证据索引

- `../torch-compile-20260718-v2/compile.json`
- `../gdino-module-profile-20260718-v2/profile.json`
- `../topk-decision-comparisons-20260718/`
- `../onnx-intermediate-20260718-v1/result.json`
- `../onnx-pytorch-boundary-comparison-20260718/`
- `../torch-tensorrt-hybrid-20260718-v4-decision/`
- `../torch-tensorrt-hybrid-20260718-v5-experimental/`
- `../torch-tensorrt-hybrid-20260718-v6-text-embedding-fallback/`
- `../semantic-proxy-20260718-v1/selection.json`

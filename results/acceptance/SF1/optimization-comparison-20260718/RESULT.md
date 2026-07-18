# 语义重排与推理优化对比（2026-07-18）

结论：本轮找到了语义签名重排的正向诊断信号，也测得 AMP 约 1.35 倍的模型核心
加速；但没有一项满足主链切换门。Torch-TensorRT 在容器 eager FP32 与现有冻结
PyTorch FP32 的第一道对齐门即停止，TF-TRT 因缺少 native TensorFlow FP32 oracle
未进入性能测试。

## 对比结果

| 路线 | 实测结果 | 正确性/独立性门 | 当前结论 |
|---|---|---|---|
| 实例级语义签名重排 | dev R@1 `0.4961→0.6378`；已打开的 58-query proxy 为 `46→51` 个 Top-1 命中 | 未读 hero GT，但既有 projection 的训练标签覆盖 proxy 502/502 tracklet；旧 selection 还暴露过全体转写覆盖率 | 仅证明有信号，不是独立冻结集结果，更不是 `0.8083→0.8793` |
| PyTorch AMP FP16 | P50 `680.43→497.98 ms`，`1.366×`；检测数仍为 `[5,5]` | 标签/顺序结构不等价，严格决策门 FAIL | 不直接启用 |
| PyTorch AMP BF16 | P50 `680.43→502.50 ms`，`1.354×` | 检测数 `[5,5]→[4,5]`，严格决策门 FAIL | 不直接启用 |
| Torch-TensorRT 混合执行 | 官方 26.06 派生容器 eager FP32 P50 `657.45 ms` | 对冻结 FP32：有限 logits 最大偏差 `4.1671`，boxes `0.9841`；第一门 FAIL | 未进入 TRT 分区/编译，不报告加速 |
| 自定义算子转换 | FP16/BF16 的文本、视觉、encoder/decoder 多阶段均观察到漂移 | 4096 点阶段抽样只能定位“首次观测”，不能证明致因算子 | NO-GO；先做算子级 trace 与 FP32 island 消融 |
| TF-TRT | 未测性能 | 项目无 TensorFlow、无 native TF Grounding DINO SavedModel，现有 ONNX oracle 也未对齐 PyTorch | NO-GO；不拉 TF 容器制造不可比数字 |

## 语义重排证据边界

- candidate-only tutor 伪标签共 116 个 identity、502 个 tracklet；构造时未读取冻结
  人工 GT。旧实验按 identity hash 分为 74/42 个 identity，tracklet 交集为 0。
- 权重 `0.15` 在 127 个 dev query 上选出；已打开 partition 的点估计为
  R@1 `0.7931→0.8793`，即净增 5/58。两侧 Wilson 95% 区间分别为
  `[0.6723, 0.8775]` 与 `[0.7712, 0.9403]`，区间明显重叠；未保存逐 query
  paired flip，不能宣称统计稳定。
- 旧 projection 的训练伪标签覆盖本轮 502/502 tracklet，因此这里只能称
  “冻结既有投影上的 transductive diagnostic”。该 partition 已被打开；转写器后续
  fail-closed 修订不得再回用它宣称提升。
- 新评估器已经改成默认拒绝 learned projection、select 只构建 dev signature、
  holdout 阶段才构建 holdout signature，并冻结 evaluator/词表/属性/ingest/projection
  全部哈希。真正的 R@1 验收仍需要新 `holdout_b` 或新房间数据。

## 精度与转换定位

- AMP 计时为 batch 2、两张冻结图、单 prompt 的 forward-only CUDA Event；不包含
  预处理、H2D、后处理与服务并发。FP32 repeat 与首次 FP32 输出逐元素一致，P50
  只差约 0.39%，说明本批计时稳定。
- FP16 首个 `>1e-3` 的抽样偏差出现在 BERT layer 1，首个 `>0.1` 的抽样偏差在
  Swin stage 1；BF16 分别在 BERT layer 0 与 text projection。后续 encoder/decoder
  仍继续累积，当前没有单一 converter 靶点。
- Grounding DINO logits 中存在合法、位置一致的 `-inf` 文本 mask；门禁允许相同
  `-inf` pattern，但拒绝 NaN、`+inf`、pattern 漂移以及有限值超差。

## Torch-TensorRT 与 TF-TRT 门禁

- Torch-TensorRT 使用固定 NVIDIA 26.06 镜像 digest，派生层只安装
  `transformers==5.13.1`。运行栈为 PyTorch `2.13.0a0`、Torch-TensorRT
  `2.13.0a0`、TensorRT `11.0.0.114`、CUDA `13.3`。
- 当前冻结 oracle 来自 PyTorch `2.13.0+cu130`。同权重、同输入下，容器 eager
  FP32 已产生与此前 ONNX Runtime 同量级的偏差，因此没有把容器结果重定义成新基线，
  也没有用全 PyTorch fallback 冒充混合加速。
- TF-TRT 只应在 native TensorFlow FP32 SavedModel 与 PyTorch FP32 的 raw
  logits/boxes 先对齐后进入；当前缺少这条实现和 oracle，性能测试保持关闭。

## 下一步进入条件

1. 收集并冻结未参与本轮任何选参的新 `holdout_b`；若保留 projection，只允许使用
   dev identity 训练后再封存 holdout。
2. 对 AMP 做 FX/ATen 算子级 trace，并以 selective FP32 islands 逐段消融；在更大
   冻结检测集上用标签+IoU 匹配报告新增、丢失与重排。
3. Torch-TensorRT 先取得与现有 PyTorch build ABI/数值一致的运行栈；eager FP32
   raw oracle PASS 后，才重新打开 dry-run 分区覆盖、hybrid FP32 和 FP16 性能门。
4. TF-TRT 继续保持旁路；没有 native TF FP32 对齐产物时不创建性能数字。

## 证据索引

- `../semantic-proxy-20260718-v1/selection.json`
- `../semantic-proxy-20260718-v1/holdout.json`
- `../torch-precision-20260718-v1/precision.json`
- `../precision-drift-20260718-v1/stage-drift-v2.json`
- `../torch-tensorrt-hybrid-20260718-v2/probe.json`
- `../trt-gdino-20260718-v10-dynamo-static/onnxruntime_check.json`

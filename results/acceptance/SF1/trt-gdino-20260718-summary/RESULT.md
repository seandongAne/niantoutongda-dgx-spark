# Grounding DINO TensorRT 性能测量

结论：**当前不能切换主链**。TensorRT 对 Grounding DINO 核心计算显示出明显的速度上界，但 FP16、FP32/TF32、纯 FP32 no-TF32 三种 engine 均未通过冻结输出与最终检测决策等价门；独立 ONNX Runtime 也复现了导出图漂移，因此尚不能把名义加速计为可用优化收益。

## 冻结负载

- Spark：NVIDIA GB10，CUDA 13.0，TensorRT 10.14.1.48。
- 模型：`IDEA-Research__grounding-dino-base`。
- 输入：两张真实关键帧，batch 2，`pixel_values=[2,3,1333,750]`，文本长度 12。
- prompt：`luggage. mini fridge. water bottle. desk.`
- 计时范围：仅模型核心，不含图像预处理和检测后处理。
- PyTorch 稳定基线：5 次 warmup、30 次正式测量；TensorRT：10 次 warmup、100 次正式测量，非默认 CUDA stream。

## 性能与正确性

| 运行时 | P50 | P95 | 吞吐 | 名义加速 | 检测数 PyTorch→TRT | 等价门 |
|---|---:|---:|---:|---:|---:|---|
| PyTorch FP32 | 680.1 ms | 685.3 ms | 2.94 img/s | 1.00× | `[5,5]` | 基线 |
| TensorRT FP16 | 277.2 ms | 278.9 ms | 7.22 img/s | 2.45× | `[5,5]→[1,3]` | FAIL |
| TensorRT FP32/TF32 | 565.7 ms | 568.5 ms | 3.54 img/s | 1.20× | `[5,5]→[2,3]` | FAIL |
| TensorRT FP32 no-TF32 | 647.9 ms | 650.7 ms | 3.09 img/s | 1.05× | `[5,5]→[2,3]` | FAIL |

FP16 的合法 `-inf` 文本 mask 位置与 PyTorch 完全一致，双方 NaN 数均为 0；失败来自有限 logits 与 boxes 的实质漂移，不是 NaN 或 mask 误判。FP16 最大有限 logits 偏差 6.78，最大归一化 box 偏差 0.998。

## 导出边界

- legacy 与 dynamo 两条导出路径均完成了真实输入前置验证；mask、正弦位置编码 scale 等导出期改写在导出前对模型输出逐元素 bit-exact，且没有修改 site-packages。
- dynamo ONNX 通过 checker，实际 opset 为 18。ONNX Runtime 1.23.2 CPU 与 PyTorch 对比仍失败：最大有限 logits 偏差 4.17，最大 box 偏差 0.984。
- 因独立 ORT 已复现漂移，当前问题至少包含 PyTorch→ONNX 图语义不等价；TensorRT FP32 no-TF32 的进一步漂移也不能归因于 FP16 或 TF32。
- 动态 ONNX 首次构建被 TensorRT 的动态 `EyeLike` 限制拒绝；本次性能结果仅代表固定 batch、分辨率和文本长度。

## 采用建议

主链继续使用 PyTorch Grounding DINO。2.45× 仅作为“值得继续研究”的性能上界，不作为已实现收益。下一步应改测更小的隔离子图，或采用 PyTorch 集成式 TensorRT 路径；任何候选仍须先通过同一组冻结 raw-output、非有限值模式和最终检测决策门，再比较速度与内存。

结构化结果见 [summary.json](summary.json)。

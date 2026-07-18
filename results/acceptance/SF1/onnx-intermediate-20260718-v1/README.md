# 本地归档边界

该目录保留插桩计划、运行结果、TopK/最终输出和边界张量。完整
`encoder_class_logits_before_topk_reduce` 原数组为 42,815,616 bytes，以 gzip 形式
归档为
`tensors/03-encoder_class_logits_before_topk_reduce-feff05126f56.npy.gz`：

- 远端路径：`spark:~/proj/results/acceptance/SF1/onnx-intermediate-20260718-v1/tensors/03-encoder_class_logits_before_topk_reduce-feff05126f56.npy`
- SHA-256：`a1234fde7d052a8dc917162a1c0356e187aa446301603f89eb763d08cc284681`
- shape/dtype：`[2,20906,256] / float32`

解压后的 SHA-256 已复核为上述原始文件哈希。该数组也已在正式比较器中完成文件哈希、
shape/dtype/元素数验证和边界计算；派生报告位于
`../onnx-pytorch-boundary-comparison-20260718/`。重新计算时先将 `.npy.gz` 解压到
`result.json` 声明的 `.npy` 路径即可。

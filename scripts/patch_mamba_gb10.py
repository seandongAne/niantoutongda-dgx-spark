#!/usr/bin/env python
"""mamba-ssm 2.2.5 sdist 补丁 — GB10/CUDA 13 编译修复(2026-07-15,两处,均幂等)。

① setup.py 硬编码 Jetson 架构表(sm_53..87 + 版本门 90/100):CUDA 13 已删
   sm_53~72,nvcc 直接 fatal,且该表不读 TORCH_CUDA_ARCH_LIST → 整块替换为
   目标机真实算力单架构(GB10 = sm_121)。
② csrc/selective_scan/reverse_scan.cuh 用了 CCCL 3.x(随 CUDA 13)已删除的
   CUB 内部符号 — 实测本机 CCCL 只缺 cub::CTA_SYNC / cub::LaneId 两个
   (Uninitialized/RowMajorTid/Shuffle* 等仍在) → call-site 等价替换:
   CTA_SYNC()->__syncthreads();LaneId()->threadIdx.x%32(本文件内核均为
   1-D block,线性 tid 取模即硬件 lane)。

用法: patch_mamba_gb10.py <sdist 根目录> <算力,如 12.1>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def patch_setup_archs(root: Path, cap: str) -> None:
    path = root / "setup.py"
    src = path.read_text()
    if f"compute_{cap}" in src:
        print(f"setup.py: already patched (sm_{cap})")
        return
    block = re.compile(
        r'[ ]{8}cc_flag\.append\("-gencode"\)\n'
        r'[ ]{8}cc_flag\.append\("arch=compute_53,code=sm_53"\)'
        r'[\s\S]*?arch=compute_100,code=sm_100"\)\n'
    )
    repl = (
        '        cc_flag.append("-gencode")\n'
        f'        cc_flag.append("arch=compute_{cap},code=sm_{cap}")\n'
    )
    patched, n = block.subn(repl, src, count=1)
    if n != 1:
        sys.exit(f"setup.py: arch block not found — sdist layout changed?")
    path.write_text(patched)
    print(f"setup.py: Jetson arch table -> sm_{cap} only")


def patch_reverse_scan_cccl3(root: Path) -> None:
    path = root / "csrc" / "selective_scan" / "reverse_scan.cuh"
    src = path.read_text()
    if "cub::CTA_SYNC" not in src and "cub::LaneId" not in src:
        print("reverse_scan.cuh: already patched (no removed cub symbols)")
        return
    n_sync = src.count("cub::CTA_SYNC()")
    n_lane = src.count("cub::LaneId()")
    src = src.replace("cub::CTA_SYNC()", "__syncthreads()")
    src = src.replace(
        "cub::LaneId()", "(threadIdx.x % 32) /* cub::LaneId(), removed in CCCL3 */"
    )
    path.write_text(src)
    print(f"reverse_scan.cuh: CTA_SYNC x{n_sync}, LaneId x{n_lane} replaced")


def main() -> int:
    root, cap = Path(sys.argv[1]), sys.argv[2].replace(".", "")
    patch_setup_archs(root, cap)
    patch_reverse_scan_cccl3(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

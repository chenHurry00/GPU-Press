#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-}"

if [[ -z "${NVCC}" ]]; then
    NVCC="$(command -v nvcc || true)"
fi

# 当前机器 CUDA Toolkit 位于该路径，但仍允许用户通过 NVCC 环境变量覆盖。
if [[ -z "${NVCC}" && -x "/home/yuchen/SATA-KING/cuda-13.0/bin/nvcc" ]]; then
    NVCC="/home/yuchen/SATA-KING/cuda-13.0/bin/nvcc"
fi

if [[ -z "${NVCC}" || ! -x "${NVCC}" ]]; then
    echo "错误：未找到 nvcc。请将 CUDA bin 加入 PATH，或指定 NVCC=/path/to/nvcc。" >&2
    exit 1
fi

# native 让 nvcc 针对当前机器的 GPU 架构生成代码，避免把 RTX 3070 架构写死。
"${NVCC}" -O3 -std=c++14 -arch=native \
    "${SCRIPT_DIR}/gpu_burn.cu" \
    -o "${SCRIPT_DIR}/gpu_burn"

chmod +x "${SCRIPT_DIR}/gpu_burn"
echo "编译完成：${SCRIPT_DIR}/gpu_burn"

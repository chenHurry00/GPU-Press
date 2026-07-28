#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-}"

if [[ -z "${CONDA_BIN}" ]]; then
    CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "${CONDA_BIN}" && -x "/home/yuchen/SATA-YMK/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/home/yuchen/SATA-YMK/miniconda3/bin/conda"
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
    echo "错误：未找到 conda。可使用 CONDA_BIN=/path/to/conda 指定。" >&2
    exit 1
fi

ENV_NAME="gpu-press"
if ! "${CONDA_BIN}" env list | grep -Eq "^[[:space:]]*${ENV_NAME}[[:space:]]"; then
    echo "正在创建 conda 环境 ${ENV_NAME}..."
    "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10 flask
fi

# gpu_burn 使用 -arch=native 针对当前显卡编译。每次启动都重新编译，
# 避免更换不同计算架构的显卡后继续使用旧内核镜像。
"${SCRIPT_DIR}/build.sh"

exec "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python "${SCRIPT_DIR}/gpu_press.py"

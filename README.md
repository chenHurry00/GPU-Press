# GPU-Press

本机 NVIDIA GPU 压力测试与实时监控工具。网页端可启动/停止 CUDA 计算负载，并显示类似 FurMark“甜甜圈”界面的温度、功耗、GPU 利用率和显存曲线。

## 功能

- CUDA FP32 FMA 持续计算负载，强度分为 1～4 档
- 自动检测多张 NVIDIA GPU，可在 Web 页面中多选压测设备
- 每张 GPU 独立显示温度、功耗、利用率、显存和实时曲线
- 支持定时运行或持续运行到手动停止
- Web 页面启动、停止测试
- 通过 `nvidia-smi` 每秒采样一次：
  - GPU 温度
  - 板卡功耗
  - GPU 利用率
  - 显存占用
  - SM 时钟
- SSE 实时推送，Chart.js 绘制最近 10 分钟曲线
- 75°C 温度提示、85°C 高温警告
- Web 服务退出时自动终止压力测试子进程

## 当前本机环境

已按以下环境适配：

- NVIDIA 驱动：595.71.05
- CUDA Driver API：13.2
- CUDA Toolkit / `nvcc`：13.0

脚本仍会优先从 `PATH` 自动查找 `conda` 和 `nvcc`，上述路径仅作为当前机器的回退路径。

## 快速启动（推荐）

```bash
cd GPU-Press
chmod +x "build.sh" "start.sh"
./start.sh
```

`start.sh` 会自动：

1. 创建名为 `gpu-press` 的 Conda 环境（若不存在）；
2. 安装 Python 3.10 和 Flask；
3. 编译 CUDA 压测程序（若尚未编译）；
4. 启动 Web 服务。

启动后浏览器访问：

```text
http://127.0.0.1:8765
```

页面会默认勾选所有 GPU。可在启动前取消任意设备；同一轮中所有选中的 GPU
使用相同的运行时长和负载强度。运行期间设备选择会锁定，“停止”会终止本轮的全部
压测进程。

后端使用 NVIDIA UUID 通过 `CUDA_VISIBLE_DEVICES` 绑定每个子进程，确保页面中选择的
`nvidia-smi` 设备与实际参与压测的 CUDA 设备一致。

停止 Web 服务：在启动终端按 `Ctrl+C`。Web 服务退出时会一并停止正在运行的 GPU 压测。

> 首次运行需要 Conda 下载 Python/Flask 包，因此需要网络。后续启动不会重复创建环境。

## 分步启动

### 1. 创建 Conda 环境

```bash
conda create -y -n gpu-press python=3.10 flask
```

也可使用已有 Conda 环境：

```bash
conda activate <你的环境名>
pip install -r "requirements.txt"
```

### 2. 编译 CUDA 压测程序

```bash
cd GPU-Press
chmod +x "build.sh"
./build.sh
```

`build.sh` 使用 `-arch=native`，自动针对当前 GPU 的 Compute Capability 编译。更换不同架构的显卡后必须重新编译；`start.sh` 会在每次启动时自动执行这一步，避免复用旧显卡的 CUDA 内核镜像。

### 3. 启动后端

```bash
python /home/yuchen/scripts/GPU-Press/gpu_press.py
```

默认只监听本机回环地址 `127.0.0.1:8765`。修改端口：

```bash
GPU_PRESS_PORT=9000 ./start.sh
```

## 使用建议

1. 首次测试先选择 **强度 1**，时长 60 秒，确认散热与风扇正常。
2. 再使用 **强度 2～3** 测试 10～30 分钟。
3. 强度 4 用于极限测试，建议有人值守并持续观察温度。
4. 出现以下情况应立即停止：
   - 画面花屏或系统无响应；
   - 驱动重置、CUDA 错误；
   - 温度持续超过 85°C；
   - 异常噪音、焦味或功耗表现异常。

显卡的温控和热节流阈值因型号、厂商 BIOS 而异。Web 页面中的 75°C / 85°C 仅是保守提示，不替代当前显卡的厂商规格。

## 验证压测是否生效

测试运行时可在另一个终端执行：

```bash
watch -n 1 nvidia-smi
```

正常情况下应看到：

- `GPU-Util` 接近 100%；
- 功耗明显上升；
- 温度逐渐上升后趋于稳定；
- 时钟在温度/功耗限制范围内保持稳定。

压力测试的目标不仅是达到高温，还应观察：

- 是否出现 CUDA 错误；
- GPU 时钟是否异常大幅波动；
- 功耗与利用率是否稳定；
- 系统日志中是否出现 Xid 错误：

```bash
journalctl -k --since "10 minutes ago" | grep -iE "NVRM|Xid"
```

## 文件结构

```text
GPU-Press/
├── build.sh              # 编译 CUDA 内核
├── start.sh              # 自动创建 Conda 环境并启动
├── gpu_burn.cu           # CUDA 压力测试程序
├── gpu_press.py          # Flask 后端、进程控制、GPU 采样、SSE
├── requirements.txt      # Python 依赖
├── tests/
│   └── test_gpu_press.py # 多卡采样、选择校验与设备绑定测试
└── templates/
    └── index.html        # Web 控制台和实时曲线
```

## 安全说明

- 本工具会让 GPU 长时间接近满载，温度和功耗会显著升高。
- 请确保显卡散热器、风扇和机箱风道正常。
- 不建议无人值守运行强度 4。
- 若网页失去响应，可在终端执行以下命令停止压测：

```bash
pkill -TERM -f "/GPU-Press/gpu_burn"
```

- 如果驱动崩溃或系统完全无响应，只能通过系统重启恢复；请先从短时、低强度测试开始。

## Web 依赖说明

前端通过 jsDelivr 加载 Chart.js。首次打开页面需要能够访问：

```text
https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js
```

如果浏览器无法联网，控制按钮仍可显示，但曲线组件不会加载。此时可下载 Chart.js 到本地并修改 `templates/index.html` 中的 `<script src=...>` 路径。

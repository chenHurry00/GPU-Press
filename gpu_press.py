"""
gpu_press.py —— GPU 压力测试 Web 控制台后端

职责（单一职责）：
  - 调度编译好的 gpu_burn 二进制（启动 / 停止）
  - 周期采样 nvidia-smi 的温度 / 功耗 / 利用率 / 显存，写入环形缓冲
  - 通过 Flask + SSE（Server-Sent Events）把实时样本推给前端

前端为静态页面（templates/index.html + static/），通过 fetch /api/start
/api/stop 与 EventSource(/stream) 交互，绘制每张 GPU 的实时状态曲线。

运行：python gpu_press.py  然后 http://127.0.0.1:8765
"""

import atexit
import csv
import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime

from flask import Flask, Response, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BURN_BIN = os.path.join(BASE_DIR, "gpu_burn")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# ---------- 状态管理（线程安全） ----------
SAMPLE_CAP = 1800          # 保留最近 30 分钟（@1Hz）
sample_batches = deque(maxlen=SAMPLE_CAP)
state_lock = threading.Lock()
gpu_inventory = {}         # nvidia-smi index -> 最新设备信息
sample_sequence = 0        # SSE 使用的单调递增序号
burn_procs = {}            # nvidia-smi index -> subprocess.Popen
active_gpu_indices = []    # 本轮正在压测的设备
running = False
session_start = None       # 本轮启动时间戳
last_burn_error = None     # 压测进程最近一次异常退出信息


def now_ts():
    return datetime.now().strftime("%H:%M:%S")


# ---------- GPU 采样：通过 nvidia-smi 查询 CSV ----------
def query_gpus():
    """返回所有 GPU 的当前样本列表；失败时返回 None。"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            timeout=4,
        ).decode(errors="replace").strip()
        sampled_at = time.time()
        clock = now_ts()
        result = []
        for parts in csv.reader(out.splitlines(), skipinitialspace=True):
            if len(parts) != 11:
                raise ValueError(f"nvidia-smi 返回字段数异常: {len(parts)}")
            result.append({
                "time": sampled_at,
                "clock": clock,
                "gpu_index": int(parts[0].strip()),
                "gpu_uuid": parts[1].strip(),
                "gpu_name": parts[2].strip(),
                "driver": parts[3].strip(),
                "temp": float(parts[4]),
                "power": float(parts[5]),
                "power_limit": float(parts[6]),
                "util": float(parts[7]),
                "mem_used": float(parts[8]),
                "mem_total": float(parts[9]),
                "sm_clock": float(parts[10]),
            })
        return result
    except Exception as e:
        print("[sampler] query failed:", e)
        return None


def sampler_loop():
    """后台线程：每秒采样所有 GPU，把快照塞进环形缓冲。"""
    global sample_sequence
    while True:
        batch = query_gpus()
        if batch is not None:
            with state_lock:
                sample_batches.append(batch)
                gpu_inventory.clear()
                gpu_inventory.update({s["gpu_index"]: {
                    "index": s["gpu_index"],
                    "uuid": s["gpu_uuid"],
                    "name": s["gpu_name"],
                    "driver": s["driver"],
                    "memory_total": s["mem_total"],
                } for s in batch})
                sample_sequence += 1
        time.sleep(1.0)


# ---------- 压力测试进程控制 ----------
def is_binary_ready():
    return os.path.isfile(BURN_BIN) and os.access(BURN_BIN, os.X_OK)


def refresh_process_state():
    """同步子进程状态，处理定时结束或异常退出。调用方必须持有 state_lock。"""
    global running, active_gpu_indices, last_burn_error
    if not burn_procs:
        running = False
        return
    errors = []
    for gpu_index, proc in list(burn_procs.items()):
        return_code = proc.poll()
        if return_code is None:
            continue
        if return_code != 0:
            stderr = proc.stderr.read().decode(errors="replace").strip()
            detail = stderr or f"进程异常退出（code={return_code}）"
            errors.append(f"GPU {gpu_index}: {detail}")
        del burn_procs[gpu_index]
    if errors:
        last_burn_error = "\n".join(errors)
    running = bool(burn_procs)
    active_gpu_indices = sorted(burn_procs)


def start_burn(seconds, intensity, gpu_indices):
    global running, session_start, active_gpu_indices, last_burn_error
    with state_lock:
        refresh_process_state()
        if running:
            return False, "已经在运行中"
        if not is_binary_ready():
            return False, "gpu_burn 二进制不存在或不可执行，请先运行 build.sh"
        missing = [index for index in gpu_indices if index not in gpu_inventory]
        if missing:
            return False, f"GPU 不存在或尚未检测到: {missing}"

        started = {}
        try:
            for gpu_index in gpu_indices:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_inventory[gpu_index]["uuid"]
                started[gpu_index] = subprocess.Popen(
                    [BURN_BIN, str(seconds), str(intensity)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                )
        except Exception as e:
            for proc in started.values():
                proc.terminate()
                proc.wait(timeout=3)
            return False, f"启动压测进程失败: {e}"

        burn_procs.update(started)
        running = True
        active_gpu_indices = sorted(gpu_indices)
        session_start = time.time()
        last_burn_error = None
        return True, f"已启动 {len(gpu_indices)} 张 GPU"


def stop_burn():
    global running, active_gpu_indices
    with state_lock:
        refresh_process_state()
        if not burn_procs:
            running = False
            return False, "未在运行"
        # 先 SIGTERM，当前短内核结束后退出；超时再 SIGKILL。
        procs = list(burn_procs.values())
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        burn_procs.clear()
        running = False
        active_gpu_indices = []
        return True, "已停止"


def cleanup():
    """Web 服务退出时确保压测子进程不会残留。"""
    stop_burn()


# ---------- Flask 路由 ----------
@app.route("/")
def index():
    return send_from_directory(TEMPLATE_DIR, "index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    try:
        seconds = int(data.get("seconds", 60))
        intensity = int(data.get("intensity", 2))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="seconds 和 intensity 必须是整数"), 400
    if not (0 <= seconds <= 3600):
        return jsonify(ok=False, error="seconds 范围 0..3600"), 400
    if not (1 <= intensity <= 4):
        return jsonify(ok=False, error="intensity 范围 1..4"), 400
    gpu_indices = data.get("gpu_indices")
    if (not isinstance(gpu_indices, list) or not gpu_indices
            or any(isinstance(index, bool) or not isinstance(index, int)
                   for index in gpu_indices)):
        return jsonify(ok=False, error="gpu_indices 必须是非空整数数组"), 400
    if len(set(gpu_indices)) != len(gpu_indices):
        return jsonify(ok=False, error="gpu_indices 不能包含重复设备"), 400
    ok, msg = start_burn(seconds, intensity, gpu_indices)
    return jsonify(ok=ok, msg=msg, running=running)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    ok, msg = stop_burn()
    return jsonify(ok=ok, msg=msg, running=running)


@app.route("/api/status")
def api_status():
    with state_lock:
        refresh_process_state()
        snap = list(sample_batches)[-120:]  # 最近 2 分钟快照
        resp = {
            "ok": True,
            "running": running,
            "binary_ready": is_binary_ready(),
            "session_elapsed": (time.time() - session_start) if running and session_start else 0,
            "last_error": last_burn_error,
            "gpus": [gpu_inventory[index] for index in sorted(gpu_inventory)],
            "active_gpu_indices": active_gpu_indices,
            "samples": snap,
            "latest": snap[-1] if snap else None,
        }
    return jsonify(resp)


@app.route("/stream")
def stream():
    """SSE：每秒推送最新的多 GPU 快照。"""
    def gen():
        with state_lock:
            last_sequence = sample_sequence
        while True:
            with state_lock:
                refresh_process_state()
                current_sequence = sample_sequence
                if current_sequence > last_sequence and sample_batches:
                    batch = sample_batches[-1]
                else:
                    batch = None
                last_sequence = current_sequence
            if batch is not None:
                yield f"data: {json.dumps({'samples': batch})}\n\n"
            # 同时附带运行状态变化（前端据此刷新按钮）
            # 简化：状态由 /api/status 轮询，SSE 仅推数据
            time.sleep(1.0)
    return Response(gen(), mimetype="text/event-stream")


# ---------- 入口 ----------
def ensure_built():
    if not is_binary_ready():
        print("[warn] gpu_burn 未编译，将仅显示温度曲线（无压力负载）。")
        print("       请运行: bash build.sh")


def main():
    ensure_built()
    atexit.register(cleanup)
    # 启动采样线程
    t = threading.Thread(target=sampler_loop, daemon=True)
    t.start()

    port = int(os.environ.get("GPU_PRESS_PORT", "8765"))
    print(f"GPU-Press 已启动: http://127.0.0.1:{port}")
    # 关闭 Werkzeug 默认日志噪声
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()

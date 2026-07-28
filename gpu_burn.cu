/**
 * gpu_burn.cu —— GPU 压力测试内核
 *
 * 设计目标：让 GPU 的 FP32/FP64 计算单元与显存带宽持续高负载，
 * 触发最高功耗与温度，用于稳定性验证。
 *
 * 实现思路（KISS）：
 *   - 每个线程在寄存器中反复执行 FMA（fused multiply-add）循环，
 *     故意不做累加归约，迫使计算单元满载且不被编译器优化掉。
 *   - 使用 volatile 防止编译器消除"无用"迭代。
 *   - 周期性地写回显存以施加显存带宽压力。
 *
 * 编译：
 *   nvcc -O3 -arch=sm_86 -o gpu_burn gpu_burn.cu
 *   (sm_86 = RTX 3070 / Ampere；其它架构按需调整)
 *
 * 用法：
 *   ./gpu_burn <seconds> <intensity>
 *     seconds    运行时长（秒），0 表示直到被 SIGTERM/SIGINT
 *     intensity  1..4，控制每线程内层迭代次数
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <unistd.h>
#include <cuda_runtime.h>

// volatile 阻止编译器把循环优化掉
static volatile sig_atomic_t g_stop = 0;

static void handle_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

// 检查 CUDA 调用
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 单线程的负载内核：纯 FMA 暴力循环
__global__ void burn_kernel(volatile float *sink, int iters, unsigned long long seed) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 初始数据（每个线程不同，避免简单合并）
    float a = (float)((seed + tid) & 0xFFFF) * 1.41421356f + 0.12345f;
    float b = (float)((seed ^ tid) & 0xFFFF) * 2.71828182f + 0.98765f;
    float c = (float)((seed - tid) & 0xFFFF) * 3.14159265f + 0.55555f;
    float d = a;

    // iters 取 intensity * 常数，内层手工展开 8 次 FMA
    #pragma unroll 8
    for (int i = 0; i < iters; ++i) {
        // 故意不做任何收敛，纯计算压力
        d = d * a + b;   // FMA 1
        d = d * c + a;   // FMA 2
        d = d * b + c;   // FMA 3
        d = d * a + b;   // FMA 4
        d = d * c + a;   // FMA 5
        d = d * b + c;   // FMA 6
        d = d * a + b;   // FMA 7
        d = d * c + a;   // FMA 8
        // 周期性写回显存（施加带宽压力），每隔 1024 次外层迭代一次
        if ((i & 1023) == 0) {
            sink[tid & 0xFFFFF] = d;
        }
    }
    // 最终写回，确保 sink 被使用
    sink[tid & 0xFFFFF] = d;
}

int main(int argc, char **argv) {
    int seconds = 0;       // 0 = 直到收到信号
    int intensity = 2;     // 默认中等强度

    if (argc >= 2) seconds = atoi(argv[1]);
    if (argc >= 3) {
        intensity = atoi(argv[2]);
        if (intensity < 1) intensity = 1;
        if (intensity > 4) intensity = 4;
    }

    // 安装信号处理：SIGINT / SIGTERM 优雅退出
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_signal;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // 选 GPU 0
    CUDA_CHECK(cudaSetDevice(0));

    // 分配显存缓冲（约 4MB，足够施加写压力）
    const size_t sink_bytes = 1u << 22;   // 4 MiB
    float *d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sink_bytes));
    CUDA_CHECK(cudaMemset(d_sink, 0, sink_bytes));

    // 根据 GPU 的 SM 数量配置网格。每轮内核控制在较短时间内完成，
    // 这样收到 SIGTERM 后能在当前轮结束时快速退出，而不会长时间卡住。
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int threads_per_block = 256;
    const int blocks = prop.multiProcessorCount * 8;
    const int iters = intensity * 2048;

    // 预热一次（避免首次启动冷启动延迟影响读数）
    burn_kernel<<<blocks, threads_per_block>>>(d_sink, iters, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    fprintf(stderr, "GPU: %s, SMs: %d, intensity: %d\n",
            prop.name, prop.multiProcessorCount, intensity);

    // 主循环：持续启动内核直到时间到或收到停止信号
    unsigned long long seed = 0xABCDEFull;
    time_t start = time(nullptr);
    bool failed = false;
    for (;;) {
        if (g_stop) break;
        if (seconds > 0 && (time(nullptr) - start) >= seconds) break;

        burn_kernel<<<blocks, threads_per_block>>>(d_sink, iters, seed++);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            failed = true;
            break;
        }
        // 每轮同步以控制停止延迟；单轮内核足够短，不影响持续满载。
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
            failed = true;
            break;
        }
    }
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "Final synchronization failed: %s\n", cudaGetErrorString(sync_err));
        failed = true;
    }

    cudaFree(d_sink);
    if (failed) {
        return EXIT_FAILURE;
    }
    // 标准输出留一行成功标记，便于后端判活
    printf("gpu_burn finished\n");
    return EXIT_SUCCESS;
}

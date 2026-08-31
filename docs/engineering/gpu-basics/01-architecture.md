---
title: 1 · GPU 里面有什么
description: 用一张资源图理解 SM、Tensor Core、缓存、HBM、copy engine 与互联。
type: series
status: stable
level: beginner
updated: 2026-08-27
tags: [gpu, sm, tensor-core, hbm]
---

# GPU 里面有什么

<div class="notebook-hero" markdown>

<span class="chapter-kicker">GPU 性能基础 · 01</span>

GPU 不是一块只有“算力”的黑盒。它同时包含计算单元、存储层次、数据搬运引擎和对外互联；一次操作快不快，取决于数据在哪一层、由谁处理、经过哪条路径。

</div>

## 一张简化地图

![GPU 的计算、存储与互联资源地图](assets/gpu-resource-map.svg)

*数据通常从 HBM 经过 L2 和 SM 内部存储到达执行单元；跨设备数据还要经过 copy engine、NVLink 或 PCIe。*

| 资源 | 保存或执行什么 | 性能问题 |
| --- | --- | --- |
| SM | warp、CUDA Core、Tensor Core | 工作是否足够并行，是否被通信占用 |
| Register / Shared Memory / L1 | SM 附近的少量高速数据 | 容量有限，使用过多会限制并发 |
| L2 | 所有 SM 共享的片上缓存 | 数据能否复用，是否反复访问 HBM |
| HBM | 参数、激活、梯度和通信 buffer | 容量与带宽通常都是关键约束 |
| copy engine | 部分 DMA 拷贝 | 能否与计算并发，源和目标链路是否空闲 |
| NVLink / PCIe | GPU 对外数据路径 | 决定 GPU 间或 CPU–GPU 搬运速度 |

## 软件工作怎样落到 SM

![CUDA grid、block、warp、thread 与 SM 的映射关系](assets/execution-hierarchy.svg)

*Kernel 启动一组 thread block；block 被分配到 SM，SM 再以 32 个线程组成的 warp 为基本调度单位。*

- **Kernel**：在 GPU 上执行的函数。
- **Grid**：一次 kernel 启动产生的全部线程。
- **Thread block**：被整体分配给某个 SM 的线程组。
- **Thread**：kernel 的一个执行实例，通常负责一个或几个数据元素。
- **Warp**：SM 实际调度的一组 32 个线程。

例如，对 256 个元素执行 `c = a + b`，kernel 可以选择每个 block 启动 128 个线程，于是整个 grid 包含 2 个 block，每个 block 又被硬件分成 4 个 warp：

```text
Block 0：thread   0～127 → 4 个 Warp
Block 1：thread 128～255 → 4 个 Warp
```

### 软件与硬件的边界

```text
CPU 启动 Kernel                                      ← 软件
     │
     │ 指定：Grid 大小、Block 大小、参数、Stream
     ▼
────────────── 软件给出约束，硬件接管调度 ──────────────
     ▼
GPU 收到整个 Grid                                    ← 硬件
     │
     │ 选择哪些 Block 可以开始运行
     ▼
Block 被分配到某个 SM
     │
     │ Block 按每 32 个线程组成多个 Warp
     ▼
SM 的 Warp Scheduler 选择已经就绪的 Warp
     │
     ├─ 读取 / 写回 ──→ Load/Store Unit
     ├─ 普通 FP32 计算 → CUDA Core
     └─ 矩阵乘加 ─────→ Tensor Core
```

一个 block 开始执行后通常留在同一个 SM，直到完成并释放资源。软件不能假设 block 的执行顺序、精确落点或线程与 CUDA Core 的固定绑定。

每个 block 的线程数由 kernel 实现指定；使用 PyTorch、cuBLAS 等库时，这个选择通常隐藏在算子内部。它一般取 32 的整数倍，常见起点是 128 或 256，同时受 GPU 的单 block 线程上限、寄存器、Shared Memory 和数据形状限制。block 太小可能无法提供足够并行工作，太大又可能占用过多资源，使一个 SM 能同时容纳的 block 变少。

!!! tip "先形成这个直觉"

    Tensor 在 HBM 中，计算在 SM 中。任何计算都要先把数据送到 SM 附近，再把结果写回；“峰值 FLOPS 很高”并不保证数据送得足够快。

## 权威参考

- [NVIDIA GPU Architecture Fundamentals](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-architecture-fundamentals)
- [CUDA Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

[← 专题总览](index.md) · [→ 一次计算消耗什么](02-compute.md)

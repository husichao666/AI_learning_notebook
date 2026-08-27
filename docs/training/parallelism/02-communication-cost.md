---
title: "3.2 · 通信开销基础"
description: "从启动延迟、有效带宽、本地数据搬运和计算通信重叠出发，建立分析集合通信性能的基本模型。"
type: series
status: stable
level: intermediate
updated: 2026-08-27
tags: [distributed-training, communication, collective, performance]
---

# 一次集合通信到底花在哪里

<div class="notebook-hero" markdown>

<span class="chapter-kicker">第 3 章 · 模型状态分片</span>

上一节已经知道，FSDP 会在每个 step 中执行参数 All-Gather（AG）和梯度 Reduce-Scatter（RS）。但“传输多少字节”并不能直接回答“训练会慢多少”：同样大小的张量，在节点内 NVLink 和跨节点 RoCE 上可能耗时不同；同一次通信，真正拖慢 step 的也可能不是网络，而是通信前后的数据复制或等待。本节先建立一套简单的通信成本模型，为后续几种 FSDP 实现中的融合、零拷贝和流水线优化做准备。

</div>

!!! note "与第 1 章的关系"

    [第 1 章](00-foundations.md)介绍 All-Gather、Reduce-Scatter、All-Reduce 的语义以及 ring 等集合通信算法。本节不再解释“通信结果是什么”，而是关注“完成这次通信需要付出哪些时间”。

!!! tip "还不熟悉 GPU 硬件？"

    可以先阅读 [GPU 性能基础专题](../../engineering/gpu-basics/index.md)，用图建立 SM、HBM、copy engine、NVLink 和 CUDA stream 的基本认识，再回到本节分析通信成本。

## 01 · 一次 collective 不只是链路传输 { #cost-components }

集合通信（collective communication）是一个通信组内所有 rank 共同参与的操作。一次 `all_gather` 或 `reduce_scatter` 从输入就绪到结果可用，通常会依次经历：

```text
输入张量就绪
  → 等待数据依赖
  → 可选的 pack / dtype 转换
  → Host 下发 collective
  → 通信 kernel 搬运或归约
  → 可选的 unpack / 重排
  → 使用方等待结果
```

因此，总耗时可以粗略拆成：

\[
T_{\text{comm}}
\approx
T_{\text{launch}}
+T_{\text{pack}}
+T_{\text{transfer}}
+T_{\text{reduce}}
+T_{\text{unpack}}
+T_{\text{sync}}.
\]

各项含义如下：

| 开销 | 来自哪里 | 什么时候容易成为瓶颈 |
| --- | --- | --- |
| `launch` | Host 调用、stream 排队、通信 kernel 启动 | 参数很多、每条消息很小时 |
| `pack / unpack` | 把多个参数复制到连续 buffer，或把结果拆回参数 buffer | 链路很快、copy 暴露在关键路径时 |
| `transfer` | 数据经过 NVLink、PCIe、InfiniBand 或 RoCE | 消息较大、跨节点带宽较低时 |
| `reduce` | Reduce-Scatter / All-Reduce 中的求和 | 归约数据量大，或通信 kernel 占用较多 SM 时 |
| `sync` | 计算流等待通信完成、回收或复用 buffer | 预取太晚，或计算窗口不足以覆盖通信时 |

这里的“一次 collective”也不等于底层只发送一个网络包。Ring、tree 或分层算法会把一个 collective 拆成多轮、多个 peer 之间的数据交换，通信组大小和算法都会影响耗时。

## 02 · 固定延迟与有效带宽 { #latency-bandwidth }

最常用的近似是 \(\alpha\)-\(\beta\) 模型：

\[
T \approx \alpha + \frac{V}{B_{\text{eff}}}.
\]

- \(\alpha\) 表示一次通信步骤的固定延迟，包括调度、协议和同步等与消息大小关系不大的成本；
- \(V\) 表示实际经过链路的数据量；
- \(B_{\text{eff}}\) 表示有效带宽，它通常低于硬件标称峰值；有时也写成每字节时间 \(\beta=1/B_{\text{eff}}\)。

小消息中 \(\alpha\) 占主导，继续减小张量几乎不会缩短时间；大消息中 \(V/B_{\text{eff}}\) 占主导，提高链路利用率或降低通信 dtype 才更重要。这就是融合通信主要帮助小消息，而低精度通信主要减少大消息传输时间的原因。

以 ring 算法为例，设通信组有 \(N\) 个 rank，\(P\) 是完整逻辑张量的字节数，忽略分块流水和归约计算的细节，可以近似写成：

\[
T_{\text{AG/RS}}
\approx
(N-1)\alpha
+\frac{N-1}{N}\frac{P}{B_{\text{eff}}},
\]

\[
T_{\text{AR}}
\approx
2(N-1)\alpha
+2\frac{N-1}{N}\frac{P}{B_{\text{eff}}}.
\]

All-Reduce（AR）可以看成 Reduce-Scatter 加 All-Gather，因此 ring AR 的轮数和链路字节大约都是单独 AG/RS 的两倍。公式用于建立数量级直觉，不是 profiler 预测器：真实通信库可能选择 tree、分层算法或不同协议，多机网络还会经过节点内与节点间两个带宽不同的阶段。

## 03 · 开销和 GPU 有关，但不只由 GPU 决定 { #hardware }

同一组张量的通信时间由整条数据路径共同决定：

| 层次 | 关键因素 | 主要影响 |
| --- | --- | --- |
| GPU | HBM 带宽、SM 数量、copy engine、GPU 代际 | pack/unpack 速度，以及通信 kernel 与 GEMM 的资源竞争 |
| GPU 互联 | NVLink / NVSwitch 代际与拓扑 | 节点内或超节点内的带宽和时延 |
| GPU 到 NIC | PCIe/NVLink 路径、NUMA 亲和性、GPUDirect RDMA | 跨节点数据能否高效地从 GPU 显存进入网卡 |
| 网络 | NIC 数量与速率、InfiniBand/RoCE、交换层级、拥塞 | 跨节点有效带宽、尾延迟和可扩展性 |
| 软件 | 通信库算法、协议、bucket 大小、stream 调度 | collective 轮数、启动开销以及重叠效果 |

因此，不能只用“GPU 型号”推断通信性能。即使 GPU 相同，NIC 配比、交换网络、进程绑定和通信组映射不同，也可能让结果相差数倍。RoCE 也不等于天然低效；这里真正关心的是，相比节点内高速互联，跨节点通信通常具有更高延迟、更低的单卡有效带宽，也更容易受到拥塞影响。

## 04 · 高速链路上，本地 copy 也可能很贵 { #local-copy }

假设一次通信前要把 100 MB 参数装入融合 buffer。一次 copy 至少包含 100 MB 读取和 100 MB 写入。若有效 HBM 带宽为 1 TB/s，只看数据搬运的理想下界也约为：

\[
T_{\text{copy}}
\approx
\frac{100\ \text{MB}+100\ \text{MB}}{1\ \text{TB/s}}
=0.2\ \text{ms}.
\]

若通信前 copy-in、通信后又 copy-out，两次复制的理想下界约为 0.4 ms，实际还要加 kernel 启动和资源竞争。作为对照，8 个 rank 对一个 100 MB 完整张量做 ring AG，只计算带宽项：

| 有效链路带宽 | 近似带宽时间 \(\frac{7}{8}\frac{100\text{ MB}}{B_{\text{eff}}}\) |
| --- | --- |
| 100 GB/s | 约 0.875 ms |
| 25 GB/s | 约 3.5 ms |

在第一种高速通信域中，0.4 ms 的本地复制已经不可忽略；在第二种较慢通信域中，把很多小 collective 融成大消息带来的收益可能更重要。这里的数字只用于比较数量级，不代表某种 GPU 或网络的实测结果。

这也解释了后文为什么反复区分三种数据布局：每轮临时拼接、初始化时建立固定连续 buffer，以及让 collective 直接读写参数自己的 buffer。它们的理论网络字节可以相同，本地 HBM 流量和临时显存却不同。

## 05 · 逐参数还是融合，要看消息与拓扑 { #fusion }

通信融合（communication fusion）是把多个参数或梯度装入一个较大的 bucket，再发起一次 collective。它不一定减少 FSDP 的有效网络字节，主要是在两类成本之间做交换：

| 路径 | 收益 | 代价 | 更可能适合 |
| --- | --- | --- | --- |
| 逐参数通信 | 可以直接读写参数 buffer，少做 pack/unpack，临时显存较少 | collective 和 Host launch 更多，小消息可能打不满带宽 | 高带宽低时延域，且单参数消息足够大 |
| 融合通信 | collective 更少，大消息更容易获得较高有效带宽 | 可能需要融合 buffer、copy-in/copy-out，并推迟释放 | 跨节点或小参数很多的场景 |

“融合”也不必然等于“每轮执行 `cat`”。Megatron-FSDP 可以在初始化时就把参数放进固定 offset 的连续 buffer；HyperParallel 的 HSDP 路径还可以让逐参数 RS 直接写入融合 AR buffer 的各个 view。两者都保留了大消息通信的好处，同时减少运行时的重复拼接。

## 06 · overlap 隐藏的是等待，不是通信量 { #overlap }

如果通信与一段没有数据依赖的计算并行，理想情况下暴露在关键路径上的时间近似为：

\[
T_{\text{exposed}}
\approx
\max\left(0, T_{\text{comm}}-T_{\text{overlap window}}\right).
\]

例如通信需要 2 ms，后面有 1.6 ms 独立计算，那么理想暴露时间约为 0.4 ms。FSDP 的参数预取、梯度异步 Reduce-Scatter，以及 HSDP 的跨 unit RS/AR 流水，都是在扩大这个可覆盖窗口。

但 overlap 不是免费的：

- 预取会让当前 unit 和下一个 unit 的完整参数同时存在，增加峰值显存；
- NCCL 等通信 kernel 可能与 GEMM 争用 SM 或 HBM 带宽，使两者并发后都变慢；
- 第一段通信没有更早的计算可覆盖，最后一段通信也可能在反向结束后留下尾巴；
- stream 上已经排入异步操作，不代表使用结果时一定不需要等待。

因此，时间线上“通信块与计算块重叠”只是必要条件。最终要看 step time 是否下降，以及等待点是否真正移出了关键路径。

## 07 · 用哪些指标判断优化是否有效 { #measurement }

分析一次通信时，至少要同时观察：

1. **消息大小与 collective 次数**：判断固定启动开销是否占主导；
2. **有效带宽**：区分链路没有跑满，还是通信量本身已经太大；
3. **pack/unpack kernel**：确认网络之外是否还有明显的 HBM 搬运；
4. **暴露等待时间**：找到计算流真正等待 AG、RS 或 AR 的位置；
5. **资源竞争与峰值显存**：检查 overlap 是否拖慢 GEMM，以及预取或 bucket 是否增加过多临时存储；
6. **端到端 step time**：这是所有局部优化最终必须改善的指标。

在 NVIDIA/NCCL 环境中，可以先用 `nccl-tests` 测量目标拓扑下不同消息大小的延迟和带宽，再用 Nsight Systems 或框架 profiler 查看真实训练中的 copy、collective、计算和等待。不要直接把标称 NVLink/NIC 带宽代入公式后当作训练结果。

!!! tip "一句话总结"

    一次集合通信的成本不只是“字节数 ÷ 网卡带宽”，还包括固定启动延迟、本地 pack/unpack、归约计算和同步等待。高速域更值得减少本地 copy，慢速或小消息域更值得融合通信，而 overlap 负责减少最终暴露在 step 关键路径上的时间。

[→ 继续阅读 3.3 · Megatron 实现方案](02-megatron-fsdp.md)

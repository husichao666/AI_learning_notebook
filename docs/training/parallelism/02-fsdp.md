---
title: "3.1 · ZeRO 与 FSDP"
description: "单纯的DP，让每张卡上都持有所有的模型参数，那为什么不让每张卡各自存其中的一部分呢？"
type: series
status: stable
level: intermediate
updated: 2026-08-25
tags: [distributed-training, fsdp, zero]
---

# 把模型状态也切开：ZeRO 与 FSDP

<div class="notebook-hero" markdown>

<span class="chapter-kicker">第 3 章 · 模型状态分片</span>

单纯的DP，让每张卡上都持有所有的模型参数，这里存在较大的冗余，那为什么不让每张卡各自存其中的一部分呢？

**本章关键词：** 📦 ZeRO 1/2/3 三阶段 · 🔄 all-gather 参数 + reduce-scatter 梯度 · 📉 显存降到 1/N · 🟢 Megatron DistributedOptimizer

</div>


## 01 · 动机：DP 的冗余 { #motivation }

上一章的结论：Megatron 默认 BF16 + Adam 的纯 DDP 下，每张卡都存**完整的**BF16 参数（$2P$）、FP32 `main_grad`（$4P$）、优化器状态（$12P$），共 $18P$ 字节，完全不随卡数下降。但仔细想——这 N 份副本**内容完全相同**，为什么每张卡都要存一整份？

ZeRO（Zero Redundancy Optimizer）的答案：**不存了，切开**。每张卡只持有 $1/N$ 的模型状态，需要完整参数时临时用 `all-gather` 拼起来，用完即丢。这样显存随卡数线性下降，代价只是多一点通信。


## 02 · ZeRO 三阶段：一刀一刀往下切 { #stages }

ZeRO 切的是**静态显存**——每 step 都常驻、内容在各卡完全相同的那三块「模型状态」（参数、梯度、优化器状态）。它**不碰动态显存**（激活值，那是每卡各不相同的中间结果，得靠激活重计算 / 序列并行去省）。三个阶段按「切得越来越多」排序，先切最大的那块（优化器状态占 $12/18 \approx 66.7\%$）：

| 阶段 | 在 ZeRO-1 基础上再切什么 | 每卡静态显存 | 对应实现 |
| --- | --- | --- | --- |
| baseline DDP | 啥都不切 | $2P + 4P + 12P = 18P$ | 普通 DP |
| **ZeRO-1** | 切**优化器状态** | $2P + 4P + 12P/N = 6P + 12P/N$ | Megatron `DistributedOptimizer` · Megatron-FSDP `optim` |
| **ZeRO-2** | 再切**梯度** | $2P + (4P + 12P)/N = 2P + 16P/N$ | Megatron-FSDP `optim_grads` · DeepSpeed |
| **ZeRO-3** | 再切**参数** | $(2P+4P+12P)/N = 18P/N$ | **FSDP**（PyTorch `fully_shard`）· Megatron-FSDP `optim_grads_params` |


![ZeRO 三阶段每卡静态显存递减（N=8）](assets/02-fsdp-figure-01.svg)

*ZeRO 一刀一刀往下切「静态显存」，N 越大单卡省得越多。Megatron 的 DistributedOptimizer 做 ZeRO-1；Megatron-FSDP 与 PyTorch FSDP 可做到 ZeRO-2/3。*



## 03 · 一个 step 的流程：ZeRO-1 / 2 / 3 逐级变化 { #flow }

三个阶段每个 step 都是同一套骨架——**forward → backward → 同步梯度 → 更新 → 拼回完整参数**——区别只在**谁被切、何时做 all-gather**。逐个看：


### ZeRO-1：参数副本与梯度 buffer 仍是全尺寸，只切优化器状态

每张卡常驻完整 BF16 参数，并为全部参数分配全尺寸 FP32 梯度 buffer。Megatron DistributedOptimizer 用 reduce-scatter 同步梯度：通信后每个 rank 只得到 $1/N$ 的归约梯度，用它更新对应的 $1/N$ 参数、master 参数和 Adam 状态，再通过 all-gather 拼回完整 BF16 参数。

这里的“梯度未分片”指**显存分配**而非通信输出：ZeRO-1 虽然只使用 $1/N$ 的梯度，但全尺寸梯度 buffer 仍占 $4P$；ZeRO-2 才会释放非本 rank 的梯度，使梯度存储降到 $4P/N$。


![ZeRO-1 一个 step 的流程](assets/02-fsdp-figure-02.svg)

*ZeRO-1（Megatron 实现）：reduce-scatter 后每个 rank 只使用 $1/N$ 的最终梯度并更新对应参数分片，再 all-gather 拼回完整参数；但梯度 buffer 仍按全尺寸分配。*



### ZeRO-2：backward 边算边 reduce-scatter，梯度也降到 1/N

唯一的变化在梯度存储：反向每算完一段梯度就立即 reduce-scatter，每张卡**只保留自己那 $1/N$ 的归约结果，其余部分随即释放**，然后继续计算下一段。这样不会在每张卡上累积出完整梯度，FP32 梯度显存从 $4P$ 降到 $4P/N$。参数依然全量常驻，前向/反向不需要临时 all-gather 参数；完成分片更新后仍需同步更新后的参数。


![ZeRO-2 一个 step 的流程](assets/02-fsdp-figure-03.svg)

*ZeRO-2：反向每算完一段梯度就立即 reduce-scatter，只留下本 rank 的分片；参数依旧全量常驻。*



### ZeRO-3 = FSDP：连参数也切，用到哪层才 all-gather 哪层

FSDP 的核心节奏：**参数平时是切碎的，用到哪层才把那层 all-gather 成完整，算完立刻丢回切片态**。代价是**前向也要 all-gather 一次参数**；若暂时忽略 dtype、按等宽元素计数，通信由 DDP 的 2P 升到 3P。换来的是参数也降到 $1/N$。


![FSDP 一层的前向反向通信序列](assets/02-fsdp-figure-04.svg)

*FSDP 一个 FSDP unit（通常一层）的通信序列：前向 all-gather→算→reshard，反向 all-gather→算→reduce-scatter。*



## 04 · 动态显存：ZeRO 省不到的那半边 { #dynmem }

上面几张流程图省的都是**静态显存**（模型状态）。但一个 step 的**显存峰值 = 静态 + 动态**，而 ZeRO 对动态那半边几乎无能为力——不单独分析，就会遇到「上了 FSDP 还是 OOM」。动态显存主要是三块：

| 动态显存 | 是什么 | ZeRO 能省吗 | 怎么省 |
| --- | --- | --- | --- |
| **激活值 activations** | 前向每层的中间结果，反向要用；$\propto$ batch × seq × 层数 × hidden | **不能**：每卡喂的数据不同、激活各不相同，无法沿 DP 维切 | 激活重计算 checkpointing、序列并行 SP（第 4 章）、上下文并行 CP（第 6 章）、offload |
| **all-gather 瞬时峰值** （仅 ZeRO-3） | 当前正在算的那个 FSDP unit，all-gather 出的完整参数缓冲 | 是 ZeRO-3 **新增**的开销，不是省 | 大小 ≈ 最大一层的完整参数（非整模型）；prefetch 可能使 1～2 个 unit 在同一时刻持有完整权重；`reshard_after_forward` 控制留不留 |
| **通信 / 碎片缓冲** | reduce-scatter / all-gather 的临时 buffer、显存碎片 | — | 调 bucket 大小、复用通信桶 |


![静态显存随卡数缩小、激活值不变，大 N 下激活主导峰值](assets/02-fsdp-figure-05.svg)

*ZeRO 把静态那截压扁，动态（激活）那截纹丝不动。卡越多，OOM 的元凶越是激活而非模型状态。*


!!! warning "⚠️ 常见误区"

    「上了 ZeRO-3/FSDP 显存就够了」——不对。在本章的 Megatron BF16 账本下，ZeRO 把**静态**降到 $18P/N$，但**激活值**不随卡数下降（还会因 global batch 变大而上升）。真实峰值经常由**激活值**主导，尤其长序列。所以 FSDP 几乎总要配**激活重计算**，长序列还得叠 SP / CP。


!!! tip "🔑 一个 step 的显存峰值（ZeRO-3）"

    峰值 ≈ **18P/N**（静态，已切）+ **激活值**（动态，不随 N 降）+ **1~2 个 unit 的完整参数**（all-gather 瞬时）+ 通信缓冲。
    ZeRO 只压第一项；后面 TP / PP / SP / CP 各章，很大程度上就是在压第二项。



## 05 · 通信量：1.5× 是等宽 dtype 下的元素口径 { #comm }

回忆第 1 章的恒等式 **all-reduce = reduce-scatter + all-gather**。先假设参数与梯度使用相同 dtype，并忽略环形通信共同的 $(N-1)/N$ 系数，按传输元素数计算：

- **DDP**：每 step 一次梯度 all-reduce ≈ $2P$（reduce-scatter $P$ + all-gather $P$）。
- **FSDP / ZeRO-3**：前向 all-gather 参数（$P$）+ 反向 all-gather 参数（$P$）+ 反向 reduce-scatter 梯度（$P$）= $3P$。

在这个简化口径下，$3P / 2P = 1.5$。但 Megatron 默认是 **BF16 参数 + FP32 梯度**，必须换算成字节：DDP 的 FP32 梯度 all-reduce 约为 $2 \times 4P = 8P$ 字节；FSDP 两次 BF16 参数 all-gather 加一次 FP32 梯度 reduce-scatter，约为 $2P + 2P + 4P = 8P$ 字节。因此默认 dtype 下不能宣称网络字节固定增加 50%。实际墙钟差异还取决于 collective 次数与粒度、拓扑、预取以及计算通信重叠程度。


## 06 · ZeRO-3 在四套实现中的对应关系 { #equiv }

ZeRO-3 描述的是一种分片语义：参数、梯度和优化器状态都沿 DP 维切成 $1/N$；计算前临时 all-gather 当前单元的参数，反向后用 reduce-scatter 留下本 rank 的梯度分片。DeepSpeed、PyTorch、Megatron 和 HyperParallel 都能实现这套语义，只是接口与工程侧重点不同：

| 实现 | ZeRO-3 入口 | 分片与通信边界 | 主要特点 |
| --- | --- | --- | --- |
| DeepSpeed ZeRO-3 | 配置 `zero_optimization.stage = 3` | 运行时按参数组管理 | 配置驱动，支持参数与优化器 offload |
| PyTorch FSDP2 | `fully_shard(module)` | 由传入的 module 决定 | 使用 per-parameter `DTensor`，与 PyTorch 原生接口结合紧密 |
| Megatron-FSDP | `--use-megatron-fsdp` + `--data-parallel-sharding-strategy optim_grads_params` | FSDP unit，通常是 Transformer 层 | 面向大模型训练，集成 TP、CP、EP、TransformerEngine 与通信重叠 |
| HyperParallel | `fully_shard(module, mesh=..., comm_fusion=...)` | FSDP unit；通信可选逐参数或按 bucket 融合 | 通过二维 HSDP mesh 映射分片与复制通信域，并优化 RS / AR 跨 unit 流水 |

需要注意，Megatron 的 `DistributedOptimizer` 与 `Megatron-FSDP` 不是同一条实现路径：前者只切优化器状态，属于 ZeRO-1；后者选择 `optim_grads_params` 时才完整对应 ZeRO-3。


!!! tip "✅ 学完自测"

    1. ZeRO-1/2/3 分别切什么？为什么先切优化器状态？
    2. FSDP 前向为什么用完参数要立刻 reshard 丢弃？省的是什么显存？
    3. FSDP/DDP 的 1.5× 通信比值基于什么 dtype 前提？换成 BF16 参数、FP32 梯度后，按字节应如何重算？
    4. Megatron 的 `DistributedOptimizer` 和 FSDP 等价吗？差在哪一块显存？

[→ 继续阅读 3.2 · 通信开销基础](02-communication-cost.md)

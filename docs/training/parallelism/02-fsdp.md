---
title: "3.1 · ZeRO 与 FSDP"
description: "从经典数据并行的模型状态冗余出发，理解 ZeRO 三阶段的分片范围、数据生命周期和通信口径。"
type: series
status: stable
level: intermediate
updated: 2026-09-02
tags: [distributed-training, fsdp, zero]
---

# 模型状态分片：ZeRO 与 FSDP

<div class="notebook-hero" markdown>

<span class="chapter-kicker">第 3 章 · 模型状态分片</span>

经典数据并行在每个 rank 上复制参数、梯度和优化器状态。ZeRO 沿数据并行维逐步切分这些模型状态；Fully Sharded Data Parallel（FSDP）则是在框架中实现完整参数分片语义的一类方案。

**本章关键词：** 📦 ZeRO 1/2/3 三阶段 · 🔄 all-gather 参数 + reduce-scatter 梯度 · 📉 模型状态理想降到 1/N · 🟢 Megatron DistributedOptimizer

</div>


## 01 · 动机：DP 的冗余 { #motivation }

沿用上一章的账本假设：在 BF16 参数、FP32 `main_grad` 与 Adam 主权重/动量状态的 Megatron DDP 路径中，每张卡分别为参数、梯度 buffer 和优化器状态分配 $2P$、$4P$ 与 $12P$ 字节，共 $18P$。这些对象都按完整模型尺寸分配；参数和优化器状态在同步数据并行 rank 间逻辑复制，梯度则在本地反向期间各自累积、归约后才一致。随着 DP 组增大，这部分每卡容量不会自动下降。

ZeRO（Zero Redundancy Optimizer）的思路是沿数据并行维消除这些副本。ZeRO-1 只切优化器状态，ZeRO-2 再切梯度，只有 ZeRO-3 才让参数以分片形式常驻，并在计算前临时 `all-gather` 当前单元的参数。理想静态显存随分片组大小下降，代价则取决于具体阶段的通信、临时 buffer 和调度方式。


## 02 · ZeRO 三阶段的分片范围 { #stages }

ZeRO 直接切分的是三类**模型状态**：参数、梯度与优化器状态。这里的“切分”指减少每个 rank 对完整对象的常驻容量，并不要求这些对象在 step 的每个时刻逐值相同；例如数据并行 rank 的局部梯度在归约前本来就不同。激活属于运行时中间结果，不在 ZeRO 三阶段的定义中，需要由激活重计算、SP 或 CP 等机制另行处理。三个阶段按模型状态的切分范围排序，并先从本账本中占比最大的优化器状态开始：

| 阶段 | 在 ZeRO-1 基础上再切什么 | 每卡模型状态理想容量 | 对应实现 |
| --- | --- | --- | --- |
| baseline DDP | 啥都不切 | $2P + 4P + 12P = 18P$ | 普通 DP |
| **ZeRO-1** | 切**优化器状态** | $2P + 4P + 12P/N = 6P + 12P/N$ | 经典 ZeRO-1；Megatron `DistributedOptimizer` 的常驻显存可按此口径理解 |
| **ZeRO-2** | 再切**梯度** | $2P + (4P + 12P)/N = 2P + 16P/N$ | 经典 ZeRO-2；Megatron-FSDP `optim_grads` · DeepSpeed |
| **ZeRO-3** | 再切**参数** | $(2P+4P+12P)/N = 18P/N$ | **FSDP**（PyTorch `fully_shard`）· Megatron-FSDP `optim_grads_params` |


![ZeRO 三阶段的每卡模型状态理想容量递减（N=8）](assets/02-fsdp-figure-01.svg)

*在本文 dtype 与无 padding 的理想账本中，ZeRO 逐阶段扩大模型状态分片范围。Megatron DistributedOptimizer 的常驻容量对应 ZeRO-1 口径；Megatron-FSDP 与 PyTorch FSDP 的具体配置可实现更完整的分片。*



## 03 · ZeRO-1 / 2 / 3 的 step 数据生命周期 { #flow }

三个阶段都执行前向、反向、梯度同步和参数更新，但只有采用分片参数更新且要求每个 rank 继续持有完整参数的实现，才需要在更新后同步参数；只有 ZeRO-3 需要在层计算前临时恢复参数。下面分别说明每一步当前保存什么、通信后留下什么。


### ZeRO-1：参数与梯度逻辑上复制，只切优化器状态

在经典 ZeRO-1 语义中，每个 rank 仍保存完整参数和完整的最终梯度，只把优化器状态分片。具体实现不必都使用同一通信序列。本节后续采用 Megatron DistributedOptimizer 作为例子：它为全部参数分配全尺寸 FP32 梯度 buffer，但用 reduce-scatter 让每个 rank 只消费 $1/N$ 的归约梯度，更新对应的 master 参数与 Adam 状态分片，再通过 all-gather 同步更新后的 BF16 参数。

因此，下图中的“梯度未分片”特指该 Megatron 路径的**常驻显存分配**，不是 reduce-scatter 的通信输出：通信输出只有 $1/N$，全尺寸梯度 buffer 仍占 $4P$。其他经典 ZeRO-1 实现也可以用 all-reduce 得到完整梯度；两者共享的是“梯度存储不分片”的算法口径。


![ZeRO-1 一个 step 的流程](assets/02-fsdp-figure-02.svg)

*ZeRO-1（Megatron 实现）：reduce-scatter 后每个 rank 只使用 $1/N$ 的最终梯度并更新对应参数分片，再 all-gather 拼回完整参数；但梯度 buffer 仍按全尺寸分配。*



### ZeRO-2：归约后只常驻梯度分片

相对 ZeRO-1，变化在最终梯度的存储口径：每个 rank 只需常驻自己负责的 $1/N$ 归约梯度。典型 bucketed 实现会在反向产生一段梯度后发起 reduce-scatter，并在通信安全完成后释放不再需要的完整输入区，从而避免把整模型最终梯度长期保留在每张卡上。FP32 梯度的理想常驻容量由 $4P$ 降到 $4P/N$。参数依然全量常驻，前向/反向不需要为参数分片临时 all-gather；分片更新后仍需把更新结果同步到各 rank 的完整参数副本。


![ZeRO-2 一个 step 的流程](assets/02-fsdp-figure-03.svg)

*ZeRO-2：反向每算完一段梯度就立即 reduce-scatter，只留下本 rank 的分片；参数依旧全量常驻。*



### ZeRO-3 与 full-shard FSDP：参数按单元临时恢复

ZeRO-3 定义了参数、梯度和优化器状态都分片的语义。PyTorch FSDP、Megatron-FSDP 等实现通常把模型划成若干 FSDP unit：参数平时以分片形式常驻，当前 unit 计算前才 all-gather 成完整参数，并按配置在计算后恢复为分片态。代价是前向也需要参数 all-gather；若暂时忽略 dtype、按等宽元素计数，典型及时 reshard 路径的通信由 DDP 的 $2P$ 变为 $3P$。


![FSDP 一层的前向反向通信序列](assets/02-fsdp-figure-04.svg)

*FSDP 一个 FSDP unit（通常一层）的通信序列：前向 all-gather→算→reshard，反向 all-gather→算→reduce-scatter。*



## 04 · ZeRO 与动态显存 { #dynmem }

上面几张流程图计算的是**常驻模型状态**。一个 step 的显存峰值还包含激活、临时完整参数、集合通信工作区与 allocator 保留空间；这些项不属于同一种生命周期，不能简单把“理想模型状态除以 $N$”当作整卡峰值。运行时的主要附加项包括：

| 运行时显存项 | 是什么 | ZeRO 是否直接切分 | 控制方式 |
| --- | --- | --- | --- |
| **激活值 activations** | 前向每层的中间结果，反向要用；$\propto$ batch × seq × 层数 × hidden | **否**：不属于 ZeRO 定义中的模型状态 | 激活重计算 checkpointing、序列并行 SP（第 4 章）、上下文并行 CP（第 6 章）、offload |
| **all-gather 瞬时峰值** （仅 ZeRO-3） | 当前正在算的那个 FSDP unit，all-gather 出的完整参数缓冲 | **否**：这是 full-shard 计算时的临时物化 | 大小约为当前或预取 unit 的完整参数；`reshard_after_forward` 与预取深度控制其生命周期 |
| **通信 / 碎片缓冲** | reduce-scatter / all-gather 的临时 buffer、显存碎片 | 无统一的 $1/N$ 结论 | 调整 bucket、限制在途 buffer、复用通信区，并观察 allocated / reserved 两种口径 |


![配置不变时，模型状态随分片组增大而缩小，激活更可能主导峰值](assets/02-fsdp-figure-05.svg)

*在 local batch、序列切分与重计算策略不变时，增大 ZeRO 分片组会压低静态模型状态，激活部分不会由 ZeRO 自动缩小，因而更可能成为峰值主项。*


!!! warning "⚠️ 常见误区"

    ZeRO-3/FSDP 只直接切分模型状态。在本章的 Megatron BF16 账本下，其理想静态下界为 $18P/N$；激活是否变化取决于 local batch、序列并行和重计算等独立配置，并不会仅因增大 FSDP 分片组而自动下降。长序列或较大 local batch 下，激活仍可能主导峰值，需要另行采用激活重计算、SP 或 CP。


!!! tip "🔑 一个 step 的显存峰值（ZeRO-3）"

    在本文账本与及时 reshard 路径下，可先写成：峰值 ≈ **18P/N**（模型状态理想项）+ **激活值** + **1～2 个 unit 的完整参数**（all-gather 瞬时）+ 通信缓冲。激活项不会仅因增大 ZeRO 分片组而下降；它是否变化还取决于 local batch、SP/CP 与重计算等配置。



## 05 · 通信量：1.5× 是等宽 dtype 下的元素口径 { #comm }

回忆第 1 章的恒等式 **all-reduce = reduce-scatter + all-gather**。先假设参数与梯度使用相同 dtype，并忽略环形通信共同的 $(N-1)/N$ 系数，按传输元素数计算：

- **DDP**：每 step 一次梯度 all-reduce ≈ $2P$（reduce-scatter $P$ + all-gather $P$）。
- **FSDP / ZeRO-3**：前向 all-gather 参数（$P$）+ 反向 all-gather 参数（$P$）+ 反向 reduce-scatter 梯度（$P$）= $3P$。

在这个简化口径下，$3P / 2P = 1.5$。换成本文账本采用的 **BF16 参数 + FP32 梯度** 后，必须按字节重算：DDP 的 FP32 梯度 all-reduce 约为 $2 \times 4P = 8P$ 字节；FSDP 两次 BF16 参数 all-gather 加一次 FP32 梯度 reduce-scatter，约为 $2P + 2P + 4P = 8P$ 字节。因此不能脱离 dtype 宣称 FSDP 网络字节固定增加 50%。实际墙钟差异还取决于 collective 次数与粒度、拓扑、预取以及计算通信重叠程度。


## 06 · ZeRO-3 语义在不同实现中的对应关系 { #equiv }

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

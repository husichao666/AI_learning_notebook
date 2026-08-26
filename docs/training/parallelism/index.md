---
title: 并行训练
description: 从训练状态与集合通信出发，系统学习 DP、ZeRO、FSDP、TP、SP、PP、CP、EP 与多维组合并行
type: series-hub
status: stable
level: beginner-to-advanced
updated: 2026-08-26
tags:
  - distributed-training
  - parallelism
---

# 并行训练：从单卡到多维组合

<div class="notebook-hero" markdown>

从显存账本和通信原语开始，逐步理解 DP、FSDP、TP、PP、CP、EP，以及它们如何组合成 nD 并行。

</div>

## 全书目录

1. [第 1 章 · 训练状态与集合通信](00-foundations.md)
   建立显存账本，理解集合通信语义、通信量口径与 ring 等底层算法。
2. [第 2 章 · Data Parallel](01-dp.md)
   从模型副本和数据切分出发，理解梯度同步与计算通信重叠。
3. **第 3 章 · 模型状态分片**
    - [3.1 · ZeRO 与 FSDP](02-fsdp.md)
    - [3.2 · Megatron 实现方案](02-megatron-fsdp.md)
    - [3.3 · PyTorch 原生方案](02-pytorch-fsdp.md)
    - [3.4 · HyperParallel 性能优化](02-hyper-fsdp.md)
4. **第 4 章 · Tensor Parallel 与 Sequence Parallel**
    - [4.1 · Tensor Parallel](03-tp.md)
    - [4.2 · Sequence Parallel](03-sp.md)
    - [4.3 · Loss Parallel](loss-parallel.md)
5. [第 5 章 · Pipeline Parallel](04-pp.md)
   沿模型深度切分计算，理解气泡、micro-batch 与 1F1B 调度。
6. [第 6 章 · Context Parallel](05-cp.md)
   沿序列维切分注意力，比较 Ring Attention、Ulysses 与 all-gather 路线。
7. **第 7 章 · Expert Parallel**
    - [7.1 · 原理](06-ep.md)
    - [7.2 · Megatron 专家并行实现](06-ep-source.md)
    - [7.3 · 专家并行性能优化](06-ep-performance.md)
    - [7.4 · MoE 辅助损失](moe-aux-loss.md)
8. [第 8 章 · 多维组合并行](07-nd.md)
   把各并行维度放入同一张设备网格，形成完整的选型与调优方法。

所有正文都以 Markdown 维护；左侧目录按“章—节”组织，右侧目录定位文章内部主题，顶部搜索可以跨整本书检索。

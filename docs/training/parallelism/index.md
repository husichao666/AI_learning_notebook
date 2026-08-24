---
title: 分布式训练并行策略
description: DP、FSDP、TP、PP、CP、EP 与 nD 并行学习路线
type: series-hub
status: stable
level: beginner-to-advanced
updated: 2026-08-24
tags:
  - distributed-training
  - parallelism
---

# 分布式训练并行策略

<div class="notebook-hero" markdown>

从显存账本和通信原语开始，逐步理解 DP、FSDP、TP、PP、CP、EP，以及它们如何组合成 nD 并行。

</div>

## 主线章节

1. [M0 · 基础铺垫：显存账本与通信原语](00-foundations.md)
2. [M1 · 数据并行 DP](01-dp.md)
3. [M2 · FSDP / ZeRO](02-fsdp.md)
4. [M3 · 张量并行 TP + SP](03-tp.md)
5. [M4 · 流水线并行 PP](04-pp.md)
6. [M5 · 上下文并行 CP](05-cp.md)
7. [M6-1 · EP 原理：路由、分发与负载均衡](06-ep.md)
8. [M6-2 · Megatron EP 源码执行链路](06-ep-source.md)
9. [M6-3 · EP 性能优化](06-ep-performance.md)
10. [M7 · nD 组合并行与工程调优](07-nd.md)

## 深入专题

- [Loss Parallel：词表并行交叉熵](loss-parallel.md)
- [MoE aux loss：负载均衡原理与实现](moe-aux-loss.md)

所有正文现在都以 Markdown 维护；左侧目录负责章节导航，右侧目录定位本章小节，顶部搜索可以跨整本手册检索。

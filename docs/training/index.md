---
title: 训练系统
description: 分布式训练、并行策略、通信与大规模训练系统
type: hub
status: growing
updated: 2026-08-25
---

# 训练系统

<div class="notebook-hero" markdown>

<span class="chapter-kicker">Knowledge Domain · Training Systems</span>

这里关注模型如何真正跑在多张 GPU 和多台机器上：显存如何切、通信如何发生、不同并行维度如何组合。

</div>

## 专题书

### [分布式训练并行策略](parallelism/)

从训练状态与集合通信开始，依次学习 DP、ZeRO/FSDP、TP/SP、PP、CP、EP，最后组合成多维并行。全书按 8 章组织，复杂主题在章内继续分节。

## 与性能工程的边界

本区回答“系统如何组织”；[性能工程](../engineering/)回答“如何采集证据、定位瓶颈并验证优化”。

---
title: TorchTitan 源码
description: 从训练入口、图编译与分布式并行实现理解 TorchTitan
type: hub
status: growing
updated: 2026-08-27
tags: [torchtitan, distributed-training, source-code]
---

# TorchTitan 源码

<div class="notebook-hero" markdown>

<span class="chapter-kicker">Framework Source · TorchTitan</span>

TorchTitan 把模型、并行策略、训练循环和 PyTorch 编译栈组织在同一个训练框架中。本专题不只罗列配置项，而是沿真实调用链追踪训练状态如何创建、通信如何进入执行路径，以及编译器如何改变训练程序的组织方式。

</div>

!!! note "源码基线"

    本专题以 TorchTitan `main` 分支为阅读对象。每篇文章会注明分析所对应的提交；`experiments` 目录中的接口和实现仍在快速变化，不应视为稳定 API。

## 正文章节

| 文章 | 核心问题 | 状态 |
| --- | --- | --- |
| [第 1 章 · DTensor 原理与使用](01-dtensor.md) | 逻辑全局张量、本地 buffer、DeviceMesh、Placement 与算子调度如何组成统一的分布式张量抽象 | 已整理 |
| [第 2 章 · DeviceMesh 与 Placement 详解](02-device-mesh-placement.md) | rank 如何组成命名 mesh、Placement 如何逐轴绑定，以及 ParallelDims 怎样派生训练所需的多套 mesh 视图 | 已整理 |

后续章节将继续补充 TorchTitan 的配置系统、模型构建、并行化、训练循环、检查点与性能工程等主题，章节号按知识依赖逐步展开。

## 独立源码分析

实验性功能和专题源码分析不占用正文章节号：

| 文章 | 核心问题 | 状态 |
| --- | --- | --- |
| [GraphTrainer：整步训练图如何捕获、变换与执行](graph-trainer.md) | 前向、loss、反向、SimpleFSDP collective 和 CUDA Graph 如何进入同一套图变换流水线 | 已整理 |

## 推荐阅读基础

前两章默认读者已经理解张量、反向传播和基本集合通信。若希望进一步对照分片参数的生命周期，可以补充阅读 [PyTorch 原生 FSDP2](../training/parallelism/02-pytorch-fsdp.md)。

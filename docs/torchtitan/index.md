---
title: TorchTitan 源码
description: 从训练入口、图编译与分布式并行实现理解 TorchTitan
type: hub
status: growing
updated: 2026-09-01
tags: [torchtitan, distributed-training, source-code]
---

# TorchTitan 源码

<div class="notebook-hero" markdown>

<span class="chapter-kicker">Framework Source · TorchTitan</span>

TorchTitan 把模型、并行策略、训练循环和 PyTorch 编译栈组织在同一个训练框架中。本专题不只罗列配置项，而是沿真实调用链追踪训练状态如何创建、通信如何进入执行路径，以及编译器如何改变训练程序的组织方式。

</div>

!!! note "源码基线"

    本专题以 TorchTitan `main` 分支为阅读对象。每篇文章会注明分析所对应的提交；`experiments` 目录中的接口和实现仍在快速变化，不应视为稳定 API。

## 阅读路线

TorchTitan 目前同时保留 `partial_dtensor` 和 `spmd_types` 两种 SPMD 后端。它们读取同一套模型布局配置，却采用不同的运行时表示：前者让 DTensor 在算子间携带 Placement，后者让本地 Tensor 执行计算，再用 SPMD 类型和显式 collective 描述分布语义。因此，本专题不会把两套机制揉在同一条源码调用链里。

前两章先建立两条路线都要用到的概念基础。这里的“共用”是指逻辑全局张量、mesh axis 和布局语义共用，并不表示两种后端都把每个激活包装成 DTensor。

### 共用基础

| 文章 | 核心问题 | 状态 |
| --- | --- | --- |
| [第 1 章 · DTensor 原理与使用](01-dtensor.md) | 逻辑全局张量、本地 buffer、DeviceMesh、Placement 与算子调度如何组成统一的分布式张量抽象 | 已整理 |
| [第 2 章 · DeviceMesh 与 Placement 详解](02-device-mesh-placement.md) | rank 如何组成命名 mesh、Placement 如何逐轴绑定，以及 ParallelDims 怎样派生训练所需的多套 mesh 视图 | 已整理 |

### `partial_dtensor` 路线

这条路线让模型并行参数和激活以 DTensor 进入算子，依靠 DTensor dispatcher、分片传播和 `redistribute()` 完成布局推导与转换。

| 文章 | 核心问题 | 状态 |
| --- | --- | --- |
| [第 3 章 · 分布式算子与分片传播](03-distributed-operators.md) | 算子怎样推导输出布局、枚举矩阵乘策略，并根据重排布成本选择 collective | 已整理 |
| [第 4 章 · 使用 ColwiseParallel 切分模型](04-colwise-parallel.md) | ColwiseParallel 怎样分发模块参数，以及 TorchTitan 的 `partial_dtensor` 后端怎样执行同一套布局配置 | 已整理 |

### `spmd_types` 路线

这条路线从普通本地 Tensor 出发，用 SPMD 类型描述每条 mesh axis 上的布局，并把布局转换写成显式 collective。

| 文章 | 核心问题 | 状态 |
| --- | --- | --- |
| [第 5 章 · ShardingConfig 与 spmd_types 后端](05-sharding-config-spmd-types.md) | 同一份布局配置怎样变成本地状态分片、SPMD 类型、forward 包装和显式 collective | 已整理 |
| [第 6 章 · torch.compile 与显式通信](06-torch-compile-explicit-collectives.md) | 并行化后的 TransformerBlock 怎样生成前反向编译图，以及显式 collective 怎样进入图级优化 | 已整理 |

第 6 章沿当前默认的 `spmd_types` 路线继续向下追踪，但关注点已经从类型检查转向真正执行的计算图：编译器看到的是本地算子和显式 collective，而不是检查元数据。

### `GraphTrainer` 路线

普通 `Trainer` 通过 `torch.compile` 为 TransformerBlock 建立局部编译区域；实验性的 `GraphTrainer` 则显式构造包含前向、loss 和反向的联合 FX 图，再由 TorchTitan 自己组织后续 graph pass。这条路线会复用前面的 DTensor 和编译基础，但当前模型配置使用 `partial_dtensor` 与 SimpleFSDP，不沿用第 5 章的 `spmd_types` 类型检查路径。

| 文章 | 核心问题 | 状态 |
| --- | --- | --- |
| [第 7 章 · GraphTrainer 的整步训练图](07-graph-trainer-step-graph.md) | GraphTrainer 怎样把模型状态、前向、loss 和参数反向变成一张联合 FX 图，并让图输出的梯度回到基础训练循环 | 已整理 |
| [第 8 章 · GraphTrainer 的图变换流水线](08-graph-trainer-passes.md) | 联合图怎样依次完成结构规范化、激活生命周期管理、分布式通信调度、Inductor 编译与 CUDA Graph 包装 | 已整理 |

下一章将继续解释 SimpleFSDP 的参数与通信生命周期。两条 Trainer 路线都讲清后，再进入训练循环、检查点与性能工程等共用主题。

## 推荐阅读基础

本专题默认读者已经理解张量、反向传播和基本集合通信。若希望进一步对照分片参数的生命周期，可以补充阅读 [PyTorch 原生 FSDP2](../training/parallelism/02-pytorch-fsdp.md)。

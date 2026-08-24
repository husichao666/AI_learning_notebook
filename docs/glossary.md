---
title: 术语表
description: 模型架构与分布式训练中的常用缩写和概念
type: reference
status: growing
updated: 2026-08-24
---

# 术语表

| 术语 | 含义 |
| --- | --- |
| DP | Data Parallel，复制模型、切分数据并同步梯度 |
| FSDP | Fully Sharded Data Parallel，切分参数、梯度和优化器状态 |
| TP | Tensor Parallel，切分单层内部的权重与计算 |
| PP | Pipeline Parallel，沿模型深度切分 stage |
| CP | Context Parallel，沿上下文序列维切分激活与注意力计算 |
| EP | Expert Parallel，把 MoE 专家放到不同设备 |
| SP | Sequence Parallel，通常与 TP 配合切分非线性区域激活 |
| MFU | Model FLOPs Utilization，模型有效计算量占硬件峰值的比例 |
| MHA | Multi-Head Attention，多头注意力 |
| RoPE | Rotary Position Embedding，旋转位置编码 |
| MoE | Mixture of Experts，每个 token 稀疏激活少量专家 |

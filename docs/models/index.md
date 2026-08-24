---
title: 模型架构
description: 从完整模型视角理解架构选择、核心模块和工程权衡
type: hub
status: growing
updated: 2026-08-24
---

# 模型架构

<div class="notebook-hero" markdown>

<span class="chapter-kicker">Knowledge Domain · Models</span>

这里关注“一个完整模型为什么这样设计”：先建立整体结构，再把注意力、位置编码、MoE、记忆和多模态模块连接起来。

</div>

## 模型分析

| 模型 | 关注重点 | 状态 |
| --- | --- | --- |
| [Qwen3.5](qwen3-5/) | 原生多模态、Gated DeltaNet、M-RoPE、Gated Attention | 持续补充 |
| [DeepSeek V4](deepseek-v4/) | 混合注意力、Engram、MoE 2.0、训练工程 | 持续补充 |

## 推荐阅读方式

先看模型整体结构，再沿文中的链接进入[核心模块](../components/)补齐局部原理；涉及并行训练和性能问题时，转到[训练系统](../training/)与[性能工程](../engineering/)。

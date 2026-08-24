---
title: 核心模块
description: 注意力、位置编码、外部记忆等可复用模型组件
type: hub
status: growing
updated: 2026-08-24
---

# 核心模块

<div class="notebook-hero" markdown>

<span class="chapter-kicker">Knowledge Domain · Components</span>

这里把模型拆成可复用的构件，强调数学原理、张量形状、源码路径和它在完整模型中的作用。

</div>

## 模块地图

- [注意力机制](attention/)：DSA、Gated Attention，以及后续的 MLA、线性注意力和长上下文方案。
- [位置编码](position-encoding/)：从标准 RoPE 到多模态 M-RoPE。
- [可扩展记忆](memory/)：Engram 的查表记忆和存算分离思路。

!!! tip "新增模块放在哪里"

    优先按“解决的问题”归类，而不是按论文或模型名归类。模型名只作为标签，避免同一个模块在多个模型目录里重复维护。

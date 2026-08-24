---
title: 注意力机制
description: 从稠密注意力到门控与稀疏注意力的模块地图
type: hub
status: growing
updated: 2026-08-24
---

# 注意力机制

## 当前内容

- [DeepSeek Sparse Attention](dsa/)：用 Lightning Indexer 选择 Top-k token，再做细粒度稀疏注意力。
- [Gated Attention](gated-attention/)：在 SDPA 输出中引入门控，分析非线性、稀疏性和 Attention Sink。

## 建议学习顺序

先掌握标准 MHA 的 $QK^\top$、softmax 与 $V$ 聚合，再读 Gated Attention 理解“如何调制注意力输出”，最后读 DSA 理解“如何减少需要参与注意力的 token”。

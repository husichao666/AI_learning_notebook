---
title: 从这里开始
description: AI Learning Notebook 的知识地图、学习路径和写作约定
type: guide
status: stable
updated: 2026-08-24
---

# 从这里开始

这个知识库按**知识领域**组织，而不是按文件格式、论文来源或更新时间组织。全站是一座书架，每个专题系列是一册可以顺序阅读的书，单篇模块笔记则可以独立查阅。

## 三条现有学习路径

### 分布式训练

[显存与通信基础](training/parallelism/00-foundations.md) → [DP](training/parallelism/01-dp.md) → [FSDP](training/parallelism/02-fsdp.md) → [TP](training/parallelism/03-tp.md) → [PP](training/parallelism/04-pp.md) → [CP](training/parallelism/05-cp.md) → [EP](training/parallelism/06-ep.md) → [nD 并行](training/parallelism/07-nd.md)

### Attention 演进

标准 MHA 基础 → [RoPE / M-RoPE](components/position-encoding/rope/) → [Gated Attention](components/attention/gated-attention/) → [DeepSeek Sparse Attention](components/attention/dsa/)

### 模型拆解

[Qwen3.5](models/qwen3-5/) 或 [DeepSeek V4](models/deepseek-v4/) → 跟随文内链接下钻核心模块 → 回到[训练系统](training/)理解分布式落地 → 用[性能工程](engineering/)验证实现。

## 内容状态

| 状态 | 含义 |
| --- | --- |
| `draft` | 结构和结论仍可能大幅变化 |
| `growing` | 主体已经可读，仍会持续补充 |
| `stable` | 结构基本稳定，以勘误和小幅更新为主 |

## 新增内容约定

1. 先确定它属于模型、模块、训练系统还是性能工程。
2. 一个独立主题使用一个目录，正文为 `index.md`，图片和代码与主题就近放置。
3. 有强顺序关系的系列才使用 `00-`、`01-` 编号。
4. 每篇文章填写 `title`、`description`、`type`、`status`、`updated` 和 `tags`。
5. 完成修改后运行严格构建，确保公式、图片和站内链接都有效。

---
title: AI Learning Notebook
description: 大模型架构、核心模块、训练系统与性能工程个人知识库
hide:
  - toc
---

<section class="kb-hero">
  <p class="kb-eyebrow">PERSONAL KNOWLEDGE BASE · MODELS TO SYSTEMS</p>
  <h1>AI Learning Notebook</h1>
  <p class="kb-lead">从完整模型、核心模块到分布式训练与性能工程，把零散阅读沉淀成可以持续生长的知识体系。</p>
  <div class="kb-actions">
    <a class="kb-button kb-button-primary" href="start-here/">浏览学习路径</a>
    <a class="kb-button" href="training/parallelism/">继续并行训练</a>
  </div>
  <p class="kb-search-hint">提示：按 <kbd>/</kbd> 可以随时搜索整座知识库</p>
</section>

## 知识地图

<div class="kb-domain-grid">
  <a class="kb-domain-card kb-models" href="models/">
    <span class="kb-card-code">MODEL</span>
    <h3>模型架构</h3>
    <p>从完整模型视角理解架构选择，以及各模块如何协同。</p>
    <span class="kb-card-meta">Qwen3.5 · DeepSeek V4</span>
  </a>
  <a class="kb-domain-card kb-components" href="components/">
    <span class="kb-card-code">MODULE</span>
    <h3>核心模块</h3>
    <p>下钻注意力、位置编码、外部记忆等可复用构件。</p>
    <span class="kb-card-meta">DSA · RoPE · Gated Attention · Engram</span>
  </a>
  <a class="kb-domain-card kb-training" href="training/">
    <span class="kb-card-code">TRAIN</span>
    <h3>训练系统</h3>
    <p>理解显存、通信、并行策略与大规模训练系统。</p>
    <span class="kb-card-meta">DP · FSDP · TP · PP · CP · EP</span>
  </a>
  <a class="kb-domain-card kb-engineering" href="engineering/">
    <span class="kb-card-code">PERF</span>
    <h3>性能工程</h3>
    <p>用 profiling 和可验证的指标定位、解释并解决瓶颈。</p>
    <span class="kb-card-meta">采集 · 分析 · 优化 · 验收</span>
  </a>
</div>

## 推荐路径

<div class="kb-path-grid">
  <div class="kb-path-card">
    <span class="kb-status kb-status-stable">完整专题</span>
    <h3>分布式训练</h3>
    <p>显存与通信 → DP → FSDP → TP → PP → CP → EP → nD</p>
    <a href="training/parallelism/00-foundations/">从 M0 开始 →</a>
  </div>
  <div class="kb-path-card">
    <span class="kb-status kb-status-growing">持续补充</span>
    <h3>Attention 演进</h3>
    <p>RoPE / M-RoPE → Gated Attention → DeepSeek Sparse Attention</p>
    <a href="components/attention/">进入模块地图 →</a>
  </div>
  <div class="kb-path-card">
    <span class="kb-status kb-status-growing">持续补充</span>
    <h3>模型拆解</h3>
    <p>先看整体架构，再沿链接下钻模块、训练和性能实现。</p>
    <a href="models/">选择一个模型 →</a>
  </div>
</div>

## 最近整理

<div class="kb-updates">
  <div><time>2026-08-24</time><a href="training/parallelism/">并行训练 M0–M7 完成 Markdown 书籍化迁移</a></div>
  <div><time>2026-08-24</time><a href="components/attention/dsa/">DeepSeek Sparse Attention：闪电索引与稀疏注意力</a></div>
  <div><time>2026-08-24</time><a href="models/qwen3-5/">Qwen3.5 模型架构与关键模块</a></div>
</div>

<p class="kb-footer-links"><a href="start-here/">阅读与写作约定</a> · <a href="glossary/">术语表</a> · <a href="https://github.com/husichao666/AI_learning_notebook">GitHub 仓库</a></p>

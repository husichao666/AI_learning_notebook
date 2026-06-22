# 分布式训练并行策略 · 学习手册

> 一份**由浅入深、面向初学者**的并行训练学习手册（DP · FSDP · TP · PP · CP · EP）。
> 每个模块都按同一条主线展开：**为什么需要它 → 它省了什么 / 花了什么 → 框架里怎么落地**。
> 看不懂公式没关系，先抓住「动机」和「省显存 / 花通信」的权衡，细节可以后补。

## 📖 在线阅读（整本手册）

> 推荐入口：**`index.html` 书壳**——左侧目录在 M0–M7 间切换，每章为独立 HTML（公式、代码高亮、SVG 图齐全）。
> GitHub 不直接渲染 HTML，请用下面链接打开渲染版（书壳含 iframe，建议走 GitHub Pages）：

- **整本书（GitHub Pages，需先开启）** 👉 <https://husichao666.github.io/AI_learning_notebook/infra/parallize/>
- **单章即点即看（htmlpreview）** 👉 [词表并行交叉熵 Loss Parallel](https://htmlpreview.github.io/?https://github.com/husichao666/AI_learning_notebook/blob/main/infra/parallize/loss_parallel/index.html)

## 🧭 一句话主线

> 单卡装不下、算得慢 → 先**复制模型切数据**（DP）→ 嫌冗余就**切状态**（FSDP/ZeRO）
> → 单层太大就**切层内**（TP）→ 层数太多就**切层间**（PP）→ 序列太长就**切序列**（CP）
> → MoE 就**切专家**（EP）→ 最后把它们**组合成 nD 并行**。

## 📈 学习路线（由浅入深）

```
        ┌─────────────────────────────────────────────────────────┐
        │  Module 0  基础铺垫：显存账本 + 通信原语（必看）          │  ★☆☆☆☆
        └───────────────────────────────┬─────────────────────────┘
                                         ▼
   ┌───────────────┐   ┌────────────────────────┐
   │ M1  数据并行 DP │──▶│ M2  FSDP / ZeRO（切状态）│                    ★★☆☆☆
   └───────────────┘   └────────────┬───────────┘
                                    ▼
                          ┌──────────────────────┐
                          │ M3  张量并行 TP (+SP) │  ←《loss_parallel》在这里  ★★★☆☆
                          └──────────┬───────────┘
                                     ▼
   ┌─────────────────┐   ┌────────────────────┐   ┌─────────────────────┐
   │ M5 上下文并行 CP │◀──│ M4  流水线并行 PP   │──▶│ M6 专家并行 EP (MoE) │     ★★★★☆
   └─────────────────┘   └─────────┬──────────┘   └─────────────────────┘
                                   ▼
                     ┌──────────────────────────────┐
                     │ M7  nD 组合并行 + 工程调优     │                       ★★★★★
                     └──────────────────────────────┘
```

> 推荐顺序：**M0 → M1 → M2 → M3 → M4 → M5 / M6（可互换）→ M7**。
> 急用某个技术也可直接跳读，但 M0 的「显存账本」和「通信原语」是后面所有模块的地基。

---

## Module 0 · 基础铺垫 ★☆☆☆☆

**核心问题：训练一个模型，显存到底花在哪、卡之间怎么传数据？**

- **显存账本**：一次训练 step 的显存 = 参数 weights + 梯度 gradients + 优化器状态 optimizer states + 激活值 activations
  - 混合精度 + Adam 下，**每个参数约 16 字节**（fp16 权重 2 + fp16 梯度 2 + fp32 master 权重 4 + Adam 一阶 m 4 + 二阶 v 4）
  - 激活值单独算，且随 **batch × seq × 层数** 增长——长序列训练的显存大头
- **通信原语（collective ops）**：理解后面一切并行的语言
  - `all-reduce`、`all-gather`、`reduce-scatter`、`broadcast`、`all-to-all`、点对点 `send/recv`
  - 关键恒等式：**all-reduce = reduce-scatter + all-gather**
- **硬件拓扑**：单机内 NVLink/NVSwitch（快）vs 跨机 InfiniBand/RoCE（慢）——决定哪种并行放在哪一层
- **衡量指标**：吞吐 throughput、MFU、流水线气泡 bubble、**通信与计算重叠 overlap**

> ✅ **学完自测**：Adam 训练 7B 模型，光优化器状态要多少显存？all-reduce 为什么可以拆成两步？

---

## Module 1 · 数据并行 DP（Data Parallel）★★☆☆☆

**核心问题：模型放得下，但想用多卡把训练**加速**。**

- 思想：**每张卡复制一份完整模型**，把一个 batch 切成多份，各算各的梯度，再 `all-reduce` 求平均
- PyTorch **DDP**：梯度分桶 bucket、反向传播时**边算边通信**（overlap）
- 概念：global batch size、梯度累积 gradient accumulation
- **局限**：模型 / 梯度 / 优化器状态在每张卡上都存了**完整一份**（冗余）——只加速，不省显存

> ✅ **学完自测**：DP 为什么省不了显存？8 卡 DP 的 global batch 和单卡相比是多少？

---

## Module 2 · FSDP / ZeRO（切状态的数据并行）★★☆☆☆

**核心问题：DP 里每张卡冗余存了 N 份模型，太浪费——能不能把状态切开？**

- **ZeRO 三阶段**（按省得越来越多排序）：
  - ZeRO-1：切分**优化器状态**
  - ZeRO-2：再切分**梯度**
  - ZeRO-3：再切分**参数** ＝ **FSDP**
- **FSDP 工作流**（核心理解点）：
  - 前向：用到某层时 `all-gather` 还原它的参数 → 算完 → **立刻丢弃**（reshard）
  - 反向：再 `all-gather` 参数 → 算梯度 → `reduce-scatter` 把梯度切回各卡
  - 关键词：unshard / reshard、通信-计算 overlap、prefetch 预取
- **FSDP1 vs FSDP2**：FlatParameter 整体切 vs 逐参数切（per-parameter，基于 DTensor，torchtitan 用的就是它）
- 进阶：CPU / NVMe offload
- **权衡**：通信量约为 DP 的 1.5×，换来显存大幅下降

> ✅ **学完自测**：ZeRO-3 和 FSDP 是什么关系？FSDP 比 DP 多了哪一次通信？

---

## Module 3 · 张量并行 TP（Tensor Parallel）+ 序列并行 SP ★★★☆☆

**核心问题：单**一层**就大到放不下（巨大 MLP / 注意力 / 词表），FSDP 也救不了——把一层切到多卡。**

- **Megatron 式 TP**（按矩阵乘法切）：
  - MLP：先**列切**（column-parallel）再**行切**（row-parallel），一次 `all-reduce` 合并
  - Attention：按 **head** 切 QKV（列切）+ 输出投影行切
  - Embedding / LM Head：**词表并行** + **词表并行交叉熵** 👉 **本系列已发布：[loss_parallel](./loss_parallel/)**
- **序列并行 SP**：TP 的搭档，把 LayerNorm/Dropout 区域的激活按 **序列维** 切，进一步省激活显存；用 `all-gather` / `reduce-scatter` 替换 `all-reduce`
- **局限**：**每一层都要通信**，通信极密集——通常只在单机内 NVLink 范围使用（TP ≤ 8）

> ✅ **学完自测**：为什么 TP 一般不跨机？词表并行交叉熵省掉了哪个巨型张量的 all-gather？（→ 读 loss_parallel）

---

## Module 4 · 流水线并行 PP（Pipeline Parallel）★★★★☆

**核心问题：层数非常多，把**不同的层**放到不同的卡上（不同 stage）。**

- 朴素 PP 的痛点：**气泡 bubble**（卡在等上一段算完，大量空闲）
- 调度演进（一条主线）：
  - **GPipe**：拆 micro-batch 填充流水线
  - **1F1B**（PipeDream）：一前一后，降显存
  - **Interleaved 1F1B**（虚拟 stage）：进一步减小气泡
  - **Zero-Bubble PP**：把反向拆成两半，几乎消除气泡
- 概念：micro-batch、warmup / steady / cooldown 三阶段、点对点 `send/recv`
- **难点**：stage 怎么切才负载均衡、和 loss / embedding 的配合

> ✅ **学完自测**：气泡是怎么产生的？1F1B 相比 GPipe 主要改善了什么？

---

## Module 5 · 上下文并行 CP（Context Parallel）★★★★☆

**核心问题：序列**超长**（几十万 token），注意力的激活单卡放不下——把**序列维**切到多卡。**

- 把 sequence 切到多卡，但**注意力需要全局的 K/V** → 这是难点
- 两条主流路线：
  - **Ring Attention**：在线 softmax（online softmax）+ 环形传递 K/V 分块
  - **DeepSpeed Ulysses**：用 `all-to-all` 在 head 维和 seq 维之间转置
- 负载均衡：causal mask 下的 zigzag / striped 分片
- 辨析：Megatron 的 **SP**（配合 TP）vs 独立的 **CP**——都切序列，但目的与位置不同

> ✅ **学完自测**：为什么序列切开后注意力不能各算各的？online softmax 解决了什么？

---

## Module 6 · 专家并行 EP（Expert Parallel · MoE）★★★★☆

**核心问题：MoE 模型专家很多，把**不同专家**放到不同卡。**

- MoE 简介：router / gating、top-k 路由、稀疏激活
- **EP**：不同 expert 放不同卡，靠 `all-to-all` 做 token 的 dispatch / combine
- 与 TP/DP 组合：EP×TP、容量因子 capacity factor、负载均衡 loss
- 工程难点：token 分布不均导致的通信倾斜

> ✅ **学完自测**：MoE 的 all-to-all 在传什么？专家负载不均会怎样？

---

## Module 7 · nD 组合并行 + 工程调优 ★★★★★

**核心问题：真实大模型训练是把上面**全部组合**起来用的。**

- **3D / 4D / 5D 并行**：DP × FSDP × TP × PP × CP × EP 自由组合
- **Device Mesh**：用一个多维网格描述 GPU 的逻辑布局（torchtitan 的 `DeviceMesh` / Megatron 的 `parallel_state`）
- **放置原则**：通信最密的（TP）放最内层 NVLink，PP 跨节点，DP/FSDP 放最外层
- **选型决策树**：放得下就 FSDP/DP → 单层太大加 TP → 层太多加 PP → 序列太长加 CP → MoE 加 EP
- 通用优化：激活重计算 activation checkpointing、bf16 / fp8 混合精度、通信-计算 overlap、profiling 定位瓶颈
- **框架对比**：Megatron-LM / Megatron-Core、DeepSpeed、**torchtitan**（PyTorch 原生 DTensor）

> ✅ **学完自测**：给定「模型 70B、序列 128k、512 卡」，你会怎么排布各并行维度？

---

## 📚 本手册章节进度

| 模块 | 主题 | 目录 | 状态 |
|---|---|---|---|
| M0 | 基础铺垫：显存账本 + 通信原语 | [`00-foundations/`](./00-foundations/) | ✅ 已发布 |
| M1 | 数据并行 DP | [`01-dp/`](./01-dp/) | ✅ 已发布 |
| M2 | FSDP / ZeRO | [`02-fsdp/`](./02-fsdp/) | ✅ 已发布 |
| M3 | 张量并行 TP + SP | [`03-tp/`](./03-tp/) | ✅ 已发布 |
| M3 深入 | 词表并行交叉熵 Loss Parallel | [`loss_parallel/`](./loss_parallel/) | ✅ 已发布 |
| M4 | 流水线并行 PP | [`04-pp/`](./04-pp/) | ✅ 已发布 |
| M5 | 上下文并行 CP | [`05-cp/`](./05-cp/) | ✅ 已发布 |
| M6 | 专家并行 EP | [`06-ep/`](./06-ep/) | ✅ 已发布 |
| M7 | nD 组合并行 + 工程 | [`07-nd/`](./07-nd/) | ✅ 已发布 |

> 每章为 `infra/parallize/<topic>/index.html` 下的独立 HTML，风格统一：
> 动机 → 数学/原理 → 通信&显存分析 → 框架源码（Megatron-LM / torchtitan）逐行对照 → 内联 SVG 示意图。
> 所有代码片段均引自两个框架的真实源码并标注 `文件:行号`。

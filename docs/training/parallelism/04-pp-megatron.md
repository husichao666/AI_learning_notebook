---
title: "5.2 · Megatron 性能优化"
description: "围绕 Megatron 的 PP 调度，解释 VPP 粒度、P2P 与 MoE 通信重叠、输出投影梯度延迟、激活管理和参数预取。"
type: engineering-note
status: stable
level: advanced
updated: 2026-09-15
tags: [distributed-training, pipeline-parallel, megatron, performance]
---

# Megatron PP：调度与性能优化 { #megatron-optimization }

<div class="notebook-hero" markdown>

<span class="chapter-kicker">5.2 · Megatron 性能优化</span>

流水线并行（Pipeline Parallel, PP）的任务顺序决定了设备何时有计算可做。Megatron 在一前向一反向（1F1B）与虚拟流水线并行（VPP）的交错调度上，进一步调整 chunk 粒度、异步收发和计算顺序。本节说明每项优化利用哪段独立计算、改变哪些状态的生命周期，以及相应的显存代价与实现约束。

**本节关键词：** VPP 负载均衡 · P2P 重叠 · MoE 前后向配对 · 权重梯度延迟 · 激活管理

</div>

层布局、micro-batch 与各类 PP 调度的基础见 [5.1 · 原理与调度](04-pp.md)。这里的 chunk 指每张卡持有的一段模型，对应一个虚拟 stage；P2P 指相邻 stage 之间的点对点通信。

!!! note "实现范围"

    本节按 2026-09-14 核对的 NVIDIA/Megatron-LM `main` 说明常规 GPT 训练路径。表中的下划线名称是 Megatron Core 配置字段；命令行参数单独标出。具体默认值和组合限制可能随版本变化。

## 01 · Megatron 主线中的调度入口 { #schedule-entry }

普通单模型训练的 `get_forward_backward_func()` 主要按 PP 规模及是否启用虚拟 stage 选择路径：

| 条件 | 调度函数 |
| --- | --- |
| PP 规模为 1 | `forward_backward_no_pipelining` |
| PP 规模大于 1，未配置虚拟 stage | `forward_backward_pipelining_without_interleaving` |
| PP 规模大于 1，配置了虚拟 stage | `forward_backward_pipelining_with_interleaving` |

后两项分别对应非交错与交错 1F1B。GPipe 是理解流水线的重要基础，但这一入口没有将它列为独立训练调度；Zero Bubble 的作者实现使用 Megatron 的独立分支，DualPipe / DualPipeV 则有单独的官方仓库。这里区分的是实现入口，不代表这些算法在其他框架中不可用。[Megatron 调度源码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)

## 02 · VPP 粒度与 stage 负载均衡 { #vpp-tuning }

先确定设备划分与任务供给，再决定有多少通信可以重叠：

- `--pipeline-model-parallel-size` 设置物理 PP 规模 $P$。
- `--num-layers-per-virtual-pipeline-stage` 设置每个虚拟 stage 的层数，是启用交错调度的一种配置方式；chunk 数应结合模型总层数、PP 规模及首尾特殊层划分核对。
- `microbatch_group_size_per_vp_stage` 控制同一虚拟 stage 的 micro-batch 分组大小，影响前后向任务的 chunk 切换顺序；它不改变每个 micro-batch 的样本数。

在固定全局 batch、没有 batch ramp-up、且可整除的常规设置下：

$$
M=\frac{B_{\mathrm{global}}}{D\,B_{\mathrm{micro}}}
$$

其中 $M$ 是每条流水线一次更新处理的 micro-batch 数，$B_{\mathrm{global}}$ 是一次优化器更新覆盖的总样本数，$D$ 是数据并行副本数，$B_{\mathrm{micro}}$ 是每个副本一次 micro-batch 的样本数。PP rank 共同处理同一批样本，因此分母中不再乘 $P$。当前交错调度要求分组大小在 $P$ 与 $M$ 之间，最后一组若不满，也至少应有 $P$ 个 micro-batch。[分组与合法性检查](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)

更小的 chunk 可以缩短调度气泡，但会增加 P2P 次数，并缩短每次可用于覆盖通信的计算窗口。固定全局 batch 时，减小 $B_{\mathrm{micro}}$ 虽然会增加 $M$，也可能让矩阵乘法变小、效率下降。因此 chunk 数和 micro-batch 数都不能只按“越多越好”设置。

此外，首部的 embedding、尾部的词表输出投影与 loss，以及不同层的 MoE 开销，会使“等层数”不等于“等耗时”。Megatron 支持用 `num_layers_in_first_pipeline_stage` / `num_layers_in_last_pipeline_stage` 调整首尾层数，或通过 `pipeline_model_parallel_layout` 指定更细的布局。应依据各 stage 的实测时间移动层；这两种布局配置在当前实现中不能同时指定。[层划分配置](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py)

## 03 · P2P 重叠：提前收发，使用前再等待 { #p2p-overlap }

观察 [四卡 VPP 示例](04-pp.md#advanced) **稳态阶段的 GPU1**。它的两个 chunk 都按以下方向收发：

- **激活：GPU0 → GPU1 → GPU2**，GPU1 收到输入后做前向，再发送输出。
- **梯度：GPU2 → GPU1 → GPU0**，GPU1 收到输出梯度后做反向，再发送自己算出的输入梯度 DX。

以下数字是 micro-batch 编号；前向处理 chunk 0，反向处理 chunk 1。

1. **B1 结束 → F6**：向 GPU0 发 DX1，同时从 GPU2 预接收 B2 的梯度；随后计算输入已收齐的 F6，利用这段时间传输梯度。
2. **F6 结束 → B2**：向 GPU2 发激活6，同时从 GPU0 预接收激活7；确认第 1 步的梯度收齐后计算 B2，利用反向计算传输激活。
3. **B2 结束 → F7**：向 GPU0 发 DX2，同时从 GPU2 预接收 B3 的梯度；确认激活7收齐后计算 F7，继续覆盖梯度传输。

![GPU1 的 VPP 稳态收发：标明通信对端、梯度预接收和数据使用前的检查点](assets/04-pp-p2p-overlap.svg){ style="max-width: 780px;" }

*所有行都属于 GPU1；图从 B1 结束后开始，F6 的输入此前已收齐。虚线处检查接收是否完成，块宽仅为示意。*

**B2 的梯度接收在 F6 开始前就已发起，因此可能被 F6 的计算覆盖；若 GPU2 尚未算完或传输未结束，B2 仍须等待。** Megatron 通过异步收发句柄，将等待推迟到数据使用前；发送 buffer 则必须保留到发送完成。[收发实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/p2p_communication.py)、[调度与等待位置](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)

配置上，当前常规 PP 路径需启用 VPP，并设置 `overlap_p2p_comm=True`、`batch_p2p_comm=False`；`overlap_p2p_comm_warmup_flush=True` 可将重叠扩展到预热和排空。实际可覆盖的时间仍取决于独立计算和 buffer 的释放时机。[配置定义](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/model_parallel_config.py)

### 预热与排空：在本次计算前预接收下一份数据

`overlap_p2p_comm_warmup_flush` 在预热时用前向覆盖前向通信，在排空时用反向覆盖反向通信。仍以 GPU1 为例，当前任务的输入已收齐后：

- **预热**：先从 GPU0 预接收下一次前向的输入 → 计算当前前向，同时接收 → 向 GPU2 发送当前输出，确认下一份输入收齐后继续前向。
- **排空**：先从 GPU2 预接收下一次反向的梯度 → 计算当前反向，同时接收 → 向 GPU0 发送当前 DX，确认下一份梯度收齐后继续反向。

![GPU1 在预热时从 GPU0 预接收下一份激活，在排空时从 GPU2 预接收下一份梯度](assets/04-pp-p2p-warmup-flush.svg){ style="max-width: 780px;" }

关键是把**下一次接收提前到本次计算之前发起**。计算结束时若数据还没收齐，仍需等待。[预热与排空的预接收实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)

## 04 · MoE 重叠：把一对前后向拆到层内调度 { #moe-overlap }

`--overlap-moe-expert-parallel-comm` 处理的是 **混合专家模型（MoE）层内的专家并行（EP）All-to-All（A2A，全互换通信）**。它保留外层 1F1B / VPP 框架，把一个 micro-batch 的前向与另一个 micro-batch 的反向交给 `combined_1f1b` 共同执行，再由层内 schedule plan 安排通信和计算。这条路径与 DualPipe 有相近的重叠思路，但不会把模型布局自动改成双副本或 V 形。

一层 MoE 被拆为：**attention 与路由等前置计算 → dispatch → 专家 MLP → combine**。反向按依赖走相反方向，其中 `combine backward` 先把输出梯度送回专家，`dispatch backward` 再把专家算出的输入梯度送回 token 来源。前后向属于不同 micro-batch，因此可以使用以下窗口：

| 已发起的通信 | 可同时执行的独立计算 |
| --- | --- |
| 反向的 combine A2A | 前向的 attention 与路由等前置计算 |
| 前向的 dispatch A2A | 反向的专家计算，包括可调度的权重梯度 |
| 反向的 dispatch A2A | 前向的专家 MLP，前提是它的 dispatch 已完成 |
| 前向的 combine A2A | 反向的 attention 等前置部分 |

![Megatron combined 1F1B 在同一卡上交错两个 micro-batch 的 MoE 通信与计算](assets/04-pp-moe-overlap.svg){ style="max-width: 780px;" }

*图中的 F、B 属于不同 micro-batch，各列是可重叠的局部窗口，宽度不表示实测耗时。每一路仍须满足自身的 dispatch、专家计算、combine 依赖；不能用尚未收到 token 的专家计算覆盖自己的 dispatch。*

源码将这些操作组织成可分别执行、带依赖事件的 schedule node，而不是完整跑完一个 `forward()` 后再调用完整 `backward()`。交错 PP 开启这项优化时还会多预热一个前向任务，为稳态中的独立配对准备输入。[`combined_1f1b.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/combined_1f1b.py)、[层内执行顺序](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/model_chunk_schedule_plan.py)

配套的 `--delay-wgrad-compute` 将线性层的 DX 与参数梯度（DW）分开：先让 DX 推进反向依赖，再把 DW 放入后续通信窗口。这里的延迟服务于层内通信重叠，并不意味着外层调度已经变成[5.1 的 Zero Bubble 调度](04-pp.md#zero-bubble)。DW 完成前，其所需的输入和输出梯度仍要保留；与 DP 梯度归约结合时，也必须等 DW 实际完成后才能把对应梯度标记为 ready。

当前实现需要注意以下组合条件：

- EP 大于 1，dispatcher 为 `alltoall` 或 `flex`；PP 大于 1 时需配置 VPP。这项 EP 优化也有 PP 为 1 的执行路径。
- 基础模型使用 BF16 或 FP16，要求 PyTorch ≥ 2.6；低精度算子和通信后端还有各自的支持条件。
- 不能与利用同层共享专家分支的 `moe_shared_expert_overlap` 同开；不支持 full recompute 或把整个 `moe` 放入重算模块列表，不能因此推断所有选择性重算都不支持。
- `delay_wgrad_compute` 需要 Transformer Engine（TE）；与 `overlap_grad_reduce` 同开时，当前 CLI 要求 TE ≥ 2.8。若再叠加 CUDA Graph，应继续核对具体捕获范围的版本限制。

同时在途的前后向、额外预热和延迟 DW 都可能提高激活峰值。`--ep-overlap-early-attn-memory-release` 可以把反向 attention 提到前向专家 MLP 之前，先释放一部分旧激活，再分配新激活；代价是原本由 attention backward 覆盖的部分 A2A 可能重新暴露。通信 kernel 也会争用 GPU 的计算资源和内存带宽，是否加速应以迭代时间为准。[兼容性检查](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py)、[提前释放的调度位置](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/model_chunk_schedule_plan.py)

## 05 · 输出投影 DW 延后：利用流水线排空窗口 { #embedding-wgrad }

最后一个 stage 的词表输出投影通常较大。它的反向既要计算传回 Transformer 的 DX，也要计算输出权重的 DW。DX 决定前面各 stage 何时继续反向，而 DW 可以在本次参数更新前完成，因此两者有不同的紧迫程度。

`--defer-embedding-wgrad-compute` 利用这一点。虽然参数名中有 embedding，常规 GPT 路径实际处理的是**末端词表输出线性层的权重梯度**，包括与输入 embedding 共享权重的情况。以四卡、每卡一个 stage 为例：

1. 前向保存输出投影的输入；反向保存对应的输出梯度。
2. 优先计算 DX，继续本 stage 的反向并尽早向前一个 stage 传梯度；DW 暂不执行。
3. GPU3 传出最后一个 micro-batch 的梯度后，GPU2 → GPU1 → GPU0 还要依次完成它们的反向。GPU3 利用这段时间调用 `finish_embedding_wgrad_compute()`，补算缓存的 DW。
4. DW 完成后，才能完成相关梯度同步并更新参数。

![GPU3 补算输出投影 DW，同时 GPU2、GPU1、GPU0 依次完成最后一个 micro-batch 的反向](assets/04-pp-embedding-wgrad.svg){ style="max-width: 780px;" }

*图中延后的 DW 全部落在其他卡的反向时间内；若 DW 更长，超出的部分仍会延长迭代。*

这项优化不减少总计算量；只有延后的 DW 能落入原来的空闲窗口，才会缩短迭代。代价是额外保存输入与输出梯度。`wgrad_deferral_limit` 限制延迟的 micro-batch 数，0 表示全部延迟；当前实现要求 PP 大于 1，并启用梯度累积融合。它与上面的 `delay_wgrad_compute` 作用范围不同，不应互相替代。[输出层 DW 收尾](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)、[配置与条件](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/model_parallel_config.py)

## 06 · PP 激活管理与参数预取的配合 { #memory-prefetch }

更细的调度和更多重叠可能增加在途状态。Megatron 还在 PP 执行过程中提供以下配合机制：

| 机制 | 如何生效 | 代价或边界 |
| --- | --- | --- |
| `deallocate_pipeline_outputs` | 边界输出发送完成后，将不再需要的数据存储释放，仅保留反向所需的图连接，并通过配套 backward 路径执行 | 只处理边界输出，不能释放层内反向仍需要的全部激活；异步发送完成前不能释放 |
| 按 micro-batch 调整重计算 | `num_microbatches_with_partial_activation_checkpoints` 在在途窗口内，让部分任务采用较轻的 checkpoint 策略，其余任务采用完整重计算 | 用额外前向计算降低激活峰值；依赖模型 forward 对 checkpoint 标志的支持，也须满足 MoE overlap 的限制 |
| 参数 All-Gather 预取 | 使用分布式优化器时，`--overlap-param-gather` 提前收集后续计算需要的参数，并在使用前等待；`align_param_gather` 在交错 PP 中协调各 stage 的发起时机 | 这是 DP 参数通信；它与 PP P2P 共用资源，过多并发可能争抢带宽。当前 CLI 还要求开启梯度归约重叠 |

重计算和输出释放解决激活存储，参数预取解决参数通信等待，三者作用对象不同。应从时间线和显存记录分别确认收益，而不能只看某个配置已经开启。[PP 调度与输出释放](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)、[参数同步实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/param_and_grad_buffer.py)

## 07 · 从时间线判断优化是否有效 { #validation }

先固定模型、全局 batch、精度与设备映射，记录稳态迭代时间和各卡峰值显存，再针对现象调整：

| 观察到的现象 | 优先核对的优化 |
| --- | --- |
| 某个 stage 长期更慢，其他卡反复等待它 | 先平衡首尾特殊层与 Transformer 层的耗时 |
| 启动、排空占比较高 | 核对 micro-batch 数与 VPP 粒度，再评估输出投影 DW 延后 |
| 每次 chunk 切换后都有 P2P 等待 | 检查异步收发、预接收与真正的等待位置 |
| MoE A2A 占主导，另一个 micro-batch 有独立计算 | 评估 `overlap-moe-expert-parallel-comm` 及 DW 延迟 |
| 开启重叠后显存峰值上升 | 检查在途前后向、延迟 DW 和通信 buffer 的寿命，再选择提前释放或兼容的重算策略 |

验收时比较实际迭代时间，而不只看通信与计算是否在图上交叉。如果两者并发后都变慢，或节省的通信时间被额外预热、重算和同步抵消，就没有获得端到端收益。

!!! tip "自测"

    1. Megatron 的 P2P overlap 与 MoE overlap 分别覆盖哪类通信？为什么发起异步请求后立即等待通常不能形成重叠？
    2. 输出投影 DW 延后为何可能缩短流水线排空时间，又为什么会增加显存？

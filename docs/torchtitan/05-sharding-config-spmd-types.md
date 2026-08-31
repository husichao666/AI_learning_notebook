---
title: "第 5 章 · ShardingConfig 与 spmd_types 后端"
description: "理解 TorchTitan 如何用 ShardingConfig 声明状态和激活布局，并在默认 spmd_types 后端中将它们落成本地 Tensor、SPMD 类型与显式集合通信。"
type: source-note
status: growing
level: intermediate
updated: 2026-08-29
tags: [torchtitan, pytorch, sharding-config, spmd-types, tensor-parallel, collective]
---

# 第 5 章 · ShardingConfig 与 spmd_types 后端

<div class="notebook-hero" markdown>

<span class="chapter-kicker">TorchTitan · spmd_types 路线 · 第 5 章</span>

前两章沿 `partial_dtensor` 路线，看到了 DTensor 怎样传播 Placement，以及 `ColwiseParallel` 怎样把参数分片和模块边界转换挂到普通 Linear 外面。从这一章开始，我们换到 TorchTitan 当前默认的 `spmd_types` 路线。

两条路线读的是同一份 `ShardingConfig`，但后面的执行方式不同：`partial_dtensor` 把状态和激活变成 DTensor，`spmd_types` 则尽量让模型继续使用普通的本地 Tensor，用独立的类型信息说明它们在各条 mesh axis 上是什么状态，并把真正改变布局的 collective 明确写进执行过程。

</div>

!!! info "版本与阅读范围"
    本文以 TorchTitan 提交 [`a3168782c`](https://github.com/pytorch/torchtitan/tree/a3168782c9a3a2e40afbd0de114818b96e2bda6e)为基准，主要对应 [`protocols/sharding.py`](https://github.com/pytorch/torchtitan/blob/a3168782c9a3a2e40afbd0de114818b96e2bda6e/torchtitan/protocols/sharding.py)、[`protocols/module.py`](https://github.com/pytorch/torchtitan/blob/a3168782c9a3a2e40afbd0de114818b96e2bda6e/torchtitan/protocols/module.py)、[`distributed/spmd_types.py`](https://github.com/pytorch/torchtitan/blob/a3168782c9a3a2e40afbd0de114818b96e2bda6e/torchtitan/distributed/spmd_types.py)与 [`decoder_sharding.py`](https://github.com/pytorch/torchtitan/blob/a3168782c9a3a2e40afbd0de114818b96e2bda6e/torchtitan/models/common/decoder_sharding.py)。TorchTitan 当前固定依赖 `spmd_types==0.2.5`，这套接口仍在快速演进。

    本章关心的是配置怎样进入默认后端，不再重复解释列并行和行并行的计算原理。需要回顾时，可以直接阅读仓库已有的 [Tensor Parallel](../training/parallelism/03-tp.md)。

## 1. ShardingConfig 解决的是什么问题

模型作者通常希望保留这样的单卡代码：

```python
class FeedForward(Module):
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

真正启用 TP、SP 或 CP 后，框架还要知道：

1. `w1`、`w2`、`w3` 的参数分别沿哪一维切；
2. `x` 进入整个 FFN 前是否需要 all-gather；
3. 每个 Linear 的本地输出在各条 mesh axis 上是什么语义；
4. `w2` 产生的局部贡献最后做 all-reduce 还是 reduce-scatter；
5. fused kernel 是否只接受本地 Tensor。

`ShardingConfig` 就是附着在模块配置上的这份分布式契约。它不实现 Linear，也不执行 collective，只描述模块的状态、入口和出口应当采用什么布局。模型构建完成后，`Module.parallelize()` 再读取这些声明，选择 `partial_dtensor` 或 `spmd_types` 后端落地。

这样一来，单卡 `forward()` 只负责计算关系，并行配置则与对应模块的 `Config` 放在一起。模型不需要为了不同并行度复制多套 forward，也不再依赖一张以字符串 FQN 为 key 的全局 `ParallelStyle` plan。

## 2. ShardingConfig 的六个字段

当前 [`ShardingConfig`](https://github.com/pytorch/torchtitan/blob/a3168782c9a3a2e40afbd0de114818b96e2bda6e/torchtitan/protocols/sharding.py)包含六个字段：

| 字段 | 描述的时刻 | 作用 |
| --- | --- | --- |
| `state_shardings` | 并行化阶段 | 声明当前模块直接持有的 parameter 和 buffer 怎样分布 |
| `in_src_shardings` | 进入模块时 | 声明输入现在是什么类型 |
| `in_dst_shardings` | 执行 forward 前 | 声明输入需要转换成什么类型 |
| `out_src_shardings` | forward 刚结束 | 声明原始输出是什么类型 |
| `out_dst_shardings` | 离开模块前 | 声明输出需要转换成什么类型 |
| `local_map` | forward 内部 | 将一段只接受本地 Tensor 的计算标成有明确输入输出类型的局部区域 |

输入和输出都拆成 `src → dst`，是因为 `spmd_types` 不从 Tensor 包装对象上读取 Placement。`src` 明确告诉类型系统“转换前是什么”，`dst` 再说明“转换后要什么”。只有 `src` 时，这个边界可以只做类型检查；一旦填写 `dst`，`spmd_types` 后端就可能在这里执行显式 collective。

`state_shardings` 采用更严格的口径：只要一个模块设置了 `ShardingConfig`，它直接持有的每个 parameter 和非空 buffer 都必须有对应声明，缺少条目会直接报错。这样可以避免新增参数后悄悄落成未定义布局。

![ShardingConfig 在 spmd_types 后端中的初始化与运行时路径](assets/05-sharding-config-spmd-types.svg)

*图 1：配置先随 Module.Config 进入模块实例；`parallelize()` 在初始化阶段切分状态并包装 forward，训练时再按 src/dst 契约执行显式 collective。中间绿色区域仍是模型原来的单卡 forward。*

表中所有带 `shardings` 的字段都使用 `SpmdType` 表示布局。它是两种后端共同读取的配置类型：`partial_dtensor` 会把它解析成 DTensor Placement，默认的 `spmd_types` 后端则直接用它标注本地 Tensor。继续跟踪配置的执行过程前，需要先看清这个类型由什么组成。

## 3. SpmdType：ShardingConfig 的布局类型

`SpmdType` 来自同名的 `spmd_types` 库，是一个描述分布布局的 Python 元数据对象，不是 Tensor 子类，也不会像 DTensor 那样包装本地 buffer。一个 `SpmdType` 可以包含两层信息：

```text
local_type      每条 mesh axis 上，本地值和反向梯度是什么关系
PartitionSpec   本地 shard 应怎样沿张量维度拼回逻辑全局张量
```

### 3.1 local_type：每条 mesh axis 上的值与梯度语义

[`spmd_types` 的类型定义](https://github.com/meta-pytorch/spmd_types/blob/main/docs/local_spmd_types.md#types)不仅关心前向值是否相同，还关心反向梯度是否等待归约：

| 类型 | 前向在该 axis 上 | 对应的梯度类型 | 常见含义 |
| --- | --- | --- | --- |
| `R`，Replicate | 各 rank 的值相同 | `P` | 同一个值被各 rank 用于不同计算，梯度需要跨 rank 求和 |
| `I`，Invariant | 各 rank 的值相同 | `I` | 各 rank 做相同计算，梯度也保持相同，不等待跨 rank 求和 |
| `V`，Varying | 各 rank 的值不同 | `V` | 只说明值不同，不单独说明沿哪个 tensor dimension 切分 |
| `P`，Partial | 各 rank 保存同形状的局部贡献 | `R` | 前向结果还有一次 sum reduction 没做 |

这里最容易混淆的是 `R` 和 `I`：它们的前向 buffer 都可以完全相同，区别发生在反向。`R` 表示每个 rank 对这个值有一份独立使用，因此产生的梯度要汇总；`I` 表示各 rank 做的是同一份重复计算，梯度不应再被当作待求和贡献。

`S(d)` 可以看成比 `V` 更具体的写法：除了说明各 rank 的值不同，还指出逻辑张量沿第 `d` 维切分。它的反向仍是同维度的 shard。

### 3.2 PartitionSpec：mesh axis 与张量维度的映射

当多条 mesh axis 同时切同一个 tensor dimension 时，只写每轴 `S(d)` 不足以表达切分顺序。TorchTitan 因此经常使用 `V + PartitionSpec`：

```python
SpmdType(
    {
        DP: spmd.V,
        CP: spmd.V,
        TP: spmd.V,
    },
    partition_spec=spmd.PartitionSpec((DP, CP), TP),
)
```

对一个二维激活来说，`PartitionSpec((DP, CP), TP)` 表示：

- tensor dim 0 先后由 DP、CP 切分；
- tensor dim 1 由 TP 切分。

因此 `V` 回答的是“各 rank 的值是否不同”，`PartitionSpec` 回答的是“这些不同的本地值怎样组成逻辑全局张量”。当前 rank 手里的对象依然是普通 Tensor，所以 `tensor.shape` 是本地 shape；逻辑全局布局保存在这份可擦除的类型信息中，而不是 Tensor wrapper 里。

## 4. ShardingConfig 如何进入模块实例

以列并行 Linear 为例，TorchTitan 的共用 helper 写成：

```python
def colwise_config() -> ShardingConfig:
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        out_src_shardings=dense_activation_placement(
            tp=spmd.S(-1),
            cp=spmd.S(0),
        ),
    )
```

`dense_param_placement()` 会把 DP、CP 都写成 `R`，再使用调用方给出的 TP 类型。因此列并行权重的完整声明可以读成：

```text
DP: R        参数在数据并行方向复制，梯度等待 FSDP 归约
CP: R        参数在上下文并行方向复制，梯度也需要归约
TP: S(0)     weight[out, in] 沿输出维切分
```

列并行 Linear 自身没有填写 `in_src_shardings` 和 `in_dst_shardings`。这是有意的：Attention 或 FFN 父模块已经在更外层把输入准备成 TP Replicate，`w1` 和 `w3` 可以共用一次 all-gather，不需要分别通信。

模型专属的 sharding helper 会把这些对象写进各级 `Module.Config`。例如 Llama 先调用 `set_llama3_sharding_config()` 填好 Attention、FFN、Norm、Embedding 和 `lm_head` 的配置；随后 `Config.build()` 构造模块时，将配置保存到实例的 `_sharding_config`。到这里仍然只是在声明布局，还没有执行参数切分或 collective。

## 5. parallelize() 在初始化阶段做什么

模型进入 [`parallelize_llama()`](https://github.com/pytorch/torchtitan/blob/a3168782c9a3a2e40afbd0de114818b96e2bda6e/torchtitan/models/llama3/parallelize.py) 后，会调用一次：

```python
model.parallelize(parallel_dims)
```

`Module.parallelize()` 先递归处理子模块，再对每个带 `_sharding_config` 的模块完成四件事：

1. 检查当前 `src → dst` 是否能由现有 helper 表达；
2. 按 `state_shardings` 分发当前模块直接持有的状态；
3. 记录原 `forward()` 的位置参数名；
4. 用“处理输入 → 原 forward → 处理输出”的函数包装 `self.forward`。

这里和上一章的原生 `ColwiseParallel` 有一点实现差异：TorchTitan 没有为 `ShardingConfig` 注册 `nn.Module.forward_pre_hook` 和 `forward_hook`，而是直接重新绑定 `self.forward`。但它保存并调用的仍是原函数，所以模型源码中的单卡 forward 不需要修改。

### 5.1 spmd_types 如何分发参数和 buffer

在 `spmd_types` 分支中，`_distribute_states()` 不调用 `torch.distributed.tensor.distribute_tensor()`，而是进入 `spmd_distribute_tensor()`：

```python
tensor = spmd.shard(
    tensor,
    mesh.get_group(axis),
    src=spmd.I,
    dst=spmd.S(dim),
)

self.register_parameter(name, nn.Parameter(tensor))
spmd.assert_type(self._parameters[name], layout)
```

`spmd.shard()` 按声明取出当前 rank 应保存的 slice，重新注册的 Parameter 仍是普通 `torch.Tensor`。`spmd.assert_type()` 再把这份本地 Tensor 与对应的 `SpmdType` 关联起来，供类型传播和边界检查使用。

这和 `partial_dtensor` 的结果很不一样：后者注册的是 `DTensor Parameter`，`weight.shape` 保留逻辑全局 shape；这里刚分片后的 `weight.shape` 就是当前 rank 的本地 shape。

## 6. forward 包装怎样插入显式通信

删掉参数绑定和异常检查后，运行时包装可以概括成：

```python
original_forward = self.forward

def forward_with_redistribution(*args, **kwargs):
    args, kwargs = self._redistribute_inputs(parallel_dims, args, kwargs)
    outputs = original_forward(*args, **kwargs)
    return self._redistribute_outputs(parallel_dims, outputs)

self.forward = forward_with_redistribution
```

### 6.1 输入：先确认 src，再转换到 dst

如果配置声明了输入布局，`_redistribute_inputs()` 会按原 `forward()` 的参数名找到对应 Tensor：

1. 类型检查开启时，用 `spmd.assert_type(value, src)` 确认调用方传来的类型；
2. 没有 `dst` 时，直接把它交给原 forward；
3. 有 `dst` 时，调用 `spmd_redistribute_per_axis(value, mesh, src, dst)`。

与 DTensor 不同，这里不会先执行 `DTensor.from_local()`，也不会让 `redistribute()` 自己读取输入 Placement。源类型和目标类型都明写在配置里。

### 6.2 输出：原 forward 先产出 src

输出方向完全对称：原 forward 先产生 `out_src_shardings` 声明的结果，类型检查器核对这个契约，然后再按 `out_dst_shardings` 执行转换。

当前 helper 会逐 axis 比较 src 和 dst，并对发生变化的那条 axis 调用：

```python
x = spmd.redistribute(
    x,
    mesh.get_group(axis),
    src=src_type,
    dst=dst_type,
)
```

常见转换仍对应熟悉的 collective：

| 类型转换 | 常见 collective |
| --- | --- |
| `S(d) → R` | `all_gather` |
| `P → R` 或 `P → I` | `all_reduce` |
| `P → S(d)` | `reduce_scatter` |

区别在于，这些通信现在是显式的带类型操作。第 3 章的 DTensor 路线先选目标 Placement，再由 `DTensor.redistribute()` 规划路径；这里的图中会直接出现一个具有 `src`、`dst` 和 process group 的 `spmd.redistribute`。

当前 TorchTitan 的边界 helper 还有明确限制：一次 src/dst 配对至多改变一条 mesh axis，不能在一次配置转换中重排多个 shard axis，也不允许以语义不够明确的 `V` 作为变化轴的源或目标。复杂转换需要在模块内部写成显式 collective，而不是期待配置层自动规划任意路径。

## 7. FFN 中的输入 all-gather 与输出 reduce-scatter

开启 Sequence Parallel 后，一个普通 FFN 的核心配置是：

```python
feed_forward_cfg.sharding_config = ShardingConfig(
    in_src_shardings={"x": sequence_parallel_layout},
    in_dst_shardings={"x": replicated_tp_layout},
)

feed_forward_cfg.w1.sharding_config = colwise_config()
feed_forward_cfg.w3.sharding_config = colwise_config()
feed_forward_cfg.w2.sharding_config = rowwise_config(output_sp=True)
```

运行时按模块嵌套顺序发生：

```text
FFN 入口
  x: TP 上按 sequence 切分
  └─ spmd.redistribute：all-gather → TP Replicate

原 FeedForward.forward(x)
  ├─ w1：本地输入 × Shard(0) 权重 → feature shard
  ├─ w3：本地输入 × Shard(0) 权重 → feature shard
  ├─ SiLU 与逐元素乘：继续本地计算
  └─ w2：feature shard × Shard(1) 权重 → Partial
       └─ spmd.redistribute：reduce-scatter → sequence shard

FFN 出口
  重新留下适合下一个模块消费的 sequence shard
```

这里能看出为什么输入 all-gather 放在父 FFN，而不是分别放在 `w1` 和 `w3`：父模块只通信一次，两个列并行投影共同消费准备好的输入。`w2` 的配置则把本地矩阵乘结果明确声明为 `P`，再显式转换成 sequence shard。

整个过程中，原来的：

```python
self.w2(F.silu(self.w1(x)) * self.w3(x))
```

一个字也不用改。并行逻辑来自父子模块各自的配置与 forward 包装，而不是另一份“分布式 FFN”代码。

## 8. 类型检查怎样接进训练入口

普通 Tensor 自己不知道所属 mesh 和 SPMD 类型，因此整段 forward 需要在正确的 mesh 上下文中运行。Trainer 构造 `train_context` 时，会为 `spmd_types` 后端注册 dense 与 sparse mesh，并用 `set_current_spmd_mesh()` 激活当前计算区域。

模型输入则在 `preprocess_inputs()` 中通过 `annotate_input_spmd_types()` 标注。每个直接传入模型的 Tensor 都必须有布局声明，不能把没有类型的输入悄悄送进后续传播。

当 `debug.spmd_typechecking` 开启时，Trainer 还会进入 `spmd_types.checker.typecheck(local=False)`。类型检查器只需运行 forward，就能沿普通算子传播类型，并检查：

- 模块调用方提供的 src 是否满足被调用模块的入口契约；
- `R`、`I`、`V`、`P` 的前反向关系是否一致；
- `Partial` 是否错误地穿过非线性算子；
- collective 声明的 src 和 dst 是否接得上；
- `PartitionSpec` 能否为本地算子提供一致的全局解释。

这套检查不负责替程序选择并行策略，也不会像 DTensor dispatcher 那样在普通算子前自动补一次通信。它检查的是已经写下来的本地计算和显式 collective 是否组成一个自洽的分布式程序。[`spmd_types` 的设计文档](https://github.com/meta-pytorch/spmd_types/blob/main/docs/design.md#global-spmd)将这种更严格的模式称为 Global SPMD 类型检查。

## 9. LocalMapConfig：为 fused kernel 声明类型边界

有些 attention、RoPE 或 fused kernel 只理解本地 Tensor，类型检查器也未必认识它内部的自定义算子。`LocalMapConfig` 用来给这类区域画出边界。

在 `partial_dtensor` 后端，它会变成 PyTorch 的 `local_map()`：入口把 DTensor 转成本地 Tensor，出口再包装回 DTensor。在 `spmd_types` 后端，TorchTitan 使用：

```python
spmd.no_typecheck(
    in_types=in_types,
    out_types=out_types,
)(original_forward)
```

这不是把整块计算的分布语义丢掉，而是把它当成一个带签名的黑盒：类型系统不进入 kernel 内部，但仍检查入口类型，并按配置声明出口类型。这样自定义 kernel 可以继续操作普通 Tensor，同时与外部的 DP、CP、TP 布局契约接起来。

## 10. FSDP 与 checkpoint 的 DTensor 边界

“`spmd_types` 使用普通 Tensor”说的是模型前向/反向的主要计算表示，不等于 TorchTitan 从此完全不使用 DTensor。

在当前初始化顺序中，`model.parallelize()` 先按 TP 等模型并行 axis 切状态并附加 SPMD 类型，随后 `apply_fsdp_to_decoder()` 再调用 FSDP2。FSDP2 仍使用 DTensor 表示参数存储布局，并通过模块调用周围的 hook 完成参数 all-gather 与梯度 reduce-scatter。

因此 TorchTitan 为状态传递保留了两座桥：

- `plain_tensor_to_dtensor_state_dict()`：把带 SPMD 布局的本地状态包装成 DTensor；
- `dtensor_to_plain_tensor_state_dict()`：取回 DTensor 的本地部分，交给以普通 Tensor 为计算表示的路径。

可以把两种职责分开理解：

```text
模型计算语义     普通 Tensor + SpmdType + 显式 collective
参数存储与恢复   FSDP2 / DTensor + state_dict bridge
```

## 11. spmd_types 对计算图捕获的影响

`spmd_types` 的一个重要设计目标是类型可擦除：关闭类型检查后，模型计算仍然是普通 Tensor 算子，collective 仍然是代码中的显式操作。编译器看到的核心结构更接近：

```text
local ATen ops
→ explicit all-gather
→ local ATen ops
→ explicit reduce-scatter
```

通信位置不需要等 DTensor dispatcher 在运行时根据 Placement 推导，算子图里也不必让每个激活长期携带 DTensor wrapper。这样更方便编译器捕获、分析并重新调度计算与通信。

不过 `spmd_types` 不是自动并行器。`ShardingConfig` 仍由 TorchTitan 明确写出，复杂 collective 也仍要由实现者选择；类型系统负责验证这些选择是否自洽，而不是替模型搜索最优切分方案。

## 12. 小结

`ShardingConfig` 是两种 SPMD 后端共用的声明层：`state_shardings` 处理参数和 buffer，输入与输出的 src/dst 描述模块边界，`LocalMapConfig` 则给本地 kernel 标出类型明确的黑盒区域。

进入默认 `spmd_types` 后端后，状态先被切成当前 rank 的普通 Tensor，并附加 `SpmdType`；每次模块调用再由 forward 包装检查 src、执行显式 `spmd.redistribute()`、运行原来的单卡 forward，最后整理输出。`R` 与 `I` 的差别补上了反向梯度是否待归约的语义，`PartitionSpec` 则说明本地 shard 怎样重建逻辑全局张量。

这条路线保留了单卡模型的写法，也让 collective 在计算图中变得明确。FSDP2 和 checkpoint 边界仍可继续使用 DTensor，它们负责的是参数存储与状态传递，不需要和模型计算采用完全相同的运行时表示。

---

上一章（`partial_dtensor` 路线）：[使用 ColwiseParallel 切分模型](04-colwise-parallel.md)

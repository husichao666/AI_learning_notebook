torchtitan：(二) DeviceMesh与Placement详解与使用
本文将详细解释torch原生的DeviceMesh类和Placement类的原理，从定义到代码使用示例，帮你从读懂到熟练使用。
引言
在上一节，我们详细介绍了DTensor的定义和使用方法。其中我们提到了：
DTensor = 本地张量local_tensor+ 分布规格DTensorSpec 
但并没有深入DTensorSpec，搞清楚其中的DeviceMesh与Placement 的详细逻辑。本章将对这两个类进行详细的介绍，并结合代码，实战出真知。

系列导读
​torchtitan：(一) DTensor 原理简介与使用 - 知乎
​torchtitan：(二) DeviceMesh与Placement详解与使用 - 知乎
(三....) 待补充。。。
DeviceMesh：设备的分布式拓扑结构
简介
作为一个训练大模型的工程师，你最关心的问题往往是：
我有多少张卡？这些卡是怎么分布在不同机器上的？
哪几张卡一起做数据并行？哪几张卡一起做张量并行 / pipeline 并行？
在 PyTorch 2.3 之前，我们需要手动创建、管理各种 ProcessGroup，代码里到处是「这个 group 给 DP 用，那个 group 给 TP 用」，既难读也难维护。
DeviceMesh 的设计就是为了解决这个痛点，它提供了一个更直观的视角：
把所有参与训练的设备，想象成排成一个 N 维“网格”（mesh）；
每一维对应一种并行方式（例如 dp、tp、pp、ep 等）；
你只需要在一处把这个网格描述清楚，后续所有并行 API（DTensor、FSDP、TP/CP/PP 等）都围绕这个 mesh 来工作。
从使用者的角度，可以简单地把 DeviceMesh 理解成：
一份“设备拓扑说明书”：负责告诉 PyTorch「这些 rank 是怎么组织成 dp / tp / pp / ep 的」；
一层统一的抽象：上层不用再关心具体有哪些进程组，只和 mesh / 子 mesh 打交道；
一个易组合的底座：同一批 GPU 上，可以方便地叠加多种并行策略，而不会把进程组逻辑写满全局。
这也是 torchtitan 选择围绕 ParallelDims + DeviceMesh 来构建并行系统的核心原因：让“配置并行策略”这件事，从到处散落的进程组调用，变成一处集中声明、处处按 mesh 使用。
用法与代码详解
这一小节分成两块：
初始化方式：重点介绍 init_device_mesh() 的几种常见用法；
常用属性 / 方法：例如 mesh["tp"]、_flatten()、size()、get_local_rank() 等。
1. 初始化方式：init_device_mesh()
在实际工程里，推荐使用 init_device_mesh() 来构建 DeviceMesh，而不是直接 new DeviceMesh。 它的典型签名是：
init_device_mesh(
    device_type: str,
    mesh_shape: Sequence[int] | None = None,
    mesh_dim_names: Sequence[str] | None = None,
    # 还有一些高级参数，这里不展开
)
device_type："cuda" / "cpu"，决定使用哪种设备上的默认进程组；
mesh_shape：每一维的大小，例如 (dp, tp)、(pp, dp, tp)；
mesh_dim_names：每一维的名字，用于后续索引和创建子 mesh。
下面用几个小例子说明。
（1）一维：纯数据并行
假设 WORLD_SIZE = 4，我们只做一维的数据并行，把所有 rank 看成一个 1D mesh：
from torch.distributed.device_mesh import init_device_mesh

dp_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(4,),          # 1 维，大小为 4
    mesh_dim_names=("dp",),   # 维度名为 "dp"
)
这时候：
dp_mesh.shape == (4,)，代表 4 个数据并行 rank；
后续你可以把 dp_mesh 交给 DTensor、FSDP 等 API 作为「数据并行 mesh」使用。
（2）二维：dp × tp
再看一个更贴近大模型训练的例子：WORLD_SIZE = 8，我们想要：
2 路数据并行（dp = 2）；
每个数据并行组里 4 路张量并行（tp = 4）。
from torch.distributed.device_mesh import init_device_mesh

world_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(2, 4),            # 2 x 4
    mesh_dim_names=("dp", "tp"),  # 第 0 维是 dp，第 1 维是 tp
)
这相当于在内部构造了一个 2 x 4 的 rank 矩阵，例如：
[[0, 1, 2, 3],
 [4, 5, 6, 7]]
第 0 维（大小 2）是数据并行维度；
第 1 维（大小 4）是张量并行维度；
以后在 "dp" 这维做 all-reduce，就是跨组的数据并行通信；
在 "tp" 这维做 all-reduce，就是组内的张量并行通信。
（3）多维：pp × dp × tp（示意）
更复杂的情况（例如 pp × dp × tp）也可以用同样方式声明：
world_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(pp, dp, tp),
    mesh_dim_names=("pp", "dp", "tp"),
)
实际训练框架会在这个 world_mesh 上再切分出不同的子 mesh，分别负责：
纯数据并行（dp）；
FSDP / 参数分片（dp_shard + cp）；
TP / CP / PP 等。
2. 常用属性 / 方法
DeviceMesh 创建出来后，最常用的几个属性 / 方法如下。
（1）基本属性：mesh、shape、ndim、mesh_dim_names
print(world_mesh.mesh)            # tensor([[0, 1, 2, 3],
                                  #         [4, 5, 6, 7]])  rank 在 2x4 mesh 上的排布
print(world_mesh.shape)           # torch.Size([2, 4])      mesh 形状
print(world_mesh.ndim)            # 2                       mesh 维度数
print(world_mesh.mesh_dim_names)  # ('dp', 'tp')           每一维的名字
这几项主要用来检查 mesh 形状是否符合预期。
（2）按名字取子 mesh：mesh["tp"]、mesh[("dp", "tp")]
如果在初始化时提供了 mesh_dim_names，就可以用名字来取子 mesh。注意：这里返回的是“当前 rank 所在组”的子 mesh，即不同 rank 看到的子 mesh 内容可能不同，但代码是 SPMD 的（每个 rank 都执行同一段 Python）。
下面用官方文档里的例子来说明（WORLD_SIZE = 8）：
from torch.distributed.device_mesh import init_device_mesh

# 初始化一个 2D mesh，形状为 2x4，维度名为 ("dp", "tp")
mesh_2d = init_device_mesh(
    device_type="cuda",
    mesh_shape=(2, 4),
    mesh_dim_names=("dp", "tp"),
)

tp_mesh = mesh_2d["tp"]   # 取出 "tp" 这一维
dp_mesh = mesh_2d["dp"]   # 取出 "dp" 这一维

# 下面这些说明“在不同 rank 上运行同一段代码时，看到的子 mesh 内容”：

# tp 方向的子 mesh：
#   - 在 rank 0, 1, 2, 3 上：tp_mesh.mesh == tensor([0, 1, 2, 3])
#   - 在 rank 4, 5, 6, 7 上：tp_mesh.mesh == tensor([4, 5, 6, 7])
#
# dp 方向的子 mesh：
#   - 在 rank 0, 4 上：dp_mesh.mesh == tensor([0, 4])
#   - 在 rank 1, 5 上：dp_mesh.mesh == tensor([1, 5])
#   - 在 rank 2, 6 上：dp_mesh.mesh == tensor([2, 6])
#   - 在 rank 3, 7 上：dp_mesh.mesh == tensor([3, 7])
对于多维子 mesh，可以传入一个 维度名元组/列表，返回一个多维 mesh。仍然以官方文档中的 3D mesh 为例：
mesh_3d = init_device_mesh(
    device_type="cuda",
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("dp", "pp", "cp"),
)

dp_cp_mesh = mesh_3d["dp", "cp"]  # 先 dp 再 cp
cp_dp_mesh = mesh_3d["cp", "dp"]  # 先 cp 再 dp（注意维度顺序不同）

# 在 rank 0, 1, 4, 5 上：
#   dp_cp_mesh.mesh == tensor([[0, 1], [4, 5]])
#   cp_dp_mesh.mesh == tensor([[0, 4], [1, 5]])
#
# 在 rank 2, 3, 6, 7 上：
#   dp_cp_mesh.mesh == tensor([[2, 3], [6, 7]])
#   cp_dp_mesh.mesh == tensor([[2, 6], [3, 7]])
#
# 可以看到：传入的维度名顺序 ("dp", "cp") / ("cp", "dp")
# 决定了返回的子 mesh 中维度的排列顺序。
单个字符串：取出这一维对应的一维 mesh（当前 rank 所在的那条“线”）；
元组 / 列表：取出若干维度组成的多维子 mesh（当前 rank 所在的那“块”）。
在 torchtitan 里，ParallelDims 会先用这种方式组合出多个子 mesh，然后再对它们做 _flatten()。
（3）扁平化多维为一维：_flatten(mesh_dim_name="dp")
很多时候，我们并不希望上层代码看到多维结构，而是只想要一个「逻辑 dp 维度」。 这时可以用 _flatten() 把若干维合成一个新的一维：
dp_like_mesh = world_mesh[("dp", "dp_shard")]
dp_like_mesh._flatten(mesh_dim_name="dp")  # 把两维合成一维，命名为 "dp"
扁平化之后：
dp_like_mesh.mesh_dim_names == ("dp",)；
可以通过 world_mesh["dp"] 直接拿到这条逻辑 dp 维度。
在 ParallelDims._build_mesh_without_ep() 中就大量使用了这种模式，用不同的维度组合出：
"dp"：数据并行；
"dp_shard_cp"：FSDP + CP；
"dp_cp"：loss all-reduce 等。
（4）进程信息：size()、get_local_rank()
在训练逻辑中，经常需要知道「当前 mesh 有多少个 rank」「当前 rank 在这个 mesh 中是第几个」：
dp_mesh = world_mesh["dp"]

dp_degree = dp_mesh.size()          # 数据并行的进程数
dp_rank = dp_mesh.get_local_rank()  # 当前进程在 dp 组里的局部 rank
在 torchtitan 里，这两个值会被用来：
配置 dataloader 的 shard（每个 dp_rank 读自己的数据分片）；
计算 global_batch_size 与 local_batch_size 的关系；
控制日志 / 评估只在 rank=0 或特定 rank 上执行等。
torchtitan中的用法
torchtitan 把「并行度」和「设备拓扑」两件事分开建模：
ParallelDims：描述并行维度的大小（dp/cp/tp/pp/ep 等），是一个纯配置对象。
DeviceMesh（world_mesh）：根据 ParallelDims 真正创建出来的物理/逻辑设备拓扑。
1. ParallelDims：声明并行形状
在 torchtitan/distributed/parallel_dims.py 中，ParallelDims 把各种并行度统一收集起来：
dp_replicate：普通数据并行的复制数；
dp_shard：参数分片的数据并行度（FSDP）；
cp：Context Parallel 维度；
tp：Tensor Parallel 维度；
pp：Pipeline Parallel 维度；
ep / etp：Expert Parallel 相关维度；
world_size：全局进程数。
_validate 会检查这些并行度是否与 WORLD_SIZE 一致，并推导缺省的 dp_shard 等，保证：
2. build_mesh：从并行度到 DeviceMesh
ParallelDims.build_mesh() 利用 init_device_mesh 来构建真正的 DeviceMesh：
挑出所有并行度大于 1 的维度（例如 "pp", "dp_replicate", "dp_shard", "cp", "tp" 等）；
把它们的大小组成一个列表 dims，把名字组成 names；
调用：
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(
    device_type,      # "cuda" 或 "cpu"
    dims,             # 例如 [pp, dp_replicate, dp_shard, cp, tp]
    mesh_dim_names=names,
)
这一步拿到的是一个「世界 mesh」（world_mesh），它的每一维对应一种物理/逻辑并行方式。
之后 ParallelDims 会基于这个 mesh 预先构造常用的几个「逻辑子 mesh」并做扁平化，例如：
"dp"：用于数据并行 / 数据加载的 mesh；
"dp_shard_cp"：用于参数分片（FSDP）+ Context Parallel 的 mesh；
"dp_cp"：用于 loss all-reduce 等操作的 mesh；
"ep"：用于 Expert Parallel 的 mesh（当 ep > 1 时）。
代码中是通过类似下面的方式实现的（简化示意）：
# 选出某几个维度名作为一个子 mesh，然后把它们 flatten 成一个逻辑维度
dp_mesh = mesh[("dp_replicate", "dp_shard")]
dp_mesh._flatten(mesh_dim_name="dp")   # 之后 world_mesh["dp"] 就能直接拿到
这样，上层训练逻辑就不需要关心具体哪些物理维度组合成了 “dp”，只需要用 world_mesh["dp"] 即可。
3. 在 Trainer 中使用 world_mesh
在 torchtitan/train.py 的 Trainer.__init__ 里，DeviceMesh 被真正用起来：
先根据并行配置和 WORLD_SIZE 构造 ParallelDims：
world_size = int(os.environ["WORLD_SIZE"])
   parallelism_config = job_config.parallelism
   self.parallel_dims = parallel_dims = self._create_parallel_dims(
       parallelism_config, world_size
   )
通过 parallel_dims.world_mesh 懒加载创建 DeviceMesh：
world_mesh = parallel_dims.world_mesh
如果开启了数据并行，就从 world_mesh 中取出逻辑的 "dp" mesh：
if parallel_dims.dp_enabled:
       dp_mesh = world_mesh["dp"]
       dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
   else:
       dp_degree, dp_rank = 1, 0
dp_degree：当前作业中数据并行的总进程数；
dp_rank：当前 rank 在数据并行组中的局部 rank。
这些信息会继续传给 dataloader 和模型并行化逻辑，用来：
 
切分/分配训练数据（每个 dp rank 只读自己的 shard）；
构造 DTensor、FSDP、TP/CP/PP 等并行算子使用的通信组；
控制随机种子、梯度规约等等。

总结一下：在 torchtitan 中，ParallelDims 负责声明“要做哪几种并行、每种多少维度”，DeviceMesh 则负责在物理设备上搭建起对应的 N 维 mesh，并提供统一的子 mesh / 通信视图给上层训练代码使用。
Placement：Tensor的分布式状态
简介
在上一节里，DeviceMesh 只解决了一个问题：设备是怎么连在一起的。 但是「张量本身」在这些设备上可以有很多种摆法，这就是 Placement 要解决的事。
在 PyTorch DTensor 中，最核心的三种 Placement 是：
Replicate：在某个 mesh 维度上，Tensor 完整复制在该维度的所有 rank 上（常见于纯数据并行的权重 / 部分激活）。
Shard：在某个 mesh 维度上，按某个 tensor 维度把数据 切成 N 份，该维度上的每个 rank 持有其中一份（典型于张量并行 / FSDP 参数分片）。
Partial：在某个 mesh 维度上，rank 上持有的是某种 待规约的部分值（pending reduction），例如 sum/avg/max 等运算对应的「局部贡献」，只有在这条 mesh 维上做完一次规约（all-reduce / reduce-scatter）后，才得到最终的逻辑张量。
你可以简单地把它们理解为：
Replicate：大家都有一份一模一样的完整副本；
Shard：每人拿一块，合起来才是完整 Tensor；
Partial：每人手上拿着“还没做规约的那一份”，所有 rank 的值按约定做一次规约后，才等价于完整的 Tensor。
DeviceMesh 负责「把 GPU 摆成网格」，而 Placement 则负责「在这个网格上，具体这块 Tensor 怎么放」——两者组合起来，才构成了 DTensor 的完整分布式语义。
重排布的原理
当我们从一种 Placement 切换到另一种 Placement 时，本质上就是在不同 rank 之间移动 / 复制数据。 PyTorch 在 DTensor 里把这件事抽象成「根据源 Placement 和目标 Placement，自动插入必要的集合通信」。
下面用几个常见的转换来直观理解（这里只强调「语义上对应哪类 collective」，不严格区分具体算子如何产生这些 Placement）：
Replicate -> Shard（复制 → 切片）
 
初始状态：每个 rank 都有完整的一份 Tensor（Replicate）。
目标状态：每个 rank 只保留自己负责的那一块（Shard）。
通信：语义上不需要跨 rank 通信，只要按照 mesh 维上的约定切片即可（实现中通常是本地切片或只在创建时拉取自己那部分）。

Shard -> Replicate（切片 → 复制）
 
初始状态：每个 rank 只有一部分数据（Shard）。
目标状态：每个 rank 都有完整 Tensor（Replicate）。
通信：对应一次 all-gather，把所有 shard 沿着对应 mesh 维拼在一起，并让每个参与 rank 都拿到完整结果。

Partial -> Replicate（待规约的部分值 → 完整结果）
 
初始状态：每个 rank 上是某个 reduce_op（如 sum/avg）对应的「局部贡献」，Placement 为 Partial(reduce_op)。
目标状态：每个 rank 上都拿到规约后的完整张量，Placement 变成 Replicate。
通信：对应一次 all-reduce，在给定 mesh 维上对所有 rank 的 Partial 值做规约（如求和 / 求平均），结果在该维上变成 Replicate。

Partial -> Shard（待规约的部分值 → 规约后仍按 shard 分布）
 
初始状态：每个 rank 上是某个 reduce_op 对应的 Partial 值。
目标状态：规约完成后，结果仍旧以 Shard 的形式分布在各个 rank 上（每个 rank 只保留自己那一块）。
通信：对应一次 reduce-scatter，在同一 shard 位置对所有 rank 的值做规约（如求和），并把规约后的每一块分发给各个 rank，使结果保持 Shard 布局。

例子：用法与详解
这一节我们用几个小例子，配合 DeviceMesh 一起看 Placement 的实际效果。
1. 在一维 dp mesh 上的 Replicate / Shard
先构造一个简单的 1D 数据并行 mesh（假设 WORLD_SIZE = 4）：
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
import torch

mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(4,),
    mesh_dim_names=("dp",),
)

global_x = torch.arange(8, dtype=torch.float32)  # tensor([0., 1., ..., 7.])
（1）Replicate：每个 dp rank 都有一模一样的 global_x
dt_replicate = distribute_tensor(
    global_x,
    mesh,
    placements=[Replicate()],  # 在 "dp" 维上复制
)

print(dt_replicate.to_local())
# 在 rank 0/1/2/3 上，to_local() 都是 tensor([0., 1., ..., 7.])
（2）Shard：在 dp 维上按 dim=0 切 4 份
dt_shard = distribute_tensor(
    global_x,
    mesh,
    placements=[Shard(0)],  # 在 dim=0 上切分，沿 "dp" 维分发
)

print(rank, dt_shard.to_local())
# 假设 world_size=4：
#   rank 0: tensor([0., 1.])
#   rank 1: tensor([2., 3.])
#   rank 2: tensor([4., 5.])
#   rank 3: tensor([6., 7.])
同一个 DeviceMesh，只改 Placement，就从「全员一份完整副本」变成了「每人一块切片」。
2. 从 Shard 变回 Replicate：重排布 + 通信
继续沿用上面的 dt_shard，我们希望在某一步做一个需要完整张量的操作（比如某个非分布式的正则化），这时可以把它「重排布」成 Replicate：
dt_replicate_again = dt_shard.redistribute(
    mesh,
    placements=[Replicate()],
)
在这一步里，DTensor 会自动：
检查源 Placement 是 Shard，目标 Placement 是 Replicate；
在底层插入一次 all-gather，把 4 个 shard 拼成完整 Tensor；
最终在每个 rank 上都得到完整的 global_x。
对用户来说，只需要关心「我现在想要 Replicate 的视图」，不必手动写 all-gather。
3. 在 torchtitan 中：Placement + DeviceMesh 的组合
在 torchtitan 的实际模型并行代码里，会在并行计划中显式使用 Replicate / Shard 来描述张量在 mesh 上的布局，但几乎不用自己手写 all-reduce / all-gather 这类底层通信：
先通过 ParallelDims 构造出 world_mesh，再从中取出需要的子 mesh，例如 tp_mesh = world_mesh["tp"]；
然后在各个模型的 parallelize.py 里，调用
parallelize_module(module, tp_mesh, parallelize_plan=...)；
在 parallelize_plan 里，用 ColwiseParallel、RowwiseParallel、SequenceParallel、PrepareModuleInput 等「并行风格」来组合 Replicate / Shard，声明：  
某个子模块输入/输出在 tp 维上的布局（哪一维 Shard、哪一维 Replicate）；
哪些地方需要在 tp 维插入必要的通信（all-gather / reduce-scatter 等）。

这些「并行风格」本质上就是：在给定的 DeviceMesh 子 mesh 上，为每个模块指定期望的 Placement 和需要的重排布；而像 Partial 这种布局，则更多作为 DTensor 运算的中间状态，由框架在内部管理。
结语
本篇文章只负责打好 DeviceMesh + Placement 这两个基础概念，至于 deepseekv3 / llama4 中 parallelize_deepseekv3、apply_non_moe_tp 等函数里具体是如何用 ColwiseParallel/RowwiseParallel/SequenceParallel 来写这些并行计划，会在后续源码解析文章中单独展开。
# 文档写作问题 History

本文件记录文章修改过程中实际暴露的问题及其原因。同类问题集中在同一类别下追加案例，后续再根据出现频率和影响范围，将稳定、普遍的问题总结到 `AGENTS.md`。

## H001 · 核心机制被应用流程遮蔽

| 日期 | 文档位置 | 具体问题 | 原因 | 处理方式 |
| --- | --- | --- | --- | --- |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 第 4 节 | 先展开 Llama3 如何遍历每层 Config，读完仍不清楚 Module、Config 与 `build()` 的基本关系；补充后又因重复构造流程导致篇幅过长 | 从模型应用代码开始讲解，没有先识别 `Configurable` 协议才是决定行为的核心机制 | 改为先说明 `Config._owner → Config.build() → Module`，再用一小段组合模块代码概括逐层循环 |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 引言 | 文章先介绍布局类型和配置字段，没有在开头说明“显式通信是程序主体，SPMD 类型系统负责检查”这一核心职责分工 | 按代码对象依次展开内容，却没有先给出贯穿全章的设计结论，容易让读者误以为类型传播会自动规划和插入通信 | 在引言明确通信由配置和模型代码事先写入、`parallelize()` 安装边界包装，类型系统只传播和检查元数据；关闭检查不影响通信执行 |
| 2026-09-01 | `docs/torchtitan/06-torch-compile-explicit-collectives.md` 开头与第 6 节 | 文章直接进入编译对象和捕获流程，直到后文才比较 CUDA Graph，读者无法先判断 `torch.compile` 的核心作用，容易把它理解成单纯的 launch 重放 | 按源码调用顺序讲“怎么编译”，没有先回答编译器“改变了什么”，也没有说明降低 host 开销存在程序生成与命令重放两条路径 | 在开头先说明捕获、反向生成、算子融合、代码生成和图级调度，并用“重新组织程序 / 重放程序”区分主职责；第 6 节补充 `reduce-overhead` 等 backend 内部 CUDA Graph 与 TorchTitan 外层 wrapper 可以叠加 |

## H002 · 使用未引入或指代不明的概念

| 日期 | 文档位置 | 具体问题 | 原因 | 处理方式 |
| --- | --- | --- | --- | --- |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 第 5 节 | 写“检查当前 `src → dst` 是否能由现有 helper 表达”，但没有说明 helper 是哪个函数、来自哪里、检查什么 | 用概括性词语替代了具体代码对象，导致列表第一项无法独立理解 | 明确写出 TorchTitan 的 `spmd_validate_redistributions()` 和 `spmd_redistribute_per_axis()`，并区分初始化校验与运行时通信 |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 第 7 节 | 直接讲“类型检查怎样接进训练入口”，且后续只说“按照算子的规则传播”，没有解释类型检查是什么、规则具体指什么 | 先写了 Trainer 接入流程；补充概念时仍使用抽象概括，没有落到算子类别和输入输出映射 | 先区分 SPMD 类型与 Python 类型、`dtype`，再说明线性、多线性、非线性三类 Partial 规则以及矩阵乘的 `S(d)` 传播，最后介绍 Trainer 接入顺序 |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 第 6、8 节 | 使用“带输入输出签名的 `spmd.no_typecheck()`”，但没有解释“签名”是什么，读者无法知道它与普通 `no_typecheck()` 的区别 | 借用了接口术语，却没有先定义 `in_types`、`out_types` 描述的对象以及它们在边界上的作用 | 改称“边界类型声明”，明确它描述参数和返回值的 SPMD 布局，并按入口检查、内部关闭传播、出口标注三个步骤解释 |
| 2026-09-01 | `docs/torchtitan/06-torch-compile-explicit-collectives.md` 初稿 | `Async Tensor Parallel` 和 `CUDA Graph` 先出现在标题或图中，`FX`、guard、PP、FSDP 等术语的首次出现也缺少定义 | 按源码组件组织提纲时直接沿用了代码名称，没有逐项检查读者在当前位置是否已经获得必要上下文 | 在对应标题前先说明概念来源和本节承接关系，并在首次出现处补充 FX 图、guard、PP、AC、FSDP、symmetric memory 与 `MUST_SAVE` 的含义 |
| 2026-09-03 | `docs/components/attention/dsa/index.md` IndexShare 小节 | 用一句“一个 Full 层的索引器同时蒸馏多个目标层的注意力分布”概括训练感知的 IndexCache，没有定义教师分布、学生分布、梯度流向，也没有说明 Shared 层是否仍有独立 Indexer | 直接沿用论文的 multi-layer distillation 术语，没有沿训练和推理生命周期展开其中的数据与参数角色 | 以 `F → S → S → S` 为例定义多层 KL 损失和平均教师，区分梯度等价与损失数值相等，并按 dense warm-up、sparse training、inference 说明教师分布、Full Indexer 与共享 indices 的去向 |
| 2026-09-03 | `docs/components/attention/dsa/index.md` 索引分数、两阶段训练与 IndexShare 小节 | 将 Indexer 概括为“输出索引”，但 KL 公式又直接使用其“输出分布”，没有解释连续分数怎样分别变成训练概率和整数位置；推理代码只返回 `topk_indices`，进一步造成 KL 无从计算的表面矛盾 | 没有区分 Top-k 前的 $\mathbf I_t$、训练辅助分布 $\operatorname{Softmax}(\mathbf I_t)$ 和 Top-k 后的 $\mathcal S_t$，也没有声明示例代码只覆盖推理路径 | 增加三类对象对照表和双分支公式，说明推理只保留整数位置、训练需保留连续分数；补充 Sparse training 在共享 Top-k 内构造教师/学生分布，以及硬 Top-k 对集合外位置没有直接梯度的边界 |
| 2026-09-02 | `docs/training/parallelism/06-ep-performance.md` 开头及通信优化小节 | ETP、D2H、TMA、MNNVL、SM、IBGDA 等缩写直接进入性能分析，读者难以判断它们分别属于专家切分、主机同步、机内搬运还是跨节点通信 | 沿用源码、profiler 和硬件文档中的缩写，没有在文章第一次使用时建立术语到数据路径的映射 | 在性能模型前定义 ETP 与 D2H，并在 HybridEP 数据路径首次出现处展开硬件与通信缩写 |

## H003 · 章节边界割裂同一条调用链

| 日期 | 文档位置 | 具体问题 | 原因 | 处理方式 |
| --- | --- | --- | --- | --- |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 原第 5、6 节 | 第 5 节写 `parallelize()` 安装 forward 包装，第 6 节才解释这个包装，两个同属一个机制的内容被拆成同级章节 | 按“初始化”和“运行时”表面阶段分段，没有优先保持安装动作与被安装逻辑的从属关系 | 将 forward 包装并入第 5 节作为 5.2，后续章节顺延 |

## H004 · 标题承担了不必要的解释

| 日期 | 文档位置 | 具体问题 | 原因 | 处理方式 |
| --- | --- | --- | --- | --- |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 本章标题 | 使用“怎样落实”“检查什么”“如何进入”等设问或解释句式，标题比定位内容所需的信息更长 | 把正文应解释的问题和结论写进标题，没有优先使用稳定、直接的技术对象名称 | 改为 `parallelize() 并行化`、`SPMD 类型检查`、`Config.build()` 等描述性名词短语，并通读本章统一同类标题 |
| 2026-09-02 | `docs/training/parallelism/` 多篇文章的章节标题 | 大量标题使用“为什么……”“到底在做什么”“不只是……”“还能更……”等设问或判断句，部分标题还先于正文引入新实现名词 | 为追求口语化，把过渡、结论和悬念集中塞进标题，削弱了目录的信息定位并造成概念抢跑 | 通读训练系统全部标题，改为通信对象、数据生命周期、实现边界等描述性标题；需要解释的动机移回标题后的正文 |

## H005 · 示例缺少完整的数据生命周期

| 日期 | 文档位置 | 具体问题 | 原因 | 处理方式 |
| --- | --- | --- | --- | --- |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 第 6 节 | 最初只罗列 FFN 的 forward 通信；第一次重写虽补了 local Tensor、shape 和 backward，仍没有展示 Tensor 及其类型元数据怎样从模型输入流向普通算子、collective 和下一层，因而无法为第 7 节的类型推导建立前提 | 把示例当成并行计算流程来写，没有从后续概念所需的前置问题出发组织内容 | 改为先说明类型的三个来源，再沿模型主干和单层 FFN 跟踪每个 local Tensor 的 shape、类型及产生方式，最后说明 `TorchFunctionMode` 怎样把算子输出变成下一步的带类型输入；backward 只保留真实 autograd 数据流 |
| 2026-09-02 | `docs/components/attention/dsa/index.md` IndexPool 小节 | 只写“候选池映射回原始 token 位置”，没有明确选中一个池后是只取代表、再选一个 token，还是把池内四个 token 全部送入主注意力，读者无法跟踪池化表示与原始 MLA KV 的去向 | 把池级粗筛和 token 级精算压缩在一句话里，没有分别说明索引 Key Cache 与主 MLA KV Cache 的数据生命周期 | 增加 `p=4` 流程图，明确池化只发生在索引 Key 路径；完整池入选后展开为四个 token 位置，各自读取原始 MLA KV 并独立计算注意力权重，同时画出未满池 tail 的直通路径 |
| 2026-09-02 | `docs/training/parallelism/06-ep.md` Dispatch / Combine 小节 | 原理篇只画出来源端 permute、两次 all-to-all 与最终 unpermute，读者会以为接收后天然按本地专家连续，与源码篇显式的第二次 permutation 不一致 | 为简化概念图，把接收端 source-major → local-expert-major 的布局变换隐式合并到 all-to-all 结果，却没有标出这个抽象边界 | 在表格、图注和 SVG 中声明概念图折叠了接收端 regroup，并补全标准路径的来源端 permute、接收端 expert regroup 及其 combine 逆过程 |
| 2026-09-04 | `docs/training/parallelism/03-tp.md` Row Parallel 与 Transformer layer 图 | Row Parallel 图中的本地输入画成 `4×4`、本地权重画成 `2×4`，矩阵维度无法相乘，也没有画出两张完整形状的部分和；Transformer layer 图的残差竖线穿过节点和文字 | 绘图时只强调切分方向和通信位置，没有按每个 rank 的实际 GEMM 形状检查矩阵网格，也没有让连线在节点边界终止 | 将本地 GEMM 明确画为 `4×2 · 2×4 → 4×4`，补全两张部分和及其归约结果；拆分残差旁路线，在节点边界使用箭头和分支点连接 |

## H006 · 省略机制生效的范围

| 日期 | 文档位置 | 具体问题 | 原因 | 处理方式 |
| --- | --- | --- | --- | --- |
| 2026-08-31 | `docs/torchtitan/05-sharding-config-spmd-types.md` 第 6 节 | 写“类型链不会断开”，容易理解成模型前向的每个 Tensor 都始终携带 SPMD 元数据 | 只描述了正常传播路径，没有同时说明 `typecheck()` 上下文、`no_typecheck()` 区域、自定义 kernel 和 backward 的边界 | 明确自动传播只覆盖启用检查且可识别的普通 Torch 算子；不检查区域只在边界重新标注，当前 backward 不传播类型 |
| 2026-09-01 | `docs/torchtitan/06-torch-compile-explicit-collectives.md` 初稿 | 编译流程默认写成必然经过 Inductor，CUDA Graph 未说明设备限制，并将 `nn.Module.compile()` 概括成替换 `_call_impl` | 只沿默认运行路径讲主线，没有同时核对可选 backend、平台回退路径和 PyTorch Module 的实际字段变化 | 明确本章采用默认 Inductor backend、其他 backend 的差异和 CUDA Graph 的 NVIDIA CUDA 范围；按源码改为登记 `_compiled_call_impl` |
| 2026-09-02 | `docs/training/parallelism/02-fsdp.md`、`04-pp.md`、`05-cp.md`、`07-nd.md`、`loss-parallel.md` 与 `moe-aux-loss.md` | 将 Megatron DistributedOptimizer 的 reduce-scatter buffer、理想 PP 气泡公式、特定版本的 TE-only CP、`RankGenerator.order` 的逻辑顺序、fused CE 的 BF16/label-smoothing 边界和某个 aux-loss 归约组，分别写成对应算法或物理拓扑的普遍定义 | 文章围绕单条源码路径或简化性能模型展开时，没有持续区分算法语义、框架实现、模型假设、版本约束和 launcher 映射 | 分别补充经典 ZeRO 阶段与实现通信序列、PP 等长 stage/忽略通信前提、CP 后端版本、逻辑进程组与物理放置、fused CE 特性覆盖、标准/global aux 统计范围的边界 |
| 2026-09-03 | `docs/components/attention/dsa/index.md` IndexShare 多层蒸馏小节 | “主注意力在历史位置上的 softmax 权重沿头求平均”没有限定为 Dense warm-up，容易理解成 DSA 在整个训练期都会额外计算完整主注意力 | 介绍教师分布时省略了两阶段训练边界，也没有区分全部 query-key 配对的逻辑计算与完整 attention matrix 的物理物化 | 就地说明 Dense warm-up 计算完整因果主注意力，Sparse training 只在共享 Top-k 上计算主注意力；同时注明分块内核不必在 HBM 中常驻完整分数矩阵，Indexer 的全历史扫描是另一条计算路径 |

## H007 · 局部复杂度被表述为整体复杂度

| 日期 | 文档位置 | 具体问题 | 原因 | 处理方式 |
| --- | --- | --- | --- | --- |
| 2026-09-02 | `docs/components/attention/dsa/index.md` 核心总结、稀疏化示例与总结 | 将“主 Sparse MLA 从 O(L²) 降至 O(Lk)”概括成整个 DSA 的复杂度变化，忽略 Lightning Indexer 在 prefill 阶段仍需进行 O(L²) 的全历史打分；这也使读者无法理解 IndexShare 与 IndexPool 要解决什么问题 | 只统计 Top-k 之后的主注意力计算，没有把索引器的候选生成成本纳入完整数据流 | 将 O(Lk) 明确限定为主 Sparse MLA，补充索引器在 prefill/decode 下的复杂度，并以该遗留瓶颈自然引出跨层索引共享与索引 Key 池化 |
| 2026-09-02 | `docs/training/parallelism/06-ep.md` MoE 容量与计算小节 | 用“参数随专家数增长、计算只随 top-k 增长”概括整层甚至整模型，忽略 router、attention、shared expert、通信和重排成本 | 只统计 routed expert FFN 的参数与 GEMM，没有标明结论所覆盖的子模块 | 将参数—计算解耦限定在路由专家分支，并单列整层仍需承担的稠密模块与系统开销 |

## H008 · 逻辑张量规模与物理开销口径混淆

| 日期 | 文档位置 | 具体问题 | 原因 | 处理方式 |
| --- | --- | --- | --- | --- |
| 2026-09-02 | `docs/training/parallelism/05-cp.md` 注意力显存动机 | 把数学上的 `[S,S]` 注意力分数矩阵直接当作现代注意力内核会在 HBM 常驻的 buffer，并据此解释 CP 的全部显存收益 | 没有区分注意力的逻辑数据依赖、平方计算量与 FlashAttention 等分块内核的真实物化行为 | 在正文、图注和 SVG 中分开说明逻辑矩阵与物理工作区，并把 CP 目标改为分摊序列激活及 query-key 配对计算 |
| 2026-09-02 | `docs/training/parallelism/loss-parallel.md` 通信量小节 | 把三个 `[B,S]` 统计张量称为“几个标量”，并用逻辑张量总大小直接比较 all-gather 与 all-reduce 通信量 | 混用了单 token 元素、逻辑 payload 与集合通信算法的每 rank 链路发送量 | 改称逐 token 统计张量，并在同一理想 ring 模型下分别计算 all-gather 与三次 all-reduce 的每 rank 发送字节数，同时注明融合只减少 collective 次数、不减少元素总量 |

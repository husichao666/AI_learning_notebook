# Loss Parallel 深入解析：torchtitan 与 Megatron-LM 的词表并行交叉熵

> 当词表大到把 logits 全聚合（all-gather）变得不可承受时，如何只用几个标量的通信就算出一模一样的交叉熵？
> 本文从数学原理出发，逐行对照 Megatron-LM 的显式 autograd 实现与 torchtitan 的 DTensor 声明式实现。

## 📖 在线阅读（渲染版）

> GitHub 不会在仓库里直接渲染 HTML（点开 `index.html` 只会看到源码），请用下面任一链接打开渲染后的页面：

- **零配置即点即看** 👉 [htmlpreview 渲染](https://htmlpreview.github.io/?https://github.com/husichao666/AI_learning_notebook/blob/main/infra/parallize/loss_parallel/index.html)
- **GitHub Pages**（需先开启，见文末）👉 <https://husichao666.github.io/AI_learning_notebook/infra/parallize/loss_parallel/>

## 内容速览

1. 为什么需要 Loss Parallel —— TP 下 `lm_head` 列切分导致 logits 沿词表维分片，朴素 all-gather 完整 `[B,S,V]` 的内存/通信代价
2. 数学原理 —— 交叉熵的 log-sum-exp 稳定化分解为三个可分片归约量（全局 max / 目标 logit / Σexp）
   - 含一个 `V=16`、`tp=2` 的单 token 完整流程示意图
3. 通信量分析 —— all-gather vs 三次小标量 all-reduce，省约 4 个数量级
4. Megatron-LM 源码逐行解读 —— `VocabParallelCrossEntropy` 与未融合实现
5. Megatron 的 Fused 通信合并 —— `@jit_fuser` + 把两次 SUM 合并成一次 all-reduce
6. torchtitan 的 DTensor 实现 —— `Shard(-1)` 声明 + `loss_parallel()` 上下文 + `local_map`
7. torchtitan 的 ChunkedCELoss —— 正交的序列维分块，与词表并行叠加
8. 反向传播为何零通信 —— 梯度 = softmax − onehot，两项都在本地
9. 两套实现对比
10. 总结与实践要点

## 参考源码

- `Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py`
- `Megatron-LM/megatron/core/fusions/fused_cross_entropy.py`
- `Megatron-LM/megatron/core/models/common/language_module/language_module.py`
- `torchtitan/torchtitan/components/loss.py`
- `torchtitan/torchtitan/distributed/utils.py`
- `torchtitan/torchtitan/models/common/decoder_sharding.py`

---

### 如何开启 GitHub Pages（一次性）

1. 仓库页面 → **Settings** → 左侧 **Pages**
2. **Build and deployment → Source** 选 **Deploy from a branch**
3. **Branch** 选 `main`，目录选 `/ (root)`，点 **Save**
4. 等 1–2 分钟，本页面即可通过上面的 Pages 链接访问（因为文件名是 `index.html`，直接打开文件夹 URL 就会加载它）

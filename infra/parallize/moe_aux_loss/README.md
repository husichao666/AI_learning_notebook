# MoE 负载均衡 aux_loss 深入解析：mindformers pynative 与 Megatron-LM 逐行对照

> MoE 路由器天生会「坍缩」——把绝大多数 token 都塞给少数几个专家。
> 负载均衡辅助损失（auxiliary load-balancing loss）就是把路由器掰回均匀分配的那只手。
> 本文先把 Switch 公式拆成「3 行就能写完」的核心，再解释为什么 mindformers 的 `router.py`
> 围着这 3 行写了几百行——答案全在 **SP/CP 下的分布式归约**；最后逐项对照 Megatron-LM，确认两者算的是同一个东西。

## 📖 在线阅读（渲染版）

> GitHub 不会在仓库里直接渲染 HTML（点开 `index.html` 只会看到源码），请用下面任一链接打开渲染后的页面：

- **零配置即点即看** 👉 [htmlpreview 渲染](https://htmlpreview.github.io/?https://github.com/husichao666/AI_learning_notebook/blob/main/infra/parallize/moe_aux_loss/index.html)
- **GitHub Pages**（需先开启，见文末）👉 <https://husichao666.github.io/AI_learning_notebook/infra/parallize/moe_aux_loss/>

## 内容速览

1. 为什么需要 aux_loss —— 路由坍缩（router collapse）与 top-k 的不可导性
2. Switch 公式 —— `loss = α·E·Σ f_i·P_i`，为什么 `f_i`（计数）不带梯度、`P_i`（概率）才带梯度
   - 含一个 `E=4, T=6, topk=1` 的手算示意图
3. 核心代码只有 3 行 —— `switch_load_balancing_loss_func`，mindformers 与 Megatron **逐字符相同**
4. 梯度注入的魔法 —— `MoEAuxLossAutoScaler`，为什么不直接 `total = main + aux`
5. 三个变体 —— `aux_loss` / `seq_aux_loss` / `global_aux_loss` 的区别与各自统计口径
6. mindformers 为什么写了这么多代码 —— SP 下用 DTensor `Partial→Replicate` 替代全量 all-gather，CP 在另一张 mesh 上的额外归约
7. Megatron-LM vs mindformers 逐项对照表 —— 算法一致，差异全在框架落地
8. 总结与实践要点

## 结论速记

- **数学完全一致**：`switch_load_balancing_loss_func` 公式逐字符相同，三个变体（aux / seq / global）、梯度注入机制（`MoEAuxLossAutoScaler`）、`calculate_per_token_loss` 修正、按层 tracker 全部对齐。
- **差异只在分布式落地**：Megatron 用显式 `reduce_from_tensor_model_parallel_region(tp_cp_group)` 一次 all-reduce；mindformers 用 DTensor 的 `Partial()→Replicate()` redistribute（`_reduce_token_sum`/`_reduce_seq_sum`），并因为 CP 在路由器 TP mesh 之外，额外用 `reduce_over_aux_loss_groups` 对**计数**做 CP 归约。
- **mindformers「看不懂」的根因**：真正的 aux-loss 数学就 3 行；其余几百行是为了让这 3 行在 SP+CP 下**不做昂贵的全量 logits all-gather**也能算对——只对 `[E]`/`[bsz·E]` 这种小向量做可微归约。

## 参考源码

- `mindformers/mindformers/pynative/transformers/moe/router.py`（`_compute_aux_loss` / `_apply_aux_loss` / `_apply_seq_aux_loss` / `_apply_global_aux_loss` / `_reduce_token_sum` / `_reduce_seq_sum`）
- `mindformers/mindformers/pynative/transformers/moe/moe_utils.py`（`switch_load_balancing_loss_func` / `compute_routing_scores_for_aux_loss` / `MoEAuxLossAutoScaler` / `get_tokens_per_expert_and_token_count`）
- `Megatron-LM/megatron/core/transformer/moe/router.py`（`_apply_aux_loss` / `_apply_seq_aux_loss` / `_apply_global_aux_loss` / `attach_and_log_load_balancing_loss`）
- `Megatron-LM/megatron/core/transformer/moe/moe_utils.py`（`switch_load_balancing_loss_func` / `get_tokens_per_expert_and_token_count` / `MoEAuxLossAutoScaler`）

---

### 如何开启 GitHub Pages（一次性）

1. 仓库页面 → **Settings** → 左侧 **Pages**
2. **Build and deployment → Source** 选 **Deploy from a branch**
3. **Branch** 选 `main`，目录选 `/ (root)`，点 **Save**
4. 等 1–2 分钟，本页面即可通过上面的 Pages 链接访问（因为文件名是 `index.html`，直接打开文件夹 URL 就会加载它）

# 从 RoPE 到多模态 M-RoPE (Interleaved)

在阅读现代大模型（如 LLaMA、Qwen 等）的源码时，**RoPE（旋转位置编码）** 几乎是一个绕不开的核心组件。而随着多模态大模型（如 Qwen3.5）的爆发，RoPE 也进化出了更复杂的三维形态——**M-RoPE (Multimodal RoPE)**，甚至引入了**交错(Interleaved)**机制。

这篇博客将像剥洋葱一样，从最基础的标准 RoPE 开始，一步步带你走向多模态的 M-RoPE，最后彻底拆解 Qwen3.5 中精妙的 `M-RoPE Interleaved` 实现。即使你是刚接触大模型源码的新手，也能轻松看懂。

---

## 第一部分：标准 RoPE (Rotary Position Embedding) 基础

本文不详细解释基础RoPE的公式推导，有兴趣的读者可以参考[RoPE公式推导](https://zhuanlan.zhihu.com/p/647109286)。

### 1. 为什么需要 RoPE？
在 Transformer 架构中，Self-Attention 本身是**没有位置概念**的。打乱输入 Token 的顺序，输出的结果是一样的（排列不变性）。为了让模型知道 "A 在 B 前面"，我们需要注入位置信息。

传统的绝对位置编码（如正弦波编码、可学习位置编码）是直接加在输入 Token 上的。但语言的本质往往更依赖**相对位置**（比如“主语在谓语前面 1 个位置”），而不是绝对位置（“主语在第 5 个位置”）。

**RoPE 的核心思想是：通过在绝对位置上进行旋转操作，使得 Token 之间的内积（Attention Score）自然包含相对位置信息。**

### 2. 数学直觉：用旋转代表位置
想象一个二维平面上的向量 $q$。如果它在句子中的位置是 $m$，我们就把它旋转 $m \times \theta$ 的角度。
利用复数乘法或旋转矩阵，我们可以表示为：

$$ f(q, m) = R_m \cdot q $$

其中 $R_m$ 是旋转 $m\theta$ 角度的矩阵。

当计算 Query 和 Key 的注意力得分（内积）时，奇妙的事情发生了：

$$ \text{Score} = f(q, m) \cdot f(k, n) = (R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k $$

你看！绝对位置 $m$ 和 $n$ 消失了，留下的是它们的差值 $n-m$。**我们在绝对位置上做了旋转，却在点乘时完美得到了相对位置！**

### 3. 从输入到旋转：张量形状 (Shape) 推导

为了把数学直觉落地到代码，我们来看一个具体的例子，看看 `position_ids` 和 `freqs` 是怎么计算出来的。

假设我们输入了一句话：“我爱自然语言处理”。
*   `batch_size = 1`
*   `seq_len = 8` (假设分词后有 8 个 Token)
*   `head_dim = 128` (每个注意力头的特征维度)

#### 步骤 1：生成 `position_ids`
对于纯文本，`position_ids` 就是一个简单的一维递增序列。
它的形状是 `(batch_size, seq_len)`，即 `(1, 8)`。
内容是：`[[0, 1, 2, 3, 4, 5, 6, 7]]`。

#### 步骤 2：生成基础频率 `inv_freq`
RoPE 会预先计算好一组固定的频率。因为我们要把 `head_dim` (128) 两两分组，所以频率的数量是 `head_dim // 2 = 64`。
`inv_freq` 的形状是 `(head_dim // 2,)`，即 `(64,)`。
内容类似于：`[1.0, 0.8, 0.6, ..., 0.0001]`（呈指数衰减）。

#### 步骤 3：计算每个 Token 的具体旋转角度 `freqs`
现在，我们要把“位置”和“基础频率”乘起来。
*   将 `position_ids` 扩展为列向量：形状变成 `(1, 8, 1)`。
*   将 `inv_freq` 扩展为行向量：形状变成 `(1, 1, 64)`。
*   **矩阵相乘**：`(1, 8, 1) @ (1, 1, 64) -> (1, 8, 64)`。

这个算出来的 `(1, 8, 64)` 张量就是 `freqs`。
**它代表什么？**
它代表了这 8 个 Token，在 64 组特征维度上的**绝对旋转角度**。
比如 `freqs[0, 3, 5]`，代表第 4 个 Token（位置为 3）的第 6 组特征（索引为 5）需要旋转的角度值（即 $3 \times \theta_5$）。

#### 步骤 4：生成 `cos` 和 `sin` 并应用到 Q/K
有了角度 `freqs`，我们就可以计算余弦和正弦值。
为了和 `head_dim` (128) 对齐，我们会把 `freqs` 复制拼接一份：
`emb = torch.cat((freqs, freqs), dim=-1)` -> 形状变成 `(1, 8, 128)`。

然后计算 `cos = emb.cos()` 和 `sin = emb.sin()`，它们的形状都是 `(1, 8, 128)`。
最后，利用 PyTorch 的广播机制（增加一个 `num_heads` 维度），将它们与形状为 `(batch_size, num_heads, seq_len, head_dim)` 的 Query 和 Key 张量逐元素相乘，完成旋转操作。

---



## 第二部分：走向多模态：M-RoPE (Multimodal RoPE)

### 1. 一维序列的局限性
在普通的纯文本 LLM 中，`position_ids` 就是一个简单的一维递增序列：`[0, 1, 2, 3, ...]`。

但是，当我们引入**图片**和**视频**时，一维的 ID 就不够用了：
*   **图片是二维的**：一个图像 Patch 在画面中不仅有“第几行（高度 H）”，还有“第几列（宽度 W）”。
*   **视频是三维的**：除了高度和宽度，还有“第几帧（时间 T）”。

如果强行把图片展平成一维序列给模型看，模型就会丢失空间上的上下左右关系。

### 2. M-RoPE 的解法：3D 位置 ID
为了解决这个问题，M-RoPE 将 `position_ids` 从一维扩展成了 **3个维度**，形状变为了 `(3, batch_size, seq_len)`：
*   **`position_ids[0]`**：代表**时间维度 (T)**。
*   **`position_ids[1]`**：代表**高度维度 (H)**。
*   **`position_ids[2]`**：代表**宽度维度 (W)**。

### 3. 图文混合时的 `position_ids` 示例
假设我们输入：`3个文本 Token` + `1张图片 (切分成 2x2=4 个 Patch)` + `2个文本 Token`。
总长度 `seq_len = 9`。它的 `position_ids` 长这样：

```python
[
  [0, 1, 2,  3, 3, 3, 3,  5, 6],  # T 维度：图片没有时间流逝，所以 T 停滞在 3
  [0, 1, 2,  3, 3, 4, 4,  5, 6],  # H 维度：图片的第1行是3，第2行是4
  [0, 1, 2,  3, 4, 3, 4,  5, 6]   # W 维度：图片的第1列是3，第2列是4
]
```
**注意看文本部分（前3个和后2个）**：对于纯文本，它的 T、H、W 三个维度的 ID 是**完全一样**的！这个巧妙的设计我们后面会讲到它的妙用。

### 4. 使用三维 `position_ids` 计算 `freqs`

有了这个 `(3, batch_size, seq_len)` 的 `position_ids` 后，我们需要像第一部分那样，把它和基础频率 `inv_freq` 相乘，得到每个维度的旋转角度 `freqs`。

在 Qwen3.5 中，计算过程如下：
```python
# 1. inv_freq 原本是 (head_dim // 2,)，扩展为 (3, batch_size, head_dim // 2, 1)
# 这里的 3 代表我们要为 T、H、W 分别准备一份基础频率
inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)

# 2. position_ids 扩展为 (3, batch_size, 1, seq_len)
position_ids_expanded = position_ids[:, :, None, :].float()

# 3. 矩阵乘法：(head_dim // 2, 1) @ (1, seq_len) -> (head_dim // 2, seq_len)
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float())

# 4. 转置后，freqs 的最终形状为：(3, batch_size, seq_len, head_dim // 2)
freqs = freqs.transpose(2, 3)
```

**此时的 `freqs` 是一个由 3 层组成的张量（对应最外层的 `3`）：**
*   `freqs[0]`：全是由**时间(T)**位置算出来的角度。
*   `freqs[1]`：全是由**高度(H)**位置算出来的角度。
*   `freqs[2]`：全是由**宽度(W)**位置算出来的角度。

### 5. 普通 M-RoPE 的做法：分块拼接 (Chunked)

在引入交错（Interleaved）之前，最朴素的 M-RoPE 是如何把这 3 组频率塞进一个 Token 的特征里的呢？
答案是**分块拼接（Chunked）**。

假设 `head_dim // 2 = 64`，普通 M-RoPE 会把这 64 个特征维度直接切成三块：
*   前 21 个维度：直接使用 `freqs[0]`（时间 T 的频率）
*   中间 21 个维度：直接使用 `freqs[1]`（高度 H 的频率）
*   最后 22 个维度：直接使用 `freqs[2]`（宽度 W 的频率）

### 补充：Partial RoPE (部分旋转)

值得注意的是，在计算 `inv_freq` 时，模型配置中可能包含一个 `partial_rotary_factor` 参数（默认为 1.0）。
这意味着，模型可能**不会对整个 `head_dim` 进行旋转**，而是只对前一部分特征进行旋转。
例如，如果 `head_dim = 128`，`partial_rotary_factor = 0.5`，那么 `dim` 就只有 64。
此时生成的 `cos` 和 `sin` 的最后一维大小只有 64。

在最终应用到 Q 和 K 时（`apply_rotary_pos_emb` 函数中）：
```python
# 获取需要旋转的维度大小 (比如 64)
rotary_dim = cos.shape[-1]

# 将 Q 和 K 切分为“需要旋转的部分”和“不需要旋转的部分”
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

# 只对前半部分应用旋转
q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

# 最后再把不需要旋转的部分拼回去
q_embed = torch.cat([q_embed, q_pass], dim=-1)
k_embed = torch.cat([k_embed, k_pass], dim=-1)
```

这种设计允许模型在保留绝对特征（不旋转部分）的同时，利用部分维度（旋转部分）来感知相对位置。

## 第三部分：Qwen3.5 的魔法：M-RoPE Interleaved

为了解决普通 M-RoPE 分块拼接带来的特征割裂问题，Qwen3.5 引入了 **交错（Interleaved）** 机制。

### 1. 交错 (Interleaved) 与普通 M-RoPE 的区别

*   **普通 M-RoPE (Chunked)**：`[T, T, T, ..., H, H, H, ..., W, W, W]`（像三块不同颜色的布缝在一起）
*   **M-RoPE Interleaved**：`[T, H, W, T, H, W, T, H, W, ...]`（像三根不同颜色的线均匀地编织在一起）

这种交错编织的方式，使得任何一个局部的特征子空间，都同时包含了时间、高度和宽度的位置信息，极大地促进了多模态特征的深层融合。

### 2. 源码解析：交错是怎么发生的？
在 Qwen3.5 的源码中，我们已经在上一步得到了 3 维的 `freqs`，形状为 `(3, batch_size, seq_len, head_dim // 2)`。
然后调用交错函数：

```python
def apply_interleaved_mrope(self, freqs, mrope_section):
    # freqs_t 初始作为画板，提取出所有的 T 频率
    # 注意：这一步直接去掉了最外层的 3 维！形状变成了 (bs, seq_len, head_dim // 2)
    freqs_t = freqs[0]  
    
    # 开始遍历：H (dim=1, offset=1) 和 W (dim=2, offset=2)
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        # 按步长 3 切片替换，比如 offset=1，步长为3，就是索引 [1, 4, 7, 10...]
        idx = slice(offset, length, 3)
        # 把 freqs[1] (H的频率) 或 freqs[2] (W的频率) 塞进 freqs_t 的特定索引里
        freqs_t[..., idx] = freqs[dim, ..., idx]
        
    return freqs_t
```

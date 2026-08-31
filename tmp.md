# 使用ColwiseParallel切分你的模型
引言
通过前面3章的学习，我们知道了：
1. DTensor包含全局Tensor信息；
2. 这些信息通过DeviceMesh和Placement进行表示：DeviceMesh确定了Device的拓扑结构，Placement则表示Tensor在该拓扑上的分布；
3. 2个DTensor的计算，如何在多卡场景运行。

已经有了地基，本章节，将介绍如何基于上述的特性，进行常见的模型切分，如 TP，SP，EP。


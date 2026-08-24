---
title: 训练性能采集与分析
description: 训练性能数据的精简采集、完整采集与正确性分析
type: engineering-note
status: growing
level: intermediate
updated: 2026-08-24
tags:
  - profiling
  - training-performance
  - optimization
---

# 训练性能优化

本章主要介绍大模型训练性能优化思路

## 性能数据采集

MindFormers提供了profile配置，通过yaml配置后，就可以自动进行性能数据采集。

```yaml
profiler:
  enable_profiling: False
  output_path: "./profile"
  start_step: 6
  end_step: 6
  profiler_level: 0
  profiler_rank: [0, 64]
  profile_memory: False
  profile_cpu: True
  with_stack: True
```

### 精简采集

由与采集性能数据占用资源，会导致采集的step变慢，数据失真。因此需要采集最少数据，减少失真

```yaml
profiler:
  enable_profiling: True
  output_path: "./profile"
  start_step: 6
  end_step: 6
  profiler_level: 0
  profiler_rank: [0, 64]
  profile_memory: False
  profile_cpu: False
  with_stack: False
```

### 完整采集

精准采集时，缺失很多信息，不好进行具体问题分析，因此需要对照完整数据进行分析。下面是完整采集配置

```yaml
profiler:
  enable_profiling: True
  output_path: "./profile"
  start_step: 6
  end_step: 6
  profiler_level: 1
  profiler_rank: [0, 64]
  profile_memory: True
  profile_cpu: True
  with_stack: True
```

### 采集结果

开启profile，运行训练后，在 `output_path` 位置，将会有profile结果。

使用 MindStudio Insight 软件，把 `output_path` 文件拖入其中，可以看到数据流；

## 性能数据分析

性能数据分析要从多个方向进行分析：

1. 首先，需要保证并行切分的正确性

#### 正确性分析

首先，需要对最直观的正确性进行分析，主要涉及多卡并行切分是否符合预期，通信插入是否符合预期
1. fsdp
2. tp, sp, ep

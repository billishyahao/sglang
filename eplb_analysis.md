# EPLB (Expert Parallelism Load Balancing) 实现分析

> 基于 sglang-sa-tbo 代码库分析

## 1. 解决什么问题

在 MoE (Mixture-of-Experts) 模型中（如 DeepSeek V3），每个 token 只激活少量 expert。在 Expert Parallelism 下，不同 expert 分布在不同 GPU 上。**问题是：某些热门 expert 会收到远超平均的 token，导致 GPU 间负载不均衡**，拖慢整体推理速度。

EPLB 通过**动态复制热门 expert 到多张 GPU** 来平衡负载。

## 2. 核心概念

| 概念 | 说明 |
|------|------|
| **Logical Expert** | 模型原始定义的 expert（如 256 个） |
| **Physical Expert** | 实际运行时的 expert 槽位 = logical + redundant（如 256+4=260 个） |
| **Redundant Expert** | 额外增加的物理槽位，用于复制热门 expert |
| **physical_to_logical_map** | `[layers, num_physical_experts]`，每个物理槽位对应哪个逻辑 expert |
| **logical_to_all_physical_map** | `[layers, num_logical_experts, X]`，每个逻辑 expert 有哪些物理副本 |

## 3. 整体架构（5 个模块）

```
EPLBManager (调度中心)
    │
    ├── ExpertDistributionRecorder (统计收集)
    │       收集每个 expert 被路由了多少 token
    │
    ├── EPLB Algorithms (重平衡算法)
    │       根据统计计算最优的 physical→logical 映射
    │
    ├── ExpertLocationMetadata (映射元数据)
    │       存储当前的映射关系
    │
    ├── ExpertLocationUpdater (权重搬运)
    │       通过 P2P 通信在 GPU 间搬运 expert 权重
    │
    └── ExpertLocationDispatch (路由转换)
            推理时将 logical expert id → physical expert id
```

## 4. 各模块详解

### 4.1 EPLBManager — 调度中心

**文件**: `python/sglang/srt/eplb/eplb_manager.py`

核心是一个**生成器协程**，嵌入到推理主循环中：

```python
def _entrypoint(self):
    while True:
        # 等待 N 次 forward pass
        for _ in range(self._rebalance_num_iterations):
            yield
        # 执行重平衡
        yield from self.rebalance()
```

- 每次 forward pass 结束调 `on_forward_pass_end()` → `next(generator)`
- 累积够 `eplb_rebalance_num_iterations` 次后触发 `rebalance()`
- **利用率检查**：如果 `average_utilization_rate > threshold`，跳过本轮重平衡（已经很均衡了）
- **分层更新**：支持 `eplb_rebalance_layers_per_chunk`，每次只更新一部分层，中间 `yield` 让出执行权，避免阻塞推理太久

### 4.2 ExpertDistributionRecorder — 统计收集

**文件**: `python/sglang/srt/eplb/expert_distribution.py`

这是一个复杂的统计框架，支持多种采集模式：

- **`stat` 模式**：用 `_SelectExpertsSinglePassGatherer`，在 `on_select_experts` 时通过 `scatter_add_` 统计每个物理 expert 的 token 数
- **`stat_approx` 模式**：用 DeepEP normal 模式下的近似统计
- **`per_token` 模式**：详细记录每个 token 路由到哪个 expert

**数据流**：

1. `SinglePassGatherer` 在一次 forward pass 中收集 per-layer 的 `global_physical_count`
2. `Accumulator` 将多次 pass 的数据聚合。`_StatAccumulator` 用 `_CircularBuffer` 保存最近 N 步的统计
3. `dump_record()` 时，将 physical count 转换为 `logical_count`（通过 `physical_to_logical_map` 做 `scatter_add_`），并通过 `all_reduce` 汇总所有 rank 的数据

**利用率指标**：

```python
utilization_rate = avg_gpu_count / max_gpu_count
```

值越接近 1 表示越均衡。

### 4.3 EPLB Algorithms — 重平衡算法

提供 3 套算法：

#### A) `deepseek` 算法

**文件**: `python/sglang/srt/eplb/eplb_algorithms/deepseek.py`

源自 DeepSeek 官方 EPLB 仓库。核心思路是一个三步过程：

1. **`balanced_packing`**：将 expert group 均匀分配到节点，使节点间负载均衡（贪心装箱）
2. **`replicate_experts`**：将冗余槽位分配给负载最重的 expert。每次选 `weight/replica_count` 最大的 expert 复制
3. **再次 `balanced_packing`**：将物理 expert 均匀分配到 GPU

支持 **hierarchical** 模式（先按节点分组，再节点内分 GPU）和 **global** 模式（视为一个大组）。

#### B) `deepseek_vec` 算法

**文件**: `python/sglang/srt/eplb/eplb_algorithms/deepseek_vec.py`

向量化版本，不是 sum 所有 step 的统计，而是保留 per-step 的 `tokens_per_expert` 信息：

- `make_redundant_experts_chunkwise`：分 chunk 做冗余专家分配，在每个 chunk 内独立决策
- 排序后蛇形排列（`1::2` flip）保证 GPU 间均衡

#### C) `elasticity_aware` 算法

**文件**: `python/sglang/srt/eplb/eplb_algorithms/elasticity_aware.py`

感知 GPU 故障的弹性版本。当有 GPU 掉线时（`active_ranks` 不全为 1），减少参与的 GPU 数重新计算映射，并在结果中插入空槽位给掉线 GPU。

### 4.4 ExpertLocationMetadata — 映射元数据

**文件**: `python/sglang/srt/eplb/expert_location.py`

三种构造方式：

- `init_trivial`：identity 映射（物理 expert i = 逻辑 expert i）
- `init_by_mapping`：给定 `physical_to_logical_map`，反向推出 `logical_to_all_physical_map`
- `init_by_eplb`：给定 `logical_count` 统计，调用算法计算映射

**`_find_nearest_expert`** 是关键优化：给定一个逻辑 expert 的所有物理副本，优先选择：

1. 同 GPU 上的副本
2. 同节点（NVLink）上的副本
3. 跨节点的副本

这最小化了通信开销。

**`logical_to_rank_dispatch_physical_map`**（static 模式）：预计算每个 rank 对每个逻辑 expert 应该路由到哪个物理 expert，使用 `_fair_choices` 做均匀分配。

### 4.5 ExpertLocationUpdater — 权重搬运

**文件**: `python/sglang/srt/eplb/expert_location_updater.py`

这是最复杂的部分。当映射改变后，需要在 GPU 间搬运 expert 权重。

**`update_expert_weights_single_layer`** 的 5 种 case（从快到慢）：

| Case | 场景 | 处理方式 |
|------|------|----------|
| 1. unchanged | 该槽位的逻辑 expert 没变 | 跳过 |
| 2. same-gpu | 新逻辑 expert 在本 GPU 已有 | 本地 copy |
| 3. free-rider | 同 GPU 上另一个新槽位已经接收了这个 expert | 复用 temp buffer |
| 4. same-node | 需要从同节点其他 GPU 搬运 | P2P isend/irecv |
| 5. cross-node | 需要跨节点搬运 | P2P isend/irecv |

**通信流程**：

1. 构建所有 `irecv` 操作（接收端）
2. 构建所有 `isend` 操作（发送端）
3. 按 `logical_expert_id` 排序后用 `batch_isend_irecv` 批量执行
4. 将 temp buffer 复制到实际权重 tensor

**Canary 验证**：调试模式下，额外传输 `physical_to_logical_map` 的一小段数据，搬运完成后验证数据是否正确。

**弹性 EP 支持**：通过 `_filter_p2p_ops` 过滤掉不活跃 GPU 的 P2P 操作，记录缺失的 expert。

### 4.6 ExpertLocationDispatch — 路由转换

**文件**: `python/sglang/srt/eplb/expert_location_dispatch.py`

推理时把 router 输出的逻辑 expert id 转换为物理 expert id：

- **`static` 模式**：直接查表 `partial_logical_to_rank_dispatch_physical_map[topk_ids]`，O(1)
- **`dynamic`/`fake` 模式**：从所有物理副本中随机选择一个，通过 `randint % num_valid` 实现

## 5. 数据流总结

```
forward pass
    → router 选出 topk logical expert ids
    → ExpertLocationDispatch 转换为 physical expert ids
    → DeepEP 执行 all-to-all dispatch
    → ExpertDistributionRecorder 记录统计
    → 每 N 步：
        → dump_record() 得到 logical_count
        → all_reduce 汇总所有 rank
        → EPLB algorithm 算出新 physical_to_logical_map
        → ExpertLocationUpdater 通过 P2P 搬运权重
        → 更新 ExpertLocationMetadata
```

## 6. 配置参数

| 参数 | 含义 |
|------|------|
| `--enable-eplb` | 开启动态负载均衡 |
| `--ep-num-redundant-experts` | 冗余 expert 数量（如 4） |
| `--eplb-rebalance-num-iterations` | 每多少次 forward 重平衡一次 |
| `--expert-distribution-recorder-buffer-size` | 统计环形缓冲区大小 |
| `--eplb-rebalance-layers-per-chunk` | 每次更新多少层（分批减少中断） |
| `--ep-dispatch-algorithm` | `static`（查表）或 `dynamic`（随机） |
| `--eplb-algorithm` | 重平衡算法选择（`auto`/`deepseek`/`deepseek_vec`/`elasticity_aware` 及其 `_hierarchical` 变体） |
| `--init-expert-location` | 静态初始化映射（从 `.pt`/`.json` 文件加载） |
| `--expert-distribution-recorder-mode` | 统计模式：`stat`/`stat_approx`/`per_token`/`per_pass` |
| `--eplb-min-rebalancing-utilization-threshold` | 利用率高于此值时跳过重平衡 |

# Paged LoRA PR — Code Review 问题汇总

> 审查时间: 2026-07-14
> 审查范围: `feat/paged-lora-pr` vs `upstream/main`，29 文件，+3738 / -165 行

---

## 🔴 Critical Bug（4 个）

### Bug 1 — `init_cuda_graph_batch_info` 破坏非 csgmv 后端

| 项目 | 内容 |
|------|------|
| **文件** | `python/sglang/srt/lora/lora_manager.py` |
| **行号** | 原 138-142 |
| **触发条件** | `--lora-backend triton\|ascend\|torch_native` + CUDA graph 开启 |
| **触发场景** | `use_paged_pool=False`（非 paged 模式），但 `page_rank_size` 和 `max_lora_rank` 被无条件当 keyword argument 传给所有后端。triton/ascend/torch 后端 override `init_cuda_graph_batch_info` 时签名只有 `(self, max_bs_in_cuda_graph, num_tokens_per_bs)` 两个参数，没有 `**kwargs` |
| **结果** | `TypeError: got unexpected keyword argument 'page_rank_size'`，CUDA graph 初始化 crash |
| **状态** | ✅ **已修复** — 只在 `use_paged_pool=True` 时传 paged 参数，else 分支走原两参数调用 |

### Bug 2 — `logic_idx` 未绑定导致 UnboundLocalError

| 项目 | 内容 |
|------|------|
| **文件** | `python/sglang/srt/lora/paged_mem_pool.py` |
| **行号** | 392-406 |
| **触发条件** | `phys_page_to_uid` 记录了某个物理页属于 uid X，但 X 不在 `page_table` 中（状态不一致） |
| **触发场景** | 正常流程不会触发（`allocate_pages`/`free_pages`/`evict_pages` 同时维护两者一致性）。但若未来代码修改引入不一致，或并发场景下交错执行，会 crash |
| **结果** | `NameError: name 'logic_idx' is not defined`，scheduler crash |
| **状态** | ✅ **已修复** — 在 `if` 块前初始化 `logic_idx = -1` |

### Bug 3 — LoRA B 权重双重 scaling

| 项目 | 内容 |
|------|------|
| **文件** | `python/sglang/srt/lora/paged_mem_pool.py` (653-654) + `chunked_sgmv_expand_paged.py` (174) |
| **触发条件** | 任何 `lora_alpha != lora_rank` 的 adapter |
| **触发场景** | `_scatter_b_weight_to_pages` 在加载 B 权重到 page 时做了 `dst.mul_(scaling)`，而 paged expand kernel 又做 `partial_sum *= scaling`。Flat 路径只在 kernel 做一次 scaling（正确），paged 路径做了两次 |
| **结果** | LoRA delta 被放大 scaling² 倍。举例：`r=16, alpha=32` → scaling=2 → delta 实际上是 **4 倍**，generation 质量严重下降 |
| **状态** | ✅ **已修复** — 注释掉 `_scatter_b_weight_to_pages` 中的 `dst.mul_(scaling)`，scaling 统一只在 kernel 中生效 |

### Bug 4 — TP 切分 offset 重叠

| 项目 | 内容 |
|------|------|
| **文件** | `python/sglang/srt/lora/paged_mem_pool.py` |
| **行号** | 604, 648-649 |
| **触发条件** | `hidden_dim % tp_size != 0`（模型 hidden dim 不被 tp 整除） |
| **触发场景** | `_init_pages` 用 `ceil(dim / tp)` 分配 page buffer 宽度，但 `_scatter_a/b_weight_to_pages` 用 `floor(dim / tp)` 计算每个 rank 的 offset 起点。两个 rank 的 slice 边界会产生重叠列。举例：`input_dim=10, tp=4` → `page_hidden=ceil(10/4)=3`，rank0 offset=`0*2=0` slice `[0:3]`，rank1 offset=`1*2=2` slice `[2:5]`，column 2 重叠 |
| **结果** | all-reduce 后重叠元素被 double count，输出错误 |
| **状态** | ❌ 待修复 — offset 应使用 `self.tp_rank * page_hidden` 并 clamp 末 rank |

---

## 🟡 功能回归（1 个）

### Bug 5 — `enable_metrics=False` 不再抑制 time stats

| 项目 | 内容 |
|------|------|
| **文件** | `req_time_stats.py:579` + `tokenizer_manager.py:1723` |
| **触发条件** | `--enable-metrics=false` |
| **触发场景** | PR 删除了 `if not self.enable_metrics: return {}` guard，time stats 总是序列化并通过 IPC 发送 |
| **结果** | API 响应始终包含 TTFT/ITL 等时间字段，增加 IPC 开销，改变 API 约定 |
| **状态** | ❌ 待修复 |

---

## 🟠 测试基础设施问题（6 个）

| # | 文件 | 问题 | 状态 |
|---|------|------|------|
| 6 | `test_paged_mem_pool.py:90` | `page_access_times` 初始化为 `{}` (dict)，生产代码是 `[0.0]*total_pages` (list)。未 `mark_adapter_pages_accessed` 时调 `evict_pages` 会 KeyError | ❌ |
| 7 | 4 个测试文件 | `_patch_kernels_revision()` 用 `except Exception: pass` 包裹所有 patch 逻辑，失败时静默无效 | ❌ |
| 8 | 所有新测试文件 | 测试类用 `unittest.TestCase` 而非项目要求的 `CustomTestCase`（缺少 CI retry 逻辑） | ❌ |
| 9 | 所有新测试文件 | `register_cpu_ci` 被 `try/except ImportError: pass` 包裹，CI 注册可能静默失败 | ❌ |
| 10 | 4 个测试文件 | `_patch_kernels_revision()` ~50 行完全相同，应提取到共享 util | ❌ |
| 11 | `test_paged_kernel_correctness.py:4247` | float32 `== 0` 比较，应使用 `torch.allclose` | ❌ |

---

## 🟢 代码质量问题（10 个）

| # | 文件 | 问题 | 状态 |
|---|------|------|------|
| 12 | `layers.py:603` | B_buffer 维度校验在 paged 模式跳过，仅 debug flag 生效 | ❌ |
| 13 | `chunked_sgmv_expand_paged.py:26` + `shrink_paged.py:141` | `_next_power_of_2` 重复定义，已有 `sglang.srt.utils.common.next_power_of_2` | ❌ |
| 14 | `paged_mem_pool.py:215,232,301` | `ceil_div` 内联 ×3，已有 `sglang.srt.utils.common.ceil_div` | ❌ |
| 15 | `paged_mem_pool.py:145` | `_get_num_experts` 定义但从未调用（从 flat pool 复制来的死代码） | ❌ |
| 16 | `paged_mem_pool.py:877-878, 898-899` | `t0` / `bytes_before` 赋值后从未读取（残留脚手架） | ❌ |
| 17 | `lora_manager.py:679` | `from collections import Counter` 在方法体内延迟 import | ❌ |
| 18 | `scheduler.py` | `_try_page_in_missing` 调用 `is_complete`/`get_missing_pages`，然后 `ensure_adapter_ready` 内部又调用一遍 | ❌ |
| 19 | `scheduler.py:3105` | `lora_base_priority` 每轮 `sorted(waiting_queue)` O(n log n)，可用两遍遍历 O(n) | ❌ |
| 20 | `chunked_sgmv_expand_paged.py:2570` + `shrink_paged.py:2891` | `SGLANG_PAGED_DEBUG` 时 `.any().item()` 强制 GPU-CPU sync | ❌ |
| 21 | `chunked_backend.py:128` | `_paged_params` helper 提取 3 个属性只用一行，过度封装 | ❌ |

---

## 修复状态汇总

| 状态 | 数量 |
|------|------|
| ✅ 已修复 | 3 (Bug 1, 2, 3) |
| ❌ 待修复 | 18 (Bug 4, Bug 5, #6-#21) |

### 建议优先修复

1. **Bug 4** — TP 切分重叠，影响所有 TP>1 + dim 不被整除的模型
2. **Bug 5** — `enable_metrics` 功能回归
3. **#8** — `CustomTestCase` 规范化（CI 要求）
4. **#15** — 删除 `_get_num_experts` 死代码
5. **#16** — 删除 `t0` / `bytes_before` 死代码

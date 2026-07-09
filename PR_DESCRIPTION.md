## Motivation

Paged LoRA memory pool: page-granularity weight storage with `page_table` indirection. Instead of allocating `max_rank` per adapter (flat mode, where small-rank adapters waste 87.5%+ of weight memory), paged mode allocates fixed-size pages (`page_rank_size=8` ranks per page) and maps them via a page table. This enables higher adapter density, better GPU cache utilization, and lower memory waste.

## Modifications

### Core: paged LoRA memory pool
- `paged_mem_pool.py`: Page pool with allocate/free/evict/page_in/ensure_adapter_ready. O(1) reverse map (physical page → owning adapter) for fast eviction.
- `chunked_sgmv_shrink_paged.py`: Paged shrink Triton kernel — reads A weights from page storage via page_table, 2D grid `(max_pages, segments)`.
- `chunked_sgmv_expand_paged.py`: Paged expand Triton kernel — reads B weights from page storage, constexpr loop over pages.
- `lora_manager.py`: Paged `prepare_lora_batch` path — builds page_table tensor, copies into CUDA graph pre-allocated tensor. Added `_page_table_cache` with generation-aware invalidation.
- `chunked_backend.py`: Paged kernel dispatch methods (`run_lora_a_sgemm_paged`, `run_lora_b_sgemm_paged`, `run_qkv_lora_paged`, `run_gate_up_lora_paged`).
- `layers.py`: `_is_paged_mode()` forward branches for qkv/gate_up/o_proj/embedding.
- `scheduler.py`: `lora_base_priority` — base-model requests prioritized in prefill admission order.
- `server_args.py`: `--lora-page-rank-size`, `--lora-pages`, `--lora-base-priority`, `--max-lora-chunk-size`.

### Optimization
- `build_page_table_tensor`: CPU build + single H2D copy (was: 48 individual GPU element assignments = 48 kernel launches).
- `_page_table_cache`: `tuple(active_uids)` + `page_generation` key, bounded to single entry via `clear()` on miss.
- Shrink kernel grid: uses `batch_info.bs` in CUDA graph mode (matches expand kernel pattern).

### Tests (38 total)
- `test_paged_mem_pool.py` (16 CPU unit tests): build_page_table_tensor correctness, page_generation counter, page lifecycle.
- `test_lora_paged_manager.py` (12 CPU unit tests): cache key ordering (P1 bug fix: frozenset→tuple), cache hit/miss, bounded clear, getattr fallback.
- `test_paged_kernel_correctness.py` (6 GPU tests): paged vs flat kernel tensor comparison with `torch.allclose`.
- `test_lora_paged_e2e.py` (4 GPU E2E tests): server launch + LoRA request + response verification.

## Accuracy Tests

Kernel-level correctness: paged kernel output matches flat kernel output **bit-exact** (max_diff = 0.00e+00):

| Test | max_diff | Result |
|---|---|---|
| Shrink (rank=8, 1 page) | 0.00e+00 | ✅ |
| Shrink (rank=16, 2 pages) | 0.00e+00 | ✅ |
| Expand (rank=8, 1 page) | 0.00e+00 | ✅ |
| Shrink→Expand chain | 0.00e+00 | ✅ |
| Evicted page (-1 in page_table) | page output = 0 | ✅ |

## Speed Tests and Profiling

Model: Llama-3.1-8B-Instruct, 2 LoRA adapters (r=8 + r=64), 300 prompts, input=256, output=128, `--warmup-requests 20`.

| Metric | TP=1 Flat | TP=1 Paged | TP=4 Flat | TP=4 Paged |
|---|---|---|---|---|
| Output throughput (tok/s) | 590.67 | **629.11 (+6.5%)** | 1169.54 | **1294.12 (+10.6%)** |
| Median TPOT (ms) | 51.23 | **48.02 (-6.3%)** | 26.21 | **21.98 (-16.1%)** |
| Median E2E (ms) | 3300.22 | **3119.02 (-5.5%)** | 1670.63 | **1514.66 (-9.3%)** |
| P99 TPOT (ms) | 62.21 | **59.05 (-5.1%)** | 33.29 | **29.49 (-11.4%)** |

Paged outperforms flat in all scenarios. TP=4 advantage is larger because each GPU has less model weight (1/4 of 8B), making LoRA weight overhead proportionally larger — paged's smaller working set saves more.

## Checklist
- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process
1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.

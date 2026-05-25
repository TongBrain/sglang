"""Unit tests for LoRAPagePool — page management (B0) and weight scatter (B1).

B0 tests (page alloc / free / evict / page_table) need only __new__ + manual
field setup — no real base model, no GPU, no torch dependency.

B1 tests (weight scatter) use tiny CPU torch tensors to verify that
adapter weights are correctly distributed across physical pages.

Usage:
    python -m pytest test/registered/unit/lora/test_paged_mem_pool_unit.py -v
"""

try:
    from sglang.test.ci.ci_register import register_cuda_ci

    # CPU-only unit test; no CUDA/distributed dependencies.
    register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
except ImportError:
    pass

import sys
import unittest

import torch

from sglang.srt.lora.paged_mem_pool import LoRAPagePool


def make_bare_pool(total_pages=16, pr=8):
    """Create a LoRAPagePool instance without initialising page storage.

    Sets only the fields needed by B0 page-management methods.  B1 scatter
    tests must set ``A_pages`` / ``B_pages`` separately.
    """
    pool = LoRAPagePool.__new__(LoRAPagePool)
    pool.total_pages = total_pages
    pool.PAGE_RANK_SIZE = pr
    pool.free_page_indices = set(range(total_pages))
    pool.page_table = {}
    pool.adapter_ranks = {}
    pool.page_access_times = [0.0] * total_pages
    pool.A_pages = {}
    pool.B_pages = {}
    pool.embedding_A_pages = {}
    pool.embedding_B_pages = {}
    pool.num_layers = 1
    pool.dtype = torch.float32
    pool.device = "cpu"
    return pool


# ── B0: page management ──────────────────────────────────────────────────


class TestGetNumPagesForRank(unittest.TestCase):
    """B0: get_num_pages_for_rank correctness."""

    def setUp(self):
        self.pool = make_bare_pool()

    def test_rank_zero(self):
        self.assertEqual(self.pool.get_num_pages_for_rank(0), 0)

    def test_rank_below_page_size(self):
        self.assertEqual(self.pool.get_num_pages_for_rank(1), 1)
        self.assertEqual(self.pool.get_num_pages_for_rank(7), 1)

    def test_rank_exactly_page_size(self):
        self.assertEqual(self.pool.get_num_pages_for_rank(8), 1)

    def test_rank_crosses_page_boundary(self):
        self.assertEqual(self.pool.get_num_pages_for_rank(9), 2)

    def test_large_rank(self):
        self.assertEqual(self.pool.get_num_pages_for_rank(64), 8)

    def test_negative_rank(self):
        self.assertEqual(self.pool.get_num_pages_for_rank(-1), 0)


class TestAllocatePages(unittest.TestCase):
    """B0: allocate_pages success / failure / state consistency."""

    def setUp(self):
        self.pool = make_bare_pool(total_pages=8)

    def test_allocate_rank_zero_no_pages(self):
        ok = self.pool.allocate_pages("base", 0)
        self.assertTrue(ok)
        self.assertEqual(self.pool.page_table["base"], [])
        self.assertEqual(self.pool.adapter_ranks["base"], 0)
        self.assertEqual(len(self.pool.free_page_indices), 8)

    def test_allocate_single_page(self):
        ok = self.pool.allocate_pages("a", 8)
        self.assertTrue(ok)
        self.assertEqual(len(self.pool.page_table["a"]), 1)
        self.assertIn(self.pool.page_table["a"][0], range(8))
        self.assertEqual(self.pool.adapter_ranks["a"], 8)
        self.assertEqual(len(self.pool.free_page_indices), 7)

    def test_allocate_multi_page(self):
        ok = self.pool.allocate_pages("a", 20)
        self.assertTrue(ok)
        # ceil(20/8) = 3 pages
        self.assertEqual(len(self.pool.page_table["a"]), 3)
        self.assertEqual(len(self.pool.free_page_indices), 5)

    def test_allocate_exhausts_pool(self):
        for i in range(8):
            ok = self.pool.allocate_pages(f"a{i}", 8)
            self.assertTrue(ok)
        # No more free pages
        ok = self.pool.allocate_pages("extra", 8)
        self.assertFalse(ok)

    def test_allocate_not_enough_pages_does_not_partial(self):
        # Allocate 7/8 pages
        for i in range(7):
            self.pool.allocate_pages(f"a{i}", 8)
        # Try to allocate 2-page adapter — not enough
        ok = self.pool.allocate_pages("big", 16)
        self.assertFalse(ok)
        # Verify free count unchanged (no partial alloc)
        self.assertEqual(len(self.pool.free_page_indices), 1)

    def test_page_physical_indices_are_unique(self):
        self.pool.allocate_pages("a", 16)
        self.pool.allocate_pages("b", 8)
        pages_a = set(self.pool.page_table["a"])
        pages_b = set(self.pool.page_table["b"])
        self.assertFalse(pages_a & pages_b, "physical pages must not overlap")


class TestFreePages(unittest.TestCase):
    """B0: free_pages returns pages to the free pool."""

    def setUp(self):
        self.pool = make_bare_pool(total_pages=8)

    def test_free_all_pages(self):
        self.pool.allocate_pages("a", 16)
        self.assertEqual(len(self.pool.free_page_indices), 6)
        self.pool.free_pages("a")
        self.assertEqual(len(self.pool.free_page_indices), 8)
        self.assertNotIn("a", self.pool.page_table)

    def test_free_nonexistent_uid_is_noop(self):
        self.pool.free_pages("ghost")  # should not raise


class TestEvictPages(unittest.TestCase):
    """B0: page-level LRU eviction."""

    def setUp(self):
        self.pool = make_bare_pool(total_pages=8)

    def _load_adapters(self):
        self.pool.allocate_pages("a", 8)  # 1 page
        self.pool.allocate_pages("b", 8)  # 1 page
        self.pool.allocate_pages("c", 8)  # 1 page

    def test_evict_basic(self):
        self._load_adapters()
        # Mark "a"'s page as recently accessed, "b"'s as old
        self.pool.mark_page_accessed(self.pool.page_table["a"][0])
        self.pool.mark_page_accessed(self.pool.page_table["b"][0])

        # Evict 1 page — should pick LRU (oldest). Since we only marked
        # a and b, c has time 0.0 = oldest.
        evicted = self.pool.evict_pages(1, set())
        self.assertEqual(len(evicted), 1)
        # The evicted page's table entry should be -1
        c_page = self.pool.page_table["c"][0]
        self.assertEqual(c_page, -1)

    def test_evict_respects_protected_set(self):
        self._load_adapters()
        a_page = self.pool.page_table["a"][0]
        evicted = self.pool.evict_pages(1, {a_page})
        self.assertEqual(len(evicted), 1)
        # "a" should NOT be evicted
        self.assertNotEqual(evicted[0], a_page)
        # "a"'s page table should be intact
        self.assertEqual(self.pool.page_table["a"][0], a_page)

    def test_evict_skips_free_pages(self):
        self._load_adapters()
        # Free "a" — its page goes back to free pool
        a_page = self.pool.page_table["a"][0]
        self.pool.free_pages("a")

        evicted = self.pool.evict_pages(1, set())
        self.assertEqual(len(evicted), 1)
        # Should NOT evict a's page (it's already free)
        self.assertNotEqual(evicted[0], a_page)

    def test_evict_returns_correct_count(self):
        self._load_adapters()
        evicted = self.pool.evict_pages(2, set())
        self.assertEqual(len(evicted), 2)


class TestIsCompleteAndMissing(unittest.TestCase):
    """B0: is_complete / get_missing_pages."""

    def setUp(self):
        self.pool = make_bare_pool(total_pages=16)

    def test_uid_not_in_table_is_incomplete(self):
        self.assertFalse(self.pool.is_complete("ghost", 8))

    def test_fully_allocated_is_complete(self):
        self.pool.allocate_pages("a", 16)  # 2 pages
        self.assertTrue(self.pool.is_complete("a", 16))

    def test_rank_zero_is_always_complete(self):
        self.pool.allocate_pages("a", 0)
        self.assertTrue(self.pool.is_complete("a", 0))
        self.assertTrue(self.pool.is_complete(None, 0))

    def test_evicted_page_marks_incomplete(self):
        self.pool.allocate_pages("a", 16)  # 2 pages
        self.pool.evict_pages(1, set())  # evicts 1 page
        self.assertFalse(self.pool.is_complete("a", 16))

    def test_get_missing_pages(self):
        self.pool.allocate_pages("a", 16)  # 2 pages
        # Evict both pages
        for _ in range(2):
            self.pool.evict_pages(1, set())
        missing = self.pool.get_missing_pages("a", 16)
        self.assertEqual(len(missing), 2)

    def test_get_missing_pages_rank_zero_is_empty(self):
        self.assertEqual(self.pool.get_missing_pages("a", 0), [])

    def test_get_missing_pages_nonexistent_uid(self):
        # rank=8 needs ceil(8/8)=1 page, so only [0] is missing
        self.assertEqual(self.pool.get_missing_pages("ghost", 8), [0])


class TestBuildPageTableTensor(unittest.TestCase):
    """B0: build_page_table_tensor."""

    def setUp(self):
        self.pool = make_bare_pool(total_pages=8)

    def test_basic_tensor(self):
        self.pool.allocate_pages("a", 8)
        a_page = self.pool.page_table["a"][0]
        tensor = self.pool.build_page_table_tensor(["a", None], 2)
        self.assertEqual(tensor.shape, (2, 2))
        self.assertEqual(tensor[0, 0].item(), a_page)
        self.assertEqual(tensor[0, 1].item(), -1)  # unused slot
        self.assertEqual(tensor[1, 0].item(), -1)  # None uid

    def test_evicted_page_shows_negative_one(self):
        self.pool.allocate_pages("a", 16)  # 2 pages
        self.pool.evict_pages(1, set())
        tensor = self.pool.build_page_table_tensor(["a"], 2)
        # One should be -1 (evicted)
        count_neg = (tensor == -1).sum().item()
        self.assertEqual(count_neg, 1)


class TestGetProtectedPages(unittest.TestCase):
    """B0: get_protected_pages."""

    def setUp(self):
        self.pool = make_bare_pool(total_pages=8)

    def test_protected_set(self):
        self.pool.allocate_pages("a", 8)
        self.pool.allocate_pages("b", 8)
        protected = self.pool.get_protected_pages({"a"})
        self.assertIn(self.pool.page_table["a"][0], protected)
        self.assertNotIn(self.pool.page_table["b"][0], protected)

    def test_empty_uids(self):
        self.assertEqual(self.pool.get_protected_pages(set()), set())

    def test_none_uid(self):
        self.pool.allocate_pages(None, 0)
        protected = self.pool.get_protected_pages({None})
        self.assertEqual(protected, set())


class TestMaxPagesPerLoraForBatch(unittest.TestCase):
    """B0: max_pages_per_lora_for_batch."""

    def setUp(self):
        self.pool = make_bare_pool(total_pages=16)

    def test_empty_list(self):
        self.assertEqual(self.pool.max_pages_per_lora_for_batch([]), 0)

    def test_all_none_uids(self):
        self.assertEqual(
            self.pool.max_pages_per_lora_for_batch([None, None]), 0
        )

    def test_single_adapter(self):
        self.pool.allocate_pages("a", 16)  # 2 pages
        self.assertEqual(self.pool.max_pages_per_lora_for_batch(["a"]), 2)

    def test_multiple_adapters_max(self):
        self.pool.allocate_pages("a", 8)  # 1 page
        self.pool.allocate_pages("b", 20)  # 3 pages
        self.pool.allocate_pages("c", 4)  # 1 page
        self.assertEqual(
            self.pool.max_pages_per_lora_for_batch(["a", "b", "c"]), 3
        )

    def test_unknown_uid_ignored(self):
        self.pool.allocate_pages("a", 8)
        self.assertEqual(
            self.pool.max_pages_per_lora_for_batch(["a", "unknown"]), 1
        )


class TestTargetModulesProperty(unittest.TestCase):
    """B0: target_modules property."""

    def test_empty_pool(self):
        pool = make_bare_pool()
        self.assertEqual(pool.target_modules, set())

    def test_returns_a_pages_keys(self):
        pool = make_bare_pool()
        pool.A_pages["qkv_proj"] = [torch.zeros(1, 1, 1)]
        pool.A_pages["o_proj"] = [torch.zeros(1, 1, 1)]
        self.assertEqual(pool.target_modules, {"qkv_proj", "o_proj"})

    def test_includes_embedding_keys(self):
        pool = make_bare_pool()
        pool.A_pages["qkv_proj"] = [torch.zeros(1, 1, 1)]
        pool.embedding_A_pages["embed_tokens"] = torch.zeros(1, 1)
        self.assertEqual(
            pool.target_modules, {"qkv_proj", "embed_tokens"}
        )


# ── B1: weight scatter (CPU tensors) ──────────────────────────────────────


def make_pool_with_pages(
    module_name="qkv_proj",
    total_pages=8,
    pr=8,
    input_dim=64,
    output_dim=64,
    c=3,
):
    """Create a LoRAPagePool with A_pages / B_pages for one module."""
    pool = make_bare_pool(total_pages=total_pages, pr=pr)
    # Allocate A_pages and B_pages like _init_pages would
    pool.A_pages[module_name] = [
        torch.zeros(total_pages, pr * c, input_dim)
    ]
    pool.B_pages[module_name] = [
        torch.zeros(total_pages, output_dim, pr)
    ]
    pool.num_layers = 1
    return pool


class TestScatterAWeight(unittest.TestCase):
    """B1: _scatter_a_weight_to_pages."""

    def setUp(self):
        self.pool = make_pool_with_pages("qkv_proj", pr=8, c=3)

    def test_scatter_single_page(self):
        # rank=8, exactly 1 page
        lora_rank = 8
        weight = torch.randn(lora_rank * 3, 64)
        phys_pages = [3]

        self.pool.allocate_pages("a", lora_rank)
        self.pool._scatter_a_weight_to_pages(
            "qkv_proj", 0, weight, lora_rank, phys_pages, c=3
        )

        stored = self.pool.A_pages["qkv_proj"][0][3]  # physical page 3
        # stored shape: [pr * c, hidden] = [24, 64]
        # weight shape:  [r * c, hidden] = [24, 64]
        self.assertTrue(torch.equal(stored, weight))

    def test_scatter_two_pages(self):
        # rank=10, spans 2 pages: page0=[0,7], page1=[8,9]
        lora_rank = 10
        c = 3
        weight_a = torch.randn(lora_rank * c, 64)
        phys_pages = [5, 7]

        self.pool.allocate_pages("a", lora_rank)
        self.pool._scatter_a_weight_to_pages(
            "qkv_proj", 0, weight_a, lora_rank, phys_pages, c=3
        )

        # Page 0 should have rank [0, 7] * c
        page0 = self.pool.A_pages["qkv_proj"][0][5]
        for ci in range(c):
            start = ci * lora_rank
            end = ci * lora_rank + 8  # 8 = PAGE_RANK_SIZE
            expected = weight_a[start:end, :]
            dst_start = ci * 8
            dst_end = ci * 8 + 8
            self.assertTrue(
                torch.equal(page0[dst_start:dst_end], expected),
                f"Page0 component {ci} mismatch",
            )

        # Page 1 should have rank [8, 9] * c
        page1 = self.pool.A_pages["qkv_proj"][0][7]
        for ci in range(c):
            start = ci * lora_rank + 8
            end = ci * lora_rank + 10
            expected = weight_a[start:end, :]
            dst = ci * 8
            self.assertTrue(
                torch.equal(page1[dst:dst + 2], expected),
                f"Page1 component {ci} mismatch",
            )

    def test_scatter_none_weight_zeroes_page(self):
        self.pool.allocate_pages("a", 8)
        weight = None
        self.pool._scatter_a_weight_to_pages(
            "qkv_proj", 0, weight, 8, [2], c=3
        )
        page = self.pool.A_pages["qkv_proj"][0][2]
        self.assertTrue(torch.all(page == 0))


class TestScatterBWeight(unittest.TestCase):
    """B1: _scatter_b_weight_to_pages."""

    def setUp(self):
        self.pool = make_pool_with_pages("o_proj", pr=8, c=1, input_dim=64, output_dim=64)

    def test_scatter_with_scaling(self):
        lora_rank = 8
        weight = torch.randn(64, lora_rank)
        scaling = 2.0
        phys_pages = [1]

        self.pool.allocate_pages("a", lora_rank)
        self.pool._scatter_b_weight_to_pages(
            "o_proj", 0, weight, lora_rank, phys_pages, scaling=scaling
        )

        stored = self.pool.B_pages["o_proj"][0][1]
        expected = weight * scaling
        self.assertTrue(torch.equal(stored, expected))

    def test_scatter_two_pages(self):
        lora_rank = 10
        weight = torch.randn(64, lora_rank)
        phys_pages = [0, 2]

        self.pool.allocate_pages("a", lora_rank)
        self.pool._scatter_b_weight_to_pages(
            "o_proj", 0, weight, lora_rank, phys_pages
        )

        page0 = self.pool.B_pages["o_proj"][0][0]
        page1 = self.pool.B_pages["o_proj"][0][2]

        # Page0: weight[:, 0:8]
        self.assertTrue(torch.equal(page0[:, :8], weight[:, 0:8]))
        self.assertTrue(torch.all(page0[:, 8:] == 0))  # rest zero

        # Page1: weight[:, 8:10]
        self.assertTrue(torch.equal(page1[:, :2], weight[:, 8:10]))
        self.assertTrue(torch.all(page1[:, 2:] == 0))

    def test_scatter_none_weight_zeroes_page(self):
        lora_rank = 8
        weight = None
        phys_pages = [3]

        self.pool.allocate_pages("a", lora_rank)
        self.pool._scatter_b_weight_to_pages(
            "o_proj", 0, weight, lora_rank, phys_pages
        )

        page = self.pool.B_pages["o_proj"][0][3]
        self.assertTrue(torch.all(page == 0))


class TestPageIn(unittest.TestCase):
    """B0: page_in."""

    def setUp(self):
        self.pool = make_bare_pool(total_pages=8)

    def test_page_in_basic(self):
        self.pool.allocate_pages("a", 16)  # 2 pages
        # Evict both
        self.pool.evict_pages(2, set())
        self.assertEqual(self.pool.page_table["a"], [-1, -1])

        # Page them back in
        phys0 = self.pool.page_in("a", 0)
        self.assertNotEqual(phys0, -1)
        self.assertEqual(self.pool.page_table["a"][0], phys0)
        self.assertIn(phys0, range(8))

        phys1 = self.pool.page_in("a", 1)
        self.assertNotEqual(phys1, -1)
        self.assertNotEqual(phys0, phys1)  # distinct physical pages

    def test_page_in_no_free_pages_raises(self):
        self.pool.allocate_pages("a", 8)  # 1 page used
        self.pool.evict_pages(1, set())
        self.pool.free_page_indices.clear()  # simulate full pool
        with self.assertRaises(AssertionError):
            self.pool.page_in("a", 0)


class TestLoadFullWeight(unittest.TestCase):
    """B1: load_lora_weight_to_pages (end-to-end)."""

    def test_base_model_zeroes_pages(self):
        """uid=None → allocated pages should be zeroed."""
        pool = make_pool_with_pages("qkv_proj", pr=8, c=3)
        pool.page_table[None] = [0, 1]
        pool.adapter_ranks[None] = 0

        # Put some non-zero data in those pages first
        pool.A_pages["qkv_proj"][0][0].fill_(42.0)
        pool.A_pages["qkv_proj"][0][1].fill_(99.0)

        pool.load_lora_weight_to_pages(None, None, lora_modules=[{}])
        self.assertTrue(torch.all(pool.A_pages["qkv_proj"][0][0] == 0))
        self.assertTrue(torch.all(pool.A_pages["qkv_proj"][0][1] == 0))

    def test_rank_zero_no_pages_no_crash(self):
        """rank=0 adapter → no pages allocated → no crash."""
        pool = make_pool_with_pages("qkv_proj", pr=8, c=3)
        adapter = unittest.mock.MagicMock()
        adapter.config.r = 0
        pool.page_table["a"] = []
        pool.adapter_ranks["a"] = 0
        # Should not raise
        pool.load_lora_weight_to_pages("a", adapter, lora_modules=[{}])


if __name__ == "__main__":
    unittest.main()

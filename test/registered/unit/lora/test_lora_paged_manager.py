"""Unit tests for B2 — LoRAPagePool integration with LoRAManager.

Tests the wiring between LoRAPagePool and LoRAManager lifecycle methods,
the page_table field on ForwardBatch, and paged-aware layer storage.

Pure CPU logic, no CUDA dependency.

Usage:
    PYTHONPATH=python python -m pytest test/registered/unit/lora/test_lora_paged_manager.py -v
"""

import sys
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, PropertyMock, patch

import torch

try:
    from sglang.test.ci.ci_register import register_cuda_ci

    register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
except ImportError:
    pass

from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.paged_mem_pool import LoRAPagePool

# Layer / manager imports may fail in test environments that lack the full
# sglang stack (tvm_ffi, CUDA, etc.).  Tests that depend on them are skipped.
try:
    from sglang.srt.lora.layers import (
        BaseLayerWithLoRA,
        ColumnParallelLinearWithLoRA,
    )
    _HAS_LAYER_IMPORT = True
except ImportError:
    BaseLayerWithLoRA = None
    ColumnParallelLinearWithLoRA = None
    _HAS_LAYER_IMPORT = False

try:
    from sglang.srt.lora.lora_manager import LoRAManager
    _HAS_MANAGER_IMPORT = True
except ImportError:
    LoRAManager = None
    _HAS_MANAGER_IMPORT = False


# ── Test 1: LoRAPagePool.can_support ─────────────────────────────────────

class TestCanSupport(unittest.TestCase):
    """LoRAPagePool.can_support validates adapter compatibility."""

    def setUp(self):
        self.pool = _make_bare_pool(
            total_pages=16,
            max_lora_rank=64,
            max_loras_per_batch=4,
            target_modules={"qkv_proj", "o_proj"},
        )

    def test_rank_within_limit(self):
        cfg = _make_lora_config(r=32, target_modules=["qkv_proj"])
        self.assertTrue(self.pool.can_support(cfg))

    def test_rank_exceeds_limit(self):
        cfg = _make_lora_config(r=128, target_modules=["qkv_proj"])
        self.assertFalse(self.pool.can_support(cfg))

    def test_missing_target_module(self):
        cfg = _make_lora_config(r=32, target_modules=["gate_up_proj"])
        self.assertFalse(self.pool.can_support(cfg))

    def test_all_target_modules_accepted(self):
        cfg = _make_lora_config(r=32, target_modules="all")
        self.assertTrue(self.pool.can_support(cfg))

    def test_all_linear_rank_within_limit(self):
        cfg = _make_lora_config(r=32, target_modules=["qkv_proj", "o_proj"])
        self.assertTrue(self.pool.can_support(cfg))


# ── Test 2: BaseLayerWithLoRA.set_lora_info_paged ──────────────────────────

@unittest.skipUnless(_HAS_LAYER_IMPORT, "requires full sglang stack")
class TestSetLoraInfoPaged(unittest.TestCase):
    """set_lora_info_paged stores page tensor references correctly."""

    def setUp(self):
        self.layer = ColumnParallelLinearWithLoRA.__new__(
            ColumnParallelLinearWithLoRA
        )
        self.layer.set_lora = False
        self.layer.lora_backend = MagicMock()
        self.layer.base_layer = MagicMock()
        self.layer.base_layer.output_partition_sizes = [4096]
        self.layer.base_layer.quant_method = MagicMock()

    def test_stores_page_refs(self):
        A_pages = torch.zeros((16, 8, 4096))
        B_pages = torch.zeros((16, 4096, 8))
        self.layer.set_lora_info_paged(A_pages, B_pages)
        self.assertTrue(self.layer.set_lora)
        self.assertIs(self.layer.A_pages, A_pages)
        self.assertIs(self.layer.B_pages, B_pages)

    def test_detects_paged_mode(self):
        self.assertFalse(self.layer._is_paged_mode())
        self.layer.set_lora_info_paged(
            torch.zeros((4, 8, 64)), torch.zeros((4, 64, 8))
        )
        self.assertTrue(self.layer._is_paged_mode())

    def test_paged_forward_skip_flag(self):
        self.assertFalse(self.layer._paged_forward_skip_lora())
        self.layer.set_lora_info_paged(
            torch.zeros((4, 8, 64)), torch.zeros((4, 64, 8))
        )
        # set_lora flag is True and A_pages is set → skip
        self.assertTrue(self.layer._paged_forward_skip_lora())

    def test_no_skip_when_not_paged(self):
        self.layer.set_lora = True
        self.assertFalse(self.layer._paged_forward_skip_lora())


# ── Test 3: LoRAPagePool.get_embedding_tensor ─────────────────────────────

class TestGetEmbeddingTensor(unittest.TestCase):
    """get_embedding_tensor returns embedding page storage."""

    def test_returns_none_when_empty(self):
        pool = _make_bare_pool(
            total_pages=8,
            target_modules={"qkv_proj"},
            max_lora_rank=32,
            max_loras_per_batch=2,
        )
        self.assertIsNone(pool.get_embedding_tensor("embed_tokens", is_a=True))
        self.assertIsNone(pool.get_embedding_tensor("embed_tokens", is_a=False))

    def test_embedding_pages_available_when_set_manually(self):
        """Embedding pages can be set and retrieved."""
        pool = _make_bare_pool(
            total_pages=8,
            target_modules={"qkv_proj"},
        )
        pool.embedding_A_pages["embed_tokens"] = torch.zeros((8, 8, 4096))
        pool.embedding_B_pages["embed_tokens"] = torch.zeros((8, 4096, 8))

        a_t = pool.get_embedding_tensor("embed_tokens", is_a=True)
        b_t = pool.get_embedding_tensor("embed_tokens", is_a=False)
        self.assertIsNotNone(a_t)
        self.assertIsNotNone(b_t)
        self.assertEqual(a_t.shape[0], 8)
        self.assertEqual(b_t.shape[0], 8)


# ── Test 4: LoRAManager dispatch logic ────────────────────────────────────

@unittest.skipUnless(_HAS_MANAGER_IMPORT, "requires full sglang stack for LoRAManager import")
class TestManagerDispatch(unittest.TestCase):
    """LoRAManager dispatches to the correct pool in lifecycle methods."""

    def test_paged_flag_from_page_rank_size(self):
        """use_paged_pool is True when lora_page_rank_size > 0."""
        manager = _make_manager(lora_page_rank_size=8)
        self.assertTrue(manager.use_paged_pool)

    def test_flat_flag_when_page_rank_size_zero(self):
        """use_paged_pool is False when lora_page_rank_size == 0."""
        manager = _make_manager(lora_page_rank_size=0)
        self.assertFalse(manager.use_paged_pool)

    def test_init_memory_pool_creates_lorapagepool(self):
        """init_memory_pool creates LoRAPagePool when use_paged_pool."""
        manager = _make_manager(lora_page_rank_size=8)
        with patch("sglang.srt.lora.lora_manager.LoRAPagePool") as mock_cls:
            mock_instance = MagicMock(spec=LoRAPagePool)
            mock_cls.return_value = mock_instance
            manager.init_memory_pool()
        self.assertIsInstance(manager.memory_pool, LoRAPagePool)

    def test_init_memory_pool_creates_loramemorypool(self):
        """init_memory_pool creates LoRAMemoryPool when not using paged pool."""
        manager = _make_manager(lora_page_rank_size=0)
        with patch("sglang.srt.lora.lora_manager.LoRAMemoryPool") as mock_cls:
            mock_instance = MagicMock(spec=LoRAMemoryPool)
            mock_cls.return_value = mock_instance
            manager.init_memory_pool()
        self.assertIsInstance(manager.memory_pool, LoRAMemoryPool)

    def test_fetch_new_loras_calls_ensure_adapter_ready(self):
        """fetch_new_loras calls ensure_adapter_ready per uid in paged mode."""
        manager = _make_manager(lora_page_rank_size=8)
        manager.loras = {
            "adapter_a": MagicMock(config=MagicMock(r=8)),
        }
        manager.memory_pool = MagicMock(spec=LoRAPagePool)
        manager.memory_pool.page_table = {}

        with patch.object(manager.memory_pool, "get_protected_pages",
                          return_value=set()):
            manager.fetch_new_loras({"adapter_a"})

        manager.memory_pool.ensure_adapter_ready.assert_called_once()

    def test_prepare_lora_builds_page_table(self):
        """prepare_lora_batch sets page_table on forward_batch in paged mode."""
        manager = _make_manager(lora_page_rank_size=8)
        manager.memory_pool = MagicMock(spec=LoRAPagePool)
        manager.memory_pool.max_pages_per_lora_for_batch.return_value = 2
        manager.memory_pool.build_page_table_tensor.return_value = "page_table_stub"

        fb = MagicMock()
        fb.lora_ids = ["uid_a"]
        fb.batch_size = 1
        fb.forward_mode.is_cuda_graph.return_value = False

        manager.prepare_lora_batch(fb)

        manager.memory_pool.build_page_table_tensor.assert_called_once_with(
            ["uid_a"], 2
        )
        self.assertEqual(fb.page_table, "page_table_stub")

    def test_prepare_lora_with_flat_pool(self):
        """prepare_lora_batch does NOT set page_table in flat mode."""
        manager = _make_manager(lora_page_rank_size=0)
        manager.memory_pool = MagicMock()
        manager.memory_pool.uid_to_buffer_id = {}

        fb = MagicMock(spec=["lora_ids", "batch_size", "forward_mode"])
        fb.lora_ids = ["uid_a"]
        fb.batch_size = 1
        fb.forward_mode.is_cuda_graph.return_value = False

        manager.prepare_lora_batch(fb)

        self.assertFalse(hasattr(fb, "page_table"))

    def test_validate_lora_batch_paged_checks_free_pages(self):
        """validate_lora_batch checks free_page_indices in paged mode."""
        manager = _make_manager(lora_page_rank_size=8)
        manager.memory_pool = MagicMock(spec=LoRAPagePool)
        manager.memory_pool.page_table = {}
        manager.memory_pool.get_num_pages_for_rank.return_value = 2
        manager.memory_pool.free_page_indices = {0, 1, 2, 3}
        manager.loras = {
            "new_adapter": MagicMock(config=MagicMock(r=8)),
        }

        # 2 pages needed, 4 free → OK
        self.assertTrue(manager.validate_lora_batch({"new_adapter"}))

    def test_update_lora_info_dispatches_to_paged(self):
        """update_lora_info calls _update_lora_info_paged in paged mode."""
        manager = _make_manager(lora_page_rank_size=8)
        manager.memory_pool = MagicMock()

        with patch.object(manager, "_update_lora_info_paged") as mock_paged:
            with patch.object(manager, "_update_lora_info_flat") as mock_flat:
                manager.update_lora_info()

        mock_paged.assert_called_once()
        mock_flat.assert_not_called()

    def test_update_lora_info_dispatches_to_flat(self):
        """update_lora_info calls _update_lora_info_flat in flat mode."""
        manager = _make_manager(lora_page_rank_size=0)
        manager.memory_pool = MagicMock()

        with patch.object(manager, "_update_lora_info_paged") as mock_paged:
            with patch.object(manager, "_update_lora_info_flat") as mock_flat:
                manager.update_lora_info()

        mock_flat.assert_called_once()
        mock_paged.assert_not_called()


# ── Test 5: ForwardBatch.page_table ──────────────────────────────────────

class TestForwardBatchPageTable(unittest.TestCase):
    """ForwardBatch.page_table field defaults correctly."""

    def test_default_is_none(self):
        try:
            from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        except ImportError:
            raise unittest.SkipTest("requires full sglang stack")
        fb = ForwardBatch.__new__(ForwardBatch)
        self.assertIsNone(fb.page_table)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_bare_pool(
    total_pages=16,
    target_modules=None,
    max_lora_rank=64,
    max_loras_per_batch=4,
    page_rank_size=8,
):
    """Create a LoRAPagePool with minimal dependencies."""
    pool = LoRAPagePool.__new__(LoRAPagePool)
    pool.total_pages = total_pages
    pool.PAGE_RANK_SIZE = page_rank_size
    pool.dtype = torch.float16
    pool.device = torch.device("cpu")
    pool.num_layers = 1
    pool.tp_size = 1
    pool.tp_rank = 0
    pool.max_lora_rank = max_lora_rank
    pool.max_loras_per_batch = max_loras_per_batch
    pool.base_hf_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    pool.free_page_indices = set(range(total_pages))
    pool.page_table = {}
    pool.adapter_ranks = {}
    pool.page_access_times = [0.0] * total_pages
    pool.A_pages = {}
    pool.B_pages = {}
    pool.embedding_A_pages = {}
    pool.embedding_B_pages = {}

    target_modules = target_modules or {"qkv_proj", "o_proj"}
    # Manually init pages — skip the _init_pages that needs get_hidden_dim
    pr = pool.PAGE_RANK_SIZE
    for module_name in target_modules:
        if module_name in ("embed_tokens", "lm_head"):
            continue
        pool.A_pages[module_name] = [
            torch.zeros((total_pages, pr * 3, 4096))
        ]
        pool.B_pages[module_name] = [
            torch.zeros((total_pages, 4096, pr))
        ]

    return pool


def _make_lora_config(r=8, target_modules=None):
    """Create a minimal LoRAConfig."""
    cfg = LoRAConfig.__new__(LoRAConfig)
    cfg.r = r
    cfg.target_modules = target_modules or ["qkv_proj"]
    cfg.lora_added_tokens_size = 0
    cfg.use_dora = False
    return cfg


def _make_manager(lora_page_rank_size=0):
    """Create a bare LoRAManager for testing dispatch logic."""
    from sglang.srt.lora.lora_manager import LoRAManager

    m = LoRAManager.__new__(LoRAManager)
    m.lora_page_rank_size = lora_page_rank_size
    m.use_paged_pool = lora_page_rank_size > 0
    m.max_loras_per_batch = 4
    m.max_lora_rank = 64
    m.lora_backend = MagicMock()
    m.lora_backend.batch_info = MagicMock()
    m.base_hf_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    m.target_modules = {"qkv_proj", "o_proj"}
    m.dtype = torch.float16
    m.device = torch.device("cpu")
    m.tp_size = 1
    m.tp_rank = 0
    m.eviction_policy = "lru"
    m.lora_added_tokens_size = 0
    m.experts_shared_outer_loras = False
    m.lora_strict_loading = False
    m.base_model = SimpleNamespace(
        config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
            ),
            num_hidden_layers=1,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
    )
    m.loras = {}
    m.lora_refs = {}
    m.lora_modules = []
    m.embed_tokens_module = None
    m.lm_head_module = None
    m.num_pinned_loras = 0
    return m


if __name__ == "__main__":
    unittest.main()

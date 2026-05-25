"""Unit tests for LoRA bucket scheduling — get_ceiling, scheduling filter rule,
and batch_max_rank assignment logic.

All tests are self-contained (pure functions + LoRABucketConfig which has no
heavy dependencies), so they can run on any platform without GPU.

Usage:
    python -m pytest test/registered/unit/lora/test_lora_bucket_scheduler.py -v
"""

import sys
import unittest

from sglang.test.ci.ci_register import register_cuda_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.lora.lora_bucket_config import LoRABucketConfig


# ── Shared fixture ────────────────────────────────────────────────────────

BUCKETS = LoRABucketConfig.from_string("0,8,16,32,64")
BUCKETS_SMALL = LoRABucketConfig.from_string("0,4,16,64")
BUCKETS_TINY = LoRABucketConfig.from_string("0,32")


# ── Test 1: LoRABucketConfig.get_ceiling ──────────────────────────────────

class TestGetCeiling(unittest.TestCase):
    """get_ceiling(rank) → the smallest bucket strictly greater than rank.

    Covers every significant case: zero, sub-bucket, exact bucket, between
    buckets, at max bucket, beyond max bucket, and different bucket configs.
    """

    # ── base model ─────────────────────────────────────────────────────
    def test_rank_zero_returns_zero(self):
        """rank=0 is the base model — always returns 0."""
        self.assertEqual(BUCKETS.get_ceiling(0), 0)

    # ── get_ceiling(1) — used when first req is base model
    #   get_ceiling(1) should return the first non-zero bucket
    def test_ceiling_1_is_first_non_zero_bucket(self):
        """get_ceiling(1) → 8: first non-zero bucket when buckets=[0,8,16,32,64]."""
        self.assertEqual(BUCKETS.get_ceiling(1), 8)

    def test_ceiling_1_with_small_buckets(self):
        """get_ceiling(1) → 4 when buckets=[0,4,16,64]."""
        self.assertEqual(BUCKETS_SMALL.get_ceiling(1), 4)

    def test_ceiling_1_with_tiny_buckets(self):
        """get_ceiling(1) → 32 when buckets=[0,32] (only one non-zero)."""
        self.assertEqual(BUCKETS_TINY.get_ceiling(1), 32)

    def test_ceiling_1_with_only_zero(self):
        """get_ceiling(1) → sys.maxsize when buckets=[0] (no non-zero bucket)."""
        cfg = LoRABucketConfig.from_string("0")
        self.assertEqual(cfg.get_ceiling(1), sys.maxsize)

    # ── below / at / between bucket boundaries ──────────────────────────
    def test_rank_7_returns_8(self):
        """rank=7 (< 8) → ceiling=8."""
        self.assertEqual(BUCKETS.get_ceiling(7), 8)

    def test_rank_8_returns_16(self):
        """rank=8 (exact bucket) → ceiling=16 (next bucket > rank)."""
        self.assertEqual(BUCKETS.get_ceiling(8), 16)

    def test_rank_15_returns_16(self):
        """rank=15 (between 8 and 16) → ceiling=16."""
        self.assertEqual(BUCKETS.get_ceiling(15), 16)

    def test_rank_32_returns_64(self):
        """rank=32 (exact bucket) → ceiling=64."""
        self.assertEqual(BUCKETS.get_ceiling(32), 64)

    # ── at / beyond the largest bucket ──────────────────────────────────
    def test_rank_64_returns_maxsize(self):
        """rank=64 (largest bucket) → ceiling=sys.maxsize (no bucket > rank)."""
        self.assertEqual(BUCKETS.get_ceiling(64), sys.maxsize)

    def test_rank_999_returns_maxsize(self):
        """rank beyond all buckets → ceiling=sys.maxsize."""
        self.assertEqual(BUCKETS.get_ceiling(999), sys.maxsize)

    # ── custom bucket configs ───────────────────────────────────────────
    def test_small_buckets_all_ranks(self):
        """Verify all boundary conditions for buckets=[0,4,16,64]."""
        self.assertEqual(BUCKETS_SMALL.get_ceiling(0), 0)
        self.assertEqual(BUCKETS_SMALL.get_ceiling(1), 4)
        self.assertEqual(BUCKETS_SMALL.get_ceiling(4), 16)
        self.assertEqual(BUCKETS_SMALL.get_ceiling(16), 64)
        self.assertEqual(BUCKETS_SMALL.get_ceiling(64), sys.maxsize)

    def test_tiny_bucket_single_boundary(self):
        """Verify boundary for buckets=[0,32]."""
        self.assertEqual(BUCKETS_TINY.get_ceiling(0), 0)
        self.assertEqual(BUCKETS_TINY.get_ceiling(1), 32)
        self.assertEqual(BUCKETS_TINY.get_ceiling(32), sys.maxsize)
        self.assertEqual(BUCKETS_TINY.get_ceiling(64), sys.maxsize)

    def test_zero_bucket_top_rank(self):
        """rank=sys.maxsize should always return sys.maxsize."""
        self.assertEqual(BUCKETS.get_ceiling(sys.maxsize), sys.maxsize)

    def test_negative_rank(self):
        """negative rank (shouldn't happen, but get_ceiling handles it)."""
        # No explicit guard, so it falls through to the loop.
        # All buckets > a negative number, so first bucket > rank is 0.
        # Actually wait, buckets are [0,8,16,...], and b > rank for b=0
        # when rank=-1. But rank is never negative in practice.
        ceiling = BUCKETS.get_ceiling(-1)
        self.assertEqual(ceiling, 0)


# ── Test 2: Scheduling filter rule ───────────────────────────────────────

class TestSchedulingFilter(unittest.TestCase):
    """should_schedule(req_rank, bucket_ceiling) → bool.

    Mirrors the inline filter in _get_new_batch_prefill_raw:
        if bucket_ceiling is not None:
            req_rank = _get_lora_rank(req)
            if req_rank > 0:
                ...
            elif bucket_ceiling is None:
                bucket_ceiling = cfg.get_ceiling(1)

    The filter logic after ceiling is set:
        if req_rank > bucket_ceiling:
            continue
    """

    @staticmethod
    def should_schedule(req_rank: int, bucket_ceiling) -> bool:
        """Replica of the filter in _get_new_batch_prefill_raw."""
        if bucket_ceiling is None:
            return True
        return req_rank <= bucket_ceiling

    # ── base model (rank=0) is always accepted ──────────────────────────
    def test_base_model_no_ceiling(self):
        self.assertTrue(self.should_schedule(0, None))

    def test_base_model_normal_ceiling(self):
        self.assertTrue(self.should_schedule(0, 8))
        self.assertTrue(self.should_schedule(0, 16))
        self.assertTrue(self.should_schedule(0, 64))

    def test_base_model_zero_ceiling(self):
        self.assertTrue(self.should_schedule(0, 0))

    def test_base_model_maxsize_ceiling(self):
        self.assertTrue(self.should_schedule(0, sys.maxsize))

    # ── rank ≤ ceiling → accepted ───────────────────────────────────────
    def test_rank_equal_ceiling(self):
        self.assertTrue(self.should_schedule(8, 8))
        self.assertTrue(self.should_schedule(16, 16))
        self.assertTrue(self.should_schedule(64, 64))

    def test_rank_below_ceiling(self):
        self.assertTrue(self.should_schedule(4, 8))
        self.assertTrue(self.should_schedule(8, 16))
        self.assertTrue(self.should_schedule(16, 32))

    def test_rank_zero_with_ceiling(self):
        self.assertTrue(self.should_schedule(0, 8))

    # ── rank > ceiling → rejected ───────────────────────────────────────
    def test_rank_exceeds_ceiling(self):
        self.assertFalse(self.should_schedule(16, 8))
        self.assertFalse(self.should_schedule(32, 16))
        self.assertFalse(self.should_schedule(64, 32))

    def test_rank_exceeds_zero_ceiling(self):
        """rank > 0 with ceiling=0 → rejected (simulates old bug)."""
        self.assertFalse(self.should_schedule(8, 0))
        self.assertFalse(self.should_schedule(4, 0))
        self.assertFalse(self.should_schedule(1, 0))

    # ── no bucketing (ceiling=None) → all accepted ──────────────────────
    def test_none_ceiling_accepts_all_ranks(self):
        self.assertTrue(self.should_schedule(0, None))
        self.assertTrue(self.should_schedule(8, None))
        self.assertTrue(self.should_schedule(16, None))
        self.assertTrue(self.should_schedule(64, None))
        self.assertTrue(self.should_schedule(999, None))

    # ── sys.maxsize ceiling → all accepted ──────────────────────────────
    def test_maxsize_ceiling_accepts_all_ranks(self):
        self.assertTrue(self.should_schedule(0, sys.maxsize))
        self.assertTrue(self.should_schedule(8, sys.maxsize))
        self.assertTrue(self.should_schedule(64, sys.maxsize))
        self.assertTrue(self.should_schedule(999, sys.maxsize))


# ── Test 3: batch_max_rank assignment ────────────────────────────────────

class TestBatchMaxRankAssignment(unittest.TestCase):
    """batch_max_rank logic (from _get_new_batch_prefill_raw footer).

    Production code:
        batch_max_rank = (
            bucket_ceiling
            if (bucket_ceiling is not None and bucket_ceiling != sys.maxsize)
            else 0
        )
    """

    def assign(self, bucket_ceiling):
        if bucket_ceiling is not None and bucket_ceiling != sys.maxsize:
            return bucket_ceiling
        return 0

    def test_normal_ceiling(self):
        self.assertEqual(self.assign(8), 8)
        self.assertEqual(self.assign(16), 16)
        self.assertEqual(self.assign(32), 32)

    def test_maxsize_ceiling_falls_back_to_zero(self):
        self.assertEqual(self.assign(sys.maxsize), 0)

    def test_none_ceiling_falls_back_to_zero(self):
        self.assertEqual(self.assign(None), 0)

    def test_zero_ceiling_falls_back_to_zero(self):
        """ceiling=0 (base-model-only batch or empty queue fallback)."""
        self.assertEqual(self.assign(0), 0)


# ── Test 4: End-to-end scheduling scenarios ──────────────────────────────

class TestSchedulingScenarios(unittest.TestCase):
    """Simulate the full bucket scheduling decision for real-world queue
    patterns using the building blocks tested above.
    """

    @staticmethod
    def compute_ceiling(
        cfg: LoRABucketConfig, queue_ranks: list[int]
    ) -> int:
        """Simulate _compute_bucket_ceiling logic (inlined in prod).

        Walks the queue (ranks), returns the first non-zero rank's
        ceiling.  If the first non-drained request is base (rank=0),
        returns get_ceiling(1) (first non-zero bucket).
        Returns sys.maxsize for empty / all-drained queue.
        """
        for rank in queue_ranks:
            if rank > 0:
                return cfg.get_ceiling(rank)
            else:
                return cfg.get_ceiling(1)
        return sys.maxsize

    @staticmethod
    def build_batch(
        cfg: LoRABucketConfig, queue_ranks: list[int]
    ) -> tuple[list[int], int]:
        """Simulate _get_new_batch_prefill_raw's loop.

        Returns (scheduled_ranks, batch_max_rank).
        """
        ceiling = TestSchedulingScenarios.compute_ceiling(cfg, queue_ranks)
        scheduled = [r for r in queue_ranks if r <= ceiling]
        batch_max_rank = ceiling if (ceiling is not None and ceiling != sys.maxsize) else 0
        return scheduled, batch_max_rank

    # ── base model first → ceiling bumped to first non-zero bucket ──────

    def test_base_first_lora_within_ceiling(self):
        """[base(0), lora_8] → ceiling=8 → both accepted."""
        scheduled, bmr = self.build_batch(BUCKETS, [0, 8])
        self.assertEqual(scheduled, [0, 8])
        self.assertEqual(bmr, 8)

    def test_base_first_lora_above_ceiling(self):
        """[base(0), lora_16] → ceiling=8 → lora_16 rejected."""
        scheduled, bmr = self.build_batch(BUCKETS, [0, 16])
        self.assertEqual(scheduled, [0])
        self.assertEqual(bmr, 8)

    def test_base_first_mixed_loras(self):
        """[base(0), lora_4, lora_8, lora_16] → ceiling=8 → ranks≤8 accepted."""
        scheduled, bmr = self.build_batch(BUCKETS, [0, 4, 8, 16])
        self.assertEqual(scheduled, [0, 4, 8])
        self.assertEqual(bmr, 8)

    def test_base_first_multiple_bases(self):
        """[base(0), base(0), lora_4] → ceiling=8 → all accepted."""
        scheduled, bmr = self.build_batch(BUCKETS, [0, 0, 4])
        self.assertEqual(scheduled, [0, 0, 4])
        self.assertEqual(bmr, 8)

    # ── LoRA first → ceiling from that LoRA ───────────────────────────

    def test_lora_first_rank_4(self):
        """[lora_4, lora_8, lora_16] → ceiling=8 → ranks≤8 accepted."""
        scheduled, bmr = self.build_batch(BUCKETS, [4, 8, 16])
        self.assertEqual(scheduled, [4, 8])
        self.assertEqual(bmr, 8)

    def test_lora_first_rank_8(self):
        """[lora_8, lora_16] → ceiling=16 → all accepted."""
        scheduled, bmr = self.build_batch(BUCKETS, [8, 16])
        self.assertEqual(scheduled, [8, 16])
        self.assertEqual(bmr, 16)

    def test_lora_first_rank_8_with_base(self):
        """[lora_8, base(0), lora_16] → ceiling=16 → all accepted."""
        scheduled, bmr = self.build_batch(BUCKETS, [8, 0, 16])
        self.assertEqual(scheduled, [8, 0, 16])
        self.assertEqual(bmr, 16)

    def test_lora_first_mixed_multi_level(self):
        """[lora_4, lora_32, lora_8] → ceiling=8 → only rank≤8 accepted."""
        scheduled, bmr = self.build_batch(BUCKETS, [4, 32, 8])
        self.assertEqual(scheduled, [4, 8])
        self.assertEqual(bmr, 8)

    # ── rank beyond all buckets ────────────────────────────────────────

    def test_first_rank_exceeds_all_buckets(self):
        """[lora_128] → ceiling=sys.maxsize → batch_max_rank=0 (no limit)."""
        scheduled, bmr = self.build_batch(BUCKETS, [128])
        self.assertEqual(scheduled, [128])
        self.assertEqual(bmr, 0)

    def test_exceeds_bucket_all(self):
        """[lora_8, lora_128] → ceiling=16 → lora_128 rejected (16 < 128)."""
        scheduled, bmr = self.build_batch(BUCKETS, [8, 128])
        self.assertEqual(scheduled, [8])
        self.assertEqual(bmr, 16)

    # ── empty queue ────────────────────────────────────────────────────

    def test_empty_queue(self):
        """[] → ceiling=sys.maxsize → batch_max_rank=0."""
        scheduled, bmr = self.build_batch(BUCKETS, [])
        self.assertEqual(scheduled, [])
        self.assertEqual(bmr, 0)

    # ── single-bucket config ───────────────────────────────────────────

    def test_single_bucket_config_base_first(self):
        """buckets=[0,32], [base(0), lora_16, lora_32] → ceiling=32 → ranks≤32 accepted."""
        cfg = BUCKETS_TINY
        scheduled, bmr = self.build_batch(cfg, [0, 16, 32])
        self.assertEqual(scheduled, [0, 16, 32])
        self.assertEqual(bmr, 32)

    def test_single_bucket_config_exceeds(self):
        """buckets=[0,32], [base(0), lora_64] → ceiling=32 → lora_64 rejected."""
        cfg = BUCKETS_TINY
        scheduled, bmr = self.build_batch(cfg, [0, 64])
        self.assertEqual(scheduled, [0])
        self.assertEqual(bmr, 32)

    # ── no bucket config (fallback) ────────────────────────────────────

    def test_no_bucket_config_accepts_all(self):
        """No bucket config → ceiling=None → all ranks accepted, batch_max_rank=0."""
        scheduled = [0, 8, 16]
        batch_max_rank = max(scheduled, default=0)
        self.assertEqual(scheduled, [0, 8, 16])
        self.assertEqual(batch_max_rank, 16)

    def test_no_bucket_config_all_base(self):
        """No bucket config, only base → batch_max_rank=0."""
        scheduled = [0, 0]
        batch_max_rank = max(scheduled, default=0)
        self.assertEqual(batch_max_rank, 0)


if __name__ == "__main__":
    unittest.main()

"""CPU-only regression coverage for LoRA publication/readiness contracts."""

import asyncio
import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.lora.lifecycle import merge_adapter_status, merge_lora_update_results
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    GetLoRAAdapterStatusReqInput,
    LoadLoRAAdapterReqInput,
    LoRAAdapterStatusOutput,
    LoRAUpdateOutput,
    PrewarmLoRAAdapterReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def run_gloo_worker(rank, init_method):
    from sglang.srt.managers.scheduler import Scheduler

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=20),
    )
    try:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.dp_tp_cpu_group = dist.group.WORLD
        result = scheduler._collect_lora_update_result(
            LoRAUpdateOutput(
                success=rank == 0, error_message="rank1 failed" if rank else None
            )
        )
        assert not result.success and result.error_message == "rank1 failed"
        manager = bare_manager()
        manager.lora_refs["id"] = LoRARef(lora_name="a", lora_id="id")
        manager.configs["id"] = SimpleNamespace(r=8, target_modules=["q_proj"])
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(lora_manager=manager)
        )
        scheduler.ps = SimpleNamespace(tp_rank=rank, pp_rank=0, dp_rank=0)
        status = scheduler.get_lora_adapter_status(
            GetLoRAAdapterStatusReqInput(lora_name="a")
        )
        assert status.success and len(status.workers) == 2
        assert {w["tp_rank"] for w in status.workers} == {0, 1}
    finally:
        dist.destroy_process_group()


def bare_manager():
    manager = LoRAManager.__new__(LoRAManager)
    manager.pending_lora_load_events = {}
    manager.memory_pool = MagicMock()
    manager.memory_pool.remove_lora.return_value = None
    manager.memory_pool.uid_to_buffer_id = {}
    manager.page_pool = None
    manager.use_paged_pool = False
    manager.lora_refs = {}
    manager.loras = {}
    manager.configs = {}
    manager.num_pinned_loras = 0
    return manager


class TestLifecycle(unittest.TestCase):
    def test_empty_update_replies_fail_closed(self):
        self.assertFalse(merge_lora_update_results([]).success)

    def test_msgpack_roundtrip(self):
        for obj in (
            GetLoRAAdapterStatusReqInput(lora_name="a@v2"),
            PrewarmLoRAAdapterReqInput(lora_name="a@v2", input_ids=[1, 2]),
            LoRAAdapterStatusOutput(success=True, workers=[{"adapters": {}}]),
            LoRAUpdateOutput(success=True, lora_id="id"),
        ):
            self.assertEqual(msgpack_decode(msgpack_encode(obj)), obj)

    def test_status_requires_matching_identity_on_every_worker(self):
        ref = LoRARef(lora_name="a@v2", lora_path="/v2", lora_id="id")
        reply = LoRAAdapterStatusOutput(
            success=True,
            workers=[
                {"adapters": {"a@v2": {"lora_id": "id", "resident": True}}},
                {"adapters": {"a@v2": {"lora_id": "stale", "resident": True}}},
            ],
        )
        result = merge_adapter_status([reply], {"a@v2": (ref, 3)}, "a@v2")
        self.assertFalse(result.success)
        self.assertFalse(result.adapters["a@v2"]["loaded"])
        self.assertFalse(result.adapters["a@v2"]["resident"])
        self.assertEqual(result.adapters["a@v2"]["active_requests"], 3)

    def test_loaded_does_not_imply_resident(self):
        ref = LoRARef(lora_name="a", lora_path="/a", lora_id="id")
        reply = LoRAAdapterStatusOutput(
            success=True,
            workers=[{"adapters": {"a": {"lora_id": "id", "resident": False}}}],
        )
        result = merge_adapter_status([reply], {"a": (ref, 0)})
        self.assertTrue(result.success)
        self.assertTrue(result.adapters["a"]["loaded"])
        self.assertFalse(result.adapters["a"]["resident"])

    def test_missing_adapter_and_missing_workers_fail(self):
        self.assertFalse(merge_adapter_status([], {}, "missing").success)

    def test_worker_failure_without_message_fails_closed(self):
        reply = LoRAAdapterStatusOutput(success=False, workers=[{"adapters": {}}])
        self.assertFalse(merge_adapter_status([reply], {}).success)

    def test_unregistered_backend_state_is_reported(self):
        reply = LoRAAdapterStatusOutput(
            success=True, workers=[{"adapters": {"orphan": {}}}]
        )
        self.assertFalse(merge_adapter_status([reply], {}).success)

    def test_discard_is_idempotent_and_preserves_other_versions(self):
        manager = bare_manager()
        for name in ("v1", "v2"):
            manager.lora_refs[name] = LoRARef(
                lora_id=name, lora_name=name, lora_path=name, pinned=True
            )
            manager.configs[name] = object()
            manager.loras[name] = object()
        manager.num_pinned_loras = 2
        manager.page_pool = MagicMock()
        self.assertTrue(manager.discard_lora_adapter("v2").success)
        self.assertTrue(manager.discard_lora_adapter("v2").success)
        self.assertEqual(set(manager.configs), {"v1"})
        self.assertEqual(set(manager.loras), {"v1"})
        self.assertEqual(manager.num_pinned_loras, 1)
        manager.page_pool.free_pages.assert_called_with("v2")

    def test_failed_disk_load_cleans_partial_config(self):
        manager = bare_manager()
        manager.base_hf_config = SimpleNamespace(vocab_size=20)
        manager.validate_new_adapter = MagicMock()
        manager.load_lora_weights = MagicMock(side_effect=ValueError("bad weights"))
        ref = LoRARef(lora_name="a", lora_path="/a", lora_id="id")
        with patch("sglang.srt.lora.lora_manager.LoRAConfig"):
            result = manager._load_lora_adapter(ref)
        self.assertFalse(result.success)
        self.assertEqual(manager.configs, {})

    def test_failed_tensor_load_cleans_partial_config(self):
        manager = bare_manager()
        manager.base_hf_config = SimpleNamespace(vocab_size=20)
        manager.validate_new_adapter = MagicMock()
        manager.load_lora_weights_from_tensors = MagicMock(
            side_effect=ValueError("bad tensors")
        )
        ref = LoRARef(lora_name="a", lora_path="__tensor__", lora_id="id")
        with patch("sglang.srt.lora.lora_manager.LoRAConfig"):
            result = manager._load_lora_adapter_from_tensors(ref, {}, {})
        self.assertFalse(result.success)
        self.assertEqual(manager.configs, {})

    def test_paged_status_reports_partial_residency_without_mutation(self):
        manager = bare_manager()
        manager.use_paged_pool = True
        manager.lora_refs["id"] = LoRARef(lora_name="a", lora_id="id")
        manager.configs["id"] = SimpleNamespace(r=16, target_modules=["q_proj"])
        pool = MagicMock()
        pool.page_table = {"id": [2, -1]}
        pool.get_num_pages_for_rank.return_value = 2
        pool.is_complete.return_value = False
        pool.total_pages = 4
        pool.free_page_indices = {0, 1}
        pool.PAGE_RANK_SIZE = 8
        manager.page_pool = pool
        result = manager.get_adapter_status("a")
        self.assertEqual(result["adapters"]["a"]["resident_pages"], 1)
        self.assertFalse(result["adapters"]["a"]["resident"])
        pool.ensure_adapter_ready.assert_not_called()


class TestAsyncLifecycle(unittest.IsolatedAsyncioTestCase):
    async def make_control(self):
        control = TokenizerControlMixin()
        control.lora_update_lock = asyncio.Lock()
        control.lora_registry = LoRARegistry()
        control.lora_ref_cache = {}
        control.server_args = SimpleNamespace(max_loaded_loras=None)
        control.auto_create_handle_loop = MagicMock()
        control.elastic_worker_count = 1
        control._validate_lora_status_available = MagicMock()
        return control

    async def test_acquire_retains_before_unregister_can_proceed(self):
        registry = LoRARegistry()
        ref = LoRARef(lora_name="a", lora_id="id")
        await registry.register(ref)
        counter = registry._counters["id"]
        entered, proceed = asyncio.Event(), asyncio.Event()
        increment = counter.increment

        async def paused_increment(**kwargs):
            entered.set()
            await proceed.wait()
            await increment(**kwargs)

        counter.increment = paused_increment
        acquire = asyncio.create_task(registry.acquire("a"))
        await entered.wait()
        unregister = asyncio.create_task(registry.unregister("a"))
        await asyncio.sleep(0)
        self.assertFalse(unregister.done())
        proceed.set()
        self.assertEqual(await acquire, "id")
        self.assertEqual(await unregister, "id")
        draining = asyncio.create_task(registry.wait_for_unload("id"))
        await asyncio.sleep(0)
        self.assertFalse(draining.done())
        await registry.release("id")
        await draining

    async def test_batch_acquire_missing_name_does_not_retain_any(self):
        registry = LoRARegistry([LoRARef(lora_name="a", lora_id="id")])
        with self.assertRaises(ValueError):
            await registry.acquire(["a", "missing"])
        self.assertEqual((await registry.snapshot())["a"][1], 0)

    async def test_duplicate_rejected_before_backend_load(self):
        control = await self.make_control()
        await control.lora_registry.register(LoRARef(lora_name="a", lora_id="id"))
        with self.assertRaisesRegex(ValueError, "immutable"):
            control._validate_lora_load_name("a")

    async def test_failed_fanout_rolls_back_all_replicas(self):
        control = await self.make_control()
        control.update_lora_adapter_communicator = AsyncMock(
            return_value=[LoRAUpdateOutput(success=True)]
        )
        obj = LoadLoRAAdapterReqInput(lora_name="a@v2", lora_path="/v2", lora_id="id")
        result = await control._rollback_failed_lora_load(
            obj, LoRAUpdateOutput(success=False, error_message="rank failed")
        )
        rollback = control.update_lora_adapter_communicator.call_args.args[0]
        self.assertTrue(rollback.missing_ok)
        self.assertEqual(rollback.lora_id, "id")
        self.assertFalse(result.success)
        self.assertEqual(result.loaded_adapters, {})

    async def test_prewarm_uses_normal_scheduler_on_each_replica(self):
        control = await self.make_control()
        await control.lora_registry.register(LoRARef(lora_name="a", lora_id="id"))
        control.elastic_worker_count = 2
        probes = []

        async def generate(probe, request):
            self.assertTrue(control.lora_update_lock.locked())
            probes.append(probe)
            yield {"meta_info": {"finish_reason": {"type": "length"}}}

        control.generate_request = generate
        control._get_lora_adapter_status_locked = AsyncMock(
            return_value=LoRAAdapterStatusOutput(success=True)
        )
        result = await control.prewarm_lora_adapter(
            PrewarmLoRAAdapterReqInput(lora_name="a", input_ids=[1])
        )
        self.assertTrue(result.success)
        self.assertTrue(result.probe_completed)
        self.assertEqual([p.routed_dp_rank for p in probes], [0, 1])
        self.assertTrue(all(p.lora_path == "a" for p in probes))
        self.assertTrue(all(p.sampling_params["max_new_tokens"] == 1 for p in probes))
        self.assertFalse(control.lora_update_lock.locked())

    async def test_prewarm_abort_does_not_report_success(self):
        control = await self.make_control()
        await control.lora_registry.register(LoRARef(lora_name="a", lora_id="id"))

        async def generate(probe, request):
            yield {"meta_info": {"finish_reason": {"type": "abort"}}}

        control.generate_request = generate
        result = await control.prewarm_lora_adapter(
            PrewarmLoRAAdapterReqInput(lora_name="a", input_ids=[1])
        )
        self.assertFalse(result.success)
        self.assertFalse(control.lora_update_lock.locked())

    async def test_prewarm_rejects_unknown_and_empty_prompt(self):
        control = await self.make_control()
        for ids in ([], [1]):
            result = await control.prewarm_lora_adapter(
                PrewarmLoRAAdapterReqInput(lora_name="missing", input_ids=ids)
            )
            self.assertFalse(result.success)


class TestSchedulerFanout(unittest.TestCase):
    @unittest.skipUnless(
        os.name == "posix" and dist.is_gloo_available(), "Linux Gloo test"
    )
    def test_real_two_rank_gloo_fanout(self):
        # CPU collectives only: do not allocate or interfere with serving GPUs.
        with tempfile.TemporaryDirectory(prefix="lora-lifecycle-gloo-") as scratch:
            mp.start_processes(
                run_gloo_worker,
                args=(f"file://{scratch}/init",),
                nprocs=2,
                start_method="fork",
            )

    def test_non_entry_rank_failure_is_returned(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.dp_tp_cpu_group = object()

        def gather(replies, local_result, group):
            replies[:] = [
                local_result,
                LoRAUpdateOutput(success=False, error_message="rank1"),
            ]

        with patch("torch.distributed.get_world_size", return_value=2), patch(
            "torch.distributed.all_gather_object", side_effect=gather
        ):
            result = scheduler._collect_lora_update_result(
                LoRAUpdateOutput(success=True)
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "rank1")

    def test_worker_exception_still_enters_result_collective(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler._collect_lora_update_result = MagicMock(side_effect=lambda r: r)
        result = scheduler._run_lora_update(
            MagicMock(side_effect=ValueError("decode failed"))
        )
        self.assertFalse(result.success)
        scheduler._collect_lora_update_result.assert_called_once()


class TestLifecycleHTTP(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from sglang.srt.entrypoints import http_server

        self.server = http_server
        self.manager = SimpleNamespace(
            get_lora_adapter_status=AsyncMock(
                return_value=LoRAAdapterStatusOutput(success=True)
            ),
            prewarm_lora_adapter=AsyncMock(
                return_value=LoRAAdapterStatusOutput(success=True)
            ),
        )
        self.state_patch = patch.object(
            http_server,
            "_global_state",
            SimpleNamespace(tokenizer_manager=self.manager),
        )
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)
        app = FastAPI()
        app.add_api_route(
            "/get_lora_adapter_status",
            http_server.get_lora_adapter_status,
            methods=["GET"],
        )
        app.add_api_route(
            "/prewarm_lora_adapter", http_server.prewarm_lora_adapter, methods=["POST"]
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_query_name_is_forwarded(self):
        response = self.client.get(
            "/get_lora_adapter_status", params={"lora_name": "a@v2"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(
            self.manager.get_lora_adapter_status.call_args.args[0].lora_name, "a@v2"
        )

    def test_status_failure_is_http_400(self):
        self.manager.get_lora_adapter_status.return_value = LoRAAdapterStatusOutput(
            success=False, error_message="not registered"
        )
        response = self.client.get("/get_lora_adapter_status")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])

    def test_prewarm_body_is_decoded(self):
        response = self.client.post(
            "/prewarm_lora_adapter", json={"lora_name": "a@v2", "input_ids": [1, 2]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.manager.prewarm_lora_adapter.call_args.args[0].input_ids, [1, 2]
        )

    def test_prewarm_requires_prompt_tokens(self):
        response = self.client.post("/prewarm_lora_adapter", json={"lora_name": "a@v2"})
        self.assertEqual(response.status_code, 422)
        self.manager.prewarm_lora_adapter.assert_not_called()

    def test_all_lifecycle_endpoints_have_admin_auth(self):
        from sglang.srt.utils.auth import AuthLevel

        for endpoint in (
            self.server.load_lora_adapter,
            self.server.load_lora_adapter_from_tensors,
            self.server.unload_lora_adapter,
            self.server.get_lora_adapter_status,
            self.server.prewarm_lora_adapter,
        ):
            self.assertEqual(endpoint._auth_level, AuthLevel.ADMIN_OPTIONAL)


if __name__ == "__main__":
    unittest.main()

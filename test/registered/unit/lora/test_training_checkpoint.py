"""CPU-only exact-resume and fail-closed pause contracts."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.lora.training.checkpoint import (
    identity,
    pause_requested,
    read_checkpoint,
    request_pause,
    request_step_interval,
    requested_step_interval,
    restore_checkpoint,
    save_checkpoint,
)
from sglang.srt.lora.training.jobs import JobStore, TrainingSpec, atomic_json
from sglang.srt.lora.training.runtime import execute_training, resume_training


class TestTrainingCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_path = self.root / "base"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}")
        self.data = self.root / "data.jsonl"
        self.data.write_text('{"prompt":"Q", "response":"A"}\n')
        self.spec = TrainingSpec(
            "resume1", "user@v3", str(self.model_path), str(self.data), 4, max_steps=5
        )
        self.store = JobStore(self.root / "jobs")
        self.job = self.store.create(self.spec)
        self.directory = self.store.directory(self.spec.job_id)
        self.ident = identity(self.spec, self.job)
        torch.set_num_threads(1)

    def model_optimizer(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.Dropout(0.25), torch.nn.Linear(8, 4)
        )
        return model, torch.optim.AdamW(model.parameters(), lr=0.0002)

    def seed(self):
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

    def steps(self, model, optimizer, start, end, losses):
        for step in range(start, end):
            x = (
                torch.rand(2, 4)
                + random.random()
                + float(np.random.random())
                + step % 3
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(x).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

    def saved(self):
        self.seed()
        model, optimizer = self.model_optimizer()
        losses = []
        self.steps(model, optimizer, 0, 2, losses)
        manifest = save_checkpoint(
            self.directory,
            model,
            optimizer,
            self.ident,
            step=2,
            sample_count=3,
            losses=losses,
            initial_digest="initial",
        )
        return model, optimizer, manifest

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_continuous_equals_resume_weights_optimizer_losses_all_rng(self):
        self.seed()
        continuous, continuous_opt = self.model_optimizer()
        reference = []
        self.steps(continuous, continuous_opt, 0, 5, reference)
        expected_rng = (random.random(), float(np.random.random()), torch.rand(4))
        _, _, manifest = self.saved()
        payload, actual = read_checkpoint(self.directory, self.ident)
        self.assertEqual(manifest, actual)
        self.assertEqual(2, payload["next_sample"])
        resumed, resumed_opt = self.model_optimizer()  # consumes RNG deliberately
        restore_checkpoint(resumed, resumed_opt, payload)
        losses = list(payload["losses"])
        self.steps(resumed, resumed_opt, payload["step"], 5, losses)
        self.assert_tree_equal(continuous.state_dict(), resumed.state_dict())
        self.assert_tree_equal(continuous_opt.state_dict(), resumed_opt.state_dict())
        self.assertEqual(reference, losses)
        self.assert_tree_equal(
            expected_rng, (random.random(), float(np.random.random()), torch.rand(4))
        )

    def test_corrupt_payload_rejected(self):
        _, _, manifest = self.saved()
        path = self.directory / "checkpoints" / manifest["name"] / "state.pt"
        with path.open("ab") as stream:
            stream.write(b"damage")
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_checkpoint(self.directory, self.ident)

    def test_commit_mismatch_rejected(self):
        _, _, manifest = self.saved()
        path = self.directory / "checkpoints" / manifest["name"] / "commit.json"
        atomic_json(path, {**manifest, "step": 1})
        with self.assertRaisesRegex(ValueError, "commit"):
            read_checkpoint(self.directory, self.ident)

    def test_uncommitted_pending_never_selected(self):
        _, _, manifest = self.saved()
        (self.directory / "checkpoints" / "step-000003-dead.pending").mkdir()
        _, selected = read_checkpoint(self.directory, self.ident)
        self.assertEqual(manifest, selected)

    def test_pointer_traversal_rejected(self):
        self.saved()
        atomic_json(
            self.directory / "checkpoints" / "latest.json", {"name": "../../outside"}
        )
        with self.assertRaisesRegex(ValueError, "pointer"):
            read_checkpoint(self.directory, self.ident)

    def test_changed_training_identity_rejected(self):
        self.saved()
        bad = copy.deepcopy(self.ident)
        bad["dataset_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "identity"):
            read_checkpoint(self.directory, bad)

    def test_missing_optimizer_moment_rejected(self):
        model, optimizer, _ = self.saved()
        payload, _ = read_checkpoint(self.directory, self.ident)
        payload["optimizer"]["state"][0].pop("exp_avg")
        with self.assertRaisesRegex(ValueError, "moments"):
            restore_checkpoint(model, optimizer, payload)

    def test_factor_shape_rejected_before_copy(self):
        model, optimizer, _ = self.saved()
        before = copy.deepcopy(model.state_dict())
        payload, _ = read_checkpoint(self.directory, self.ident)
        payload["factors"]["0.weight"] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "shape"):
            restore_checkpoint(model, optimizer, payload)
        self.assert_tree_equal(before, model.state_dict())

    def test_different_optimizer_hyperparameters_rejected(self):
        model, _, _ = self.saved()
        payload, _ = read_checkpoint(self.directory, self.ident)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        with self.assertRaisesRegex(ValueError, "configuration"):
            restore_checkpoint(model, optimizer, payload)

    def test_versions_rejected(self):
        model, optimizer, _ = self.saved()
        payload, _ = read_checkpoint(self.directory, self.ident)
        payload["versions"]["torch"] = "different"
        with self.assertRaisesRegex(ValueError, "versions"):
            restore_checkpoint(model, optimizer, payload)

    def test_nonfinite_checkpoint_never_committed(self):
        model, optimizer = self.model_optimizer()
        with torch.no_grad():
            model[0].weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            save_checkpoint(
                self.directory,
                model,
                optimizer,
                self.ident,
                step=1,
                sample_count=1,
                losses=[1.0],
                initial_digest="initial",
            )
        self.assertFalse((self.directory / "checkpoints" / "latest.json").exists())

    def test_pause_is_request_not_gpu_release_and_old_run_ignored(self):
        self.store.transition("resume1", "running", run_id="run1")
        with self.store.lock("resume1"):
            result = request_pause(self.store, "resume1")
        self.assertFalse(result["gpu_released"])
        self.assertEqual("running", self.store.read("resume1")["state"])
        self.assertTrue(pause_requested(self.directory, "run1"))
        self.assertFalse(pause_requested(self.directory, "run2"))

    def test_pause_queued_rejected(self):
        with self.assertRaisesRegex(ValueError, "running"):
            request_pause(self.store, "resume1")

    def test_running_job_pace_update_is_run_scoped(self):
        self.store.transition("resume1", "running", run_id="run1")
        accepted = request_step_interval(self.store, "resume1", 2.5)
        self.assertEqual(2.5, accepted["step_interval_seconds"])
        self.assertEqual(
            2.5, requested_step_interval(self.directory, "run1", default=1.0)
        )
        self.assertEqual(
            1.0, requested_step_interval(self.directory, "new-run", default=1.0)
        )

    def test_invalid_or_nonrunning_pace_update_rejected(self):
        for value in (-1, 61, True, float("nan")):
            with self.assertRaisesRegex(ValueError, "step interval"):
                request_step_interval(self.store, "resume1", value)
        with self.assertRaisesRegex(ValueError, "running"):
            request_step_interval(self.store, "resume1", 2.0)

    def test_resume_queued_never_spawns(self):
        with patch("sglang.srt.lora.training.runtime.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "paused"):
                resume_training(self.store, "resume1")
            spawn.assert_not_called()
        self.assertEqual("queued", self.store.read("resume1")["state"])

    def test_paused_cannot_publish(self):
        self.store.transition("resume1", "running")
        self.store.transition("resume1", "paused")
        with self.assertRaises(ValueError):
            self.store.transition("resume1", "publishing")

    def managed_pause(self, gpu_after, outcome_run_id=None):
        # New managed job, fake worker exit but real ownership and state sequencing.
        spec = TrainingSpec(
            "runtime1", "runtime@v1", str(self.model_path), str(self.data), 4
        )
        manifest = {"step": 1, "name": "test", "state_sha256": "test"}

        def spawn(*_, **__):
            job = self.store.read("runtime1")
            atomic_json(
                self.store.directory("runtime1") / "worker_outcome.json",
                {
                    "run_id": outcome_run_id or job["run_id"],
                    "state": "paused",
                    "checkpoint": manifest,
                },
            )
            return Mock(pid=123456, wait=Mock(return_value=0))

        with patch(
            "sglang.srt.lora.training.runtime.ensure_gpu_idle",
            side_effect=[{}, gpu_after],
        ), patch(
            "sglang.srt.lora.training.runtime.subprocess.Popen", side_effect=spawn
        ), patch(
            "sglang.srt.lora.training.runtime.read_checkpoint",
            return_value=({"step": 1}, manifest),
        ), patch(
            "sglang.srt.lora.training.runtime.stop_owned_process"
        ):
            return execute_training(self.store, spec)

    def test_paused_only_after_worker_exit_and_gpu_idle(self):
        result = self.managed_pause({"free_mib": 81076})
        self.assertEqual("paused", result["state"])
        self.assertTrue(result["gpu_released"])

    def test_remaining_gpu_process_cannot_mark_paused(self):
        with self.assertRaisesRegex(RuntimeError, "occupied"):
            self.managed_pause(RuntimeError("occupied"))
        self.assertEqual("training_failed", self.store.read("runtime1")["state"])
        self.assertFalse(self.store.read("runtime1")["gpu_released"])

    def test_stale_worker_outcome_cannot_mark_paused(self):
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.managed_pause({}, outcome_run_id="previous-run")
        self.assertEqual("training_failed", self.store.read("runtime1")["state"])

    def test_duplicate_pause_when_paused_is_idempotent(self):
        self.store.transition("resume1", "running")
        self.store.transition("resume1", "paused")
        before = self.store.read("resume1")
        self.assertTrue(request_pause(self.store, "resume1")["already_paused"])
        self.assertEqual(before, self.store.read("resume1"))

    def test_dataset_mutation_resume_rejected_before_spawn(self):
        _, _, manifest = self.saved()
        self.store.transition("resume1", "running")
        self.store.transition("resume1", "paused", checkpoint=manifest)
        self.data.write_text('{"prompt":"changed", "response":"A"}\n')
        with patch("sglang.srt.lora.training.runtime.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "dataset"):
                resume_training(self.store, "resume1")
            spawn.assert_not_called()
        self.assertEqual("paused", self.store.read("resume1")["state"])

    def test_resume_occupied_gpu_keeps_checkpoint_retryable(self):
        _, _, manifest = self.saved()
        self.store.transition("resume1", "running")
        self.store.transition("resume1", "paused", checkpoint=manifest)
        with patch(
            "sglang.srt.lora.training.runtime.ensure_gpu_idle",
            side_effect=RuntimeError("occupied"),
        ), patch("sglang.srt.lora.training.runtime.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                resume_training(self.store, "resume1")
            spawn.assert_not_called()
        self.assertEqual("paused", self.store.read("resume1")["state"])
        self.assertEqual(manifest, self.store.read("resume1")["checkpoint"])


if __name__ == "__main__":
    unittest.main()

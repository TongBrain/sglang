"""CPU-only contracts for opt-in training and publication; no GPU allocation."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.lora.training.jobs import (
    FileLease,
    JobStore,
    TrainingSpec,
    read_samples,
)
from sglang.srt.lora.training.peft_worker import (
    encode_sample,
    load_initial_adapter,
    trainable_digest,
)
from sglang.srt.lora.training.publish import InferenceClient, publish_job
from sglang.srt.lora.training.runtime import execute_training


class FakeTokenizer:
    eos_token_id = 1

    def encode(self, value, **_):
        return [2 + ord(character) % 20 for character in value]


class FakeClient:
    endpoint = "http://127.0.0.1:13818"

    def __init__(self, model, name):
        self.model = str(model)
        self.name = name
        self.calls = []
        self.adapters = {
            "old@v1": {"lora_id": "old-id", "loaded": True, "resident": True}
        }
        self.fail_prewarm = False
        self.identity_changes = False
        self.abort_canary = False
        self.fail_cleanup = False

    def status(self, name=None):
        return {"success": True, "adapters": dict(self.adapters)}

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/get_model_info":
            return {"model_path": self.model}
        if path == "/load_lora_adapter":
            self.adapters[self.name] = {
                "lora_id": "new-id",
                "loaded": True,
                "resident": True,
            }
            return {"success": True, "lora_id": "new-id"}
        if path == "/prewarm_lora_adapter":
            if self.identity_changes:
                self.adapters[self.name]["lora_id"] = "someone-else"
            if self.fail_prewarm:
                raise RuntimeError("prewarm failed")
            return {**self.status(), "probe_completed": True}
        if path == "/generate":
            return {
                "meta_info": {
                    "finish_reason": {
                        "type": "abort" if self.abort_canary else "length"
                    },
                    "completion_tokens": 16,
                }
            }
        if path == "/unload_lora_adapter":
            if self.fail_cleanup:
                raise RuntimeError("cleanup failed")
            self.adapters.pop(body["lora_name"], None)
            return {"success": True}
        raise AssertionError(path)


class TestTrainServeLoop(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.data = self.root / "data.jsonl"
        self.data.write_text(json.dumps({"prompt": "Q:", "response": "A"}) + "\n")
        self.spec = TrainingSpec("job1", "user@v2", str(self.model), str(self.data), 4)
        self.store = JobStore(self.root / "jobs")

    def trained_job(self):
        self.store.create(self.spec)
        self.store.transition("job1", "running")
        self.artifact = {
            "path": str(self.store.directory("job1") / "v2"),
            "files": {"weights": "hash"},
        }
        self.store.transition("job1", "trained", artifact=self.artifact)
        return FakeClient(self.model, self.spec.adapter_name)

    def publish(self, client):
        with patch(
            "sglang.srt.lora.training.publish.validate_artifact",
            return_value=self.artifact,
        ):
            return publish_job(self.store, "job1", client, [2, 3], "test prompt")

    def test_safe_job_path(self):
        for identifier in ("../escape", "/root", ".", "..", "a/b"):
            with self.assertRaises(ValueError):
                self.store.directory(identifier)

    def test_duplicate_job_does_not_overwrite(self):
        self.store.create(self.spec)
        before = self.store.read("job1")
        with self.assertRaises(FileExistsError):
            self.store.create(self.spec)
        self.assertEqual(before, self.store.read("job1"))

    def test_explicit_state_machine(self):
        self.store.create(self.spec)
        with self.assertRaises(ValueError):
            self.store.transition("job1", "inference_ready")
        self.store.transition("job1", "running")
        self.store.transition("job1", "trained")
        self.assertEqual("trained", self.store.read("job1")["state"])

    def test_lease_excludes_second_owner(self):
        with FileLease(self.root / "lease"):
            with self.assertRaises(RuntimeError):
                with FileLease(self.root / "lease"):
                    self.fail("acquired twice")
        with FileLease(self.root / "lease"):
            pass

    def test_prompt_is_masked_response_retained(self):
        ids, labels = encode_sample(
            FakeTokenizer(), {"prompt": "Q:", "response": "answer"}, 8
        )
        self.assertEqual([-100, -100], labels[:2])
        self.assertEqual(ids[2:], labels[2:])
        self.assertEqual(1, ids[-1])

    def test_prompt_cannot_consume_response_budget(self):
        with self.assertRaises(ValueError):
            encode_sample(
                FakeTokenizer(), {"prompt": "long prompt", "response": "A"}, 8
            )

    def test_empty_dataset_rejected_before_create(self):
        self.data.write_text("")
        with self.assertRaises(ValueError):
            self.store.create(self.spec)
        self.assertFalse(self.store.directory("job1").exists())

    def test_malformed_dataset_rejected(self):
        self.data.write_text('{"prompt":"q","response":""}\n')
        with self.assertRaises(ValueError):
            read_samples(self.data)

    def test_training_step_interval_is_bounded_and_default_is_full_speed(self):
        self.assertEqual(0.0, self.spec.step_interval_seconds)
        for value in (-0.01, 60.01, True, "1"):
            invalid = TrainingSpec(
                "job2",
                "user@v3",
                str(self.model),
                str(self.data),
                4,
                step_interval_seconds=value,
            )
            with self.assertRaisesRegex(ValueError, "step_interval_seconds"):
                invalid.validate()
        TrainingSpec(
            "job2",
            "user@v3",
            str(self.model),
            str(self.data),
            4,
            step_interval_seconds=0.8,
        ).validate()

    def test_step_permit_requires_loopback_load_endpoint_and_bounded_controls(self):
        accepted = TrainingSpec(
            "permit-job",
            "user@permit",
            str(self.model),
            str(self.data),
            4,
            inference_load_url="http://127.0.0.1:30000/v1/loads?include=core",
        )
        accepted.validate()
        for url in (
            "https://127.0.0.1:30000/v1/loads?include=core",
            "http://example.com/v1/loads?include=core",
            "http://127.0.0.1:30000/get_load",
            "http://127.0.0.1:30000/v1/loads",
        ):
            with self.subTest(url=url):
                invalid = TrainingSpec(
                    "permit-job",
                    "user@permit",
                    str(self.model),
                    str(self.data),
                    4,
                    inference_load_url=url,
                )
                with self.assertRaisesRegex(ValueError, "inference_load_url"):
                    invalid.validate()
        for field, value in (
            ("permit_idle_seconds", -0.1),
            ("permit_poll_seconds", 0.0),
            ("permit_request_timeout_seconds", 0.0),
            ("permit_max_consecutive_errors", 0),
        ):
            with self.subTest(field=field):
                kwargs = {field: value}
                invalid = TrainingSpec(
                    "permit-job",
                    "user@permit",
                    str(self.model),
                    str(self.data),
                    4,
                    **kwargs,
                )
                with self.assertRaisesRegex(ValueError, field):
                    invalid.validate()

    def test_occupied_gpu_never_spawns(self):
        with patch(
            "sglang.srt.lora.training.runtime.ensure_gpu_idle",
            side_effect=RuntimeError("occupied"),
        ), patch("sglang.srt.lora.training.runtime.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                execute_training(self.store, self.spec)
            spawn.assert_not_called()
        self.assertEqual("training_failed", self.store.read("job1")["state"])

    def test_timeout_stops_only_owned_process_group(self):
        process = Mock(pid=123456)
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 1), 0]
        with patch(
            "sglang.srt.lora.training.runtime.ensure_gpu_idle", return_value={}
        ), patch(
            "sglang.srt.lora.training.runtime.subprocess.Popen", return_value=process
        ), patch(
            "sglang.srt.lora.training.runtime.os.killpg"
        ) as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                execute_training(self.store, self.spec)
            self.assertEqual(123456, kill.call_args.args[0])
            self.assertEqual(1, kill.call_count)
        self.assertEqual("training_failed", self.store.read("job1")["state"])

    def test_missing_load_identity_requires_reconciliation(self):
        client = self.trained_job()
        real_call = client.call

        def missing_identity(method, path, body=None):
            value = real_call(method, path, body)
            return {"success": True} if path == "/load_lora_adapter" else value

        client.call = missing_identity
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.publish(client)
        self.assertTrue(self.store.read("job1")["reconcile_required"])
        self.assertFalse(self.store.read("job1")["cleanup"]["attempted"])

    def test_success_marks_readiness_not_business_switch(self):
        client = self.trained_job()
        result = self.publish(client)
        self.assertEqual("inference_ready", result["state"])
        self.assertFalse(result["publication"]["business_routing_changed"])
        self.assertIn("old@v1", client.adapters)
        self.assertEqual("new-id", result["publication"]["lora_id"])

    def test_same_name_is_not_replaced(self):
        client = self.trained_job()
        client.adapters[self.spec.adapter_name] = {"lora_id": "already-there"}
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.publish(client)
        self.assertEqual(
            "already-there", client.adapters[self.spec.adapter_name]["lora_id"]
        )
        self.assertFalse(
            any(path == "/load_lora_adapter" for _, path, _ in client.calls)
        )

    def test_prewarm_failure_removes_only_our_new_adapter(self):
        client = self.trained_job()
        client.fail_prewarm = True
        with self.assertRaisesRegex(RuntimeError, "prewarm failed"):
            self.publish(client)
        self.assertEqual({"old@v1"}, set(client.adapters))
        self.assertEqual("publish_failed", self.store.read("job1")["state"])
        self.assertTrue(self.store.read("job1")["cleanup"]["success"])
        self.assertIn("artifact", self.store.read("job1"))

    def test_cleanup_failure_is_visible(self):
        client = self.trained_job()
        client.fail_prewarm = client.fail_cleanup = True
        with self.assertRaises(RuntimeError):
            self.publish(client)
        self.assertFalse(self.store.read("job1")["cleanup"]["success"])

    def test_changed_identity_is_not_unloaded(self):
        client = self.trained_job()
        client.fail_prewarm = client.identity_changes = True
        with self.assertRaises(RuntimeError):
            self.publish(client)
        self.assertEqual(
            "someone-else", client.adapters[self.spec.adapter_name]["lora_id"]
        )
        self.assertFalse(
            any(path == "/unload_lora_adapter" for _, path, _ in client.calls)
        )

    def test_abort_canary_rejects_publication(self):
        client = self.trained_job()
        client.abort_canary = True
        with self.assertRaisesRegex(RuntimeError, "canary"):
            self.publish(client)
        self.assertEqual("publish_failed", self.store.read("job1")["state"])

    def test_model_mismatch_rejected_before_load(self):
        client = self.trained_job()
        client.model = "/different/model"
        with self.assertRaisesRegex(ValueError, "base model"):
            self.publish(client)
        self.assertFalse(
            any(path == "/load_lora_adapter" for _, path, _ in client.calls)
        )

    def test_artifact_mutation_rejected(self):
        client = self.trained_job()
        with patch(
            "sglang.srt.lora.training.publish.validate_artifact",
            return_value={"files": {"weights": "changed"}},
        ):
            with self.assertRaisesRegex(ValueError, "changed"):
                publish_job(self.store, "job1", client, [2], "canary")
        self.assertFalse(client.calls)

    def test_no_publish_before_training(self):
        self.store.create(self.spec)
        with self.assertRaises(ValueError):
            publish_job(
                self.store, "job1", FakeClient(self.model, "user@v2"), [2], "canary"
            )

    def test_failed_publish_can_retry_with_same_immutable_asset(self):
        client = self.trained_job()
        client.fail_prewarm = True
        with self.assertRaises(RuntimeError):
            self.publish(client)
        client.fail_prewarm = False
        ready = self.publish(client)
        self.assertEqual("inference_ready", ready["state"])
        self.assertIsNone(ready["error"])
        self.assertFalse(ready["reconcile_required"])
        self.assertTrue(
            any(event.get("error") == "prewarm failed" for event in ready["events"])
        )

    def test_endpoint_credentials_are_not_accepted(self):
        with self.assertRaises(ValueError):
            InferenceClient("http://user:password@host")

    def test_real_tiny_peft_export_and_weight_update(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from sglang.srt.lora.training.artifacts import validate_artifact

        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        config.save_pretrained(self.model)
        model = get_peft_model(
            Qwen2ForCausalLM(config),
            LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        model.peft_config["default"].base_model_name_or_path = str(self.model)
        before = trainable_digest(model)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        ids = torch.tensor([[2, 3, 4, 5]])
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        self.assertNotEqual(before, trainable_digest(model))
        self.assertTrue(
            all(
                "lora_" in name
                for name, p in model.named_parameters()
                if p.requires_grad
            )
        )
        export = self.root / "adapter"
        model.save_pretrained(export, safe_serialization=True)
        self.assertEqual(4, validate_artifact(export, self.spec)["tensor_count"])
        initial_digest = trainable_digest(model)
        copied = get_peft_model(
            Qwen2ForCausalLM(config),
            LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        self.assertNotEqual(initial_digest, trainable_digest(copied))
        load_initial_adapter(copied, export)
        self.assertEqual(initial_digest, trainable_digest(copied))
        bad = json.loads((export / "adapter_config.json").read_text())
        bad["r"] = 16
        (export / "adapter_config.json").write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            validate_artifact(export, self.spec)


if __name__ == "__main__":
    unittest.main()

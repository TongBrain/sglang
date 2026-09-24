"""Local BF16 SFT worker; prompt masking; only PEFT factors are optimized."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from .checkpoint import (
    identity,
    pause_requested,
    read_checkpoint,
    requested_step_interval,
    restore_checkpoint,
    save_checkpoint,
)
from .jobs import TrainingSpec, atomic_json, file_sha256, read_samples
from .permit import wait_for_step_permit


def encode_sample(tokenizer, sample, max_length):
    prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False)
    answer = tokenizer.encode(sample["response"], add_special_tokens=False)
    if not prompt or not answer:
        raise ValueError("prompt/response must tokenize to nonempty sequences")
    if len(prompt) >= max_length - 1:
        raise ValueError("prompt leaves no room for supervised response")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define an EOS token")
    ids = prompt + answer[: max_length - len(prompt) - 1] + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt) + ids[len(prompt) :]
    return ids, labels


def trainable_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_initial_adapter(model, directory):
    """Exact attention-only A/B copy; no version-specific HF MoE conversion."""
    import torch
    from safetensors.torch import load_file

    tensors = load_file(
        str(Path(directory) / "adapter_model.safetensors"), device="cpu"
    )
    mapped = {}
    for name, tensor in tensors.items():
        destination = name.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        if destination == name or destination in mapped:
            raise ValueError("unsupported/duplicate initial adapter tensor")
        mapped[destination] = tensor
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(parameters) != set(mapped):
        raise ValueError(
            "initial adapter tensors do not exactly match trainable LoRA factors"
        )
    # Validate ALL shapes before copying any parameter.
    if any(
        tuple(parameters[name].shape) != tuple(tensor.shape)
        or not torch.isfinite(tensor).all().item()
        for name, tensor in mapped.items()
    ):
        raise ValueError("invalid initial adapter factors")
    with torch.no_grad():
        for name, tensor in mapped.items():
            parameters[name].copy_(
                tensor.to(device=parameters[name].device, dtype=parameters[name].dtype)
            )


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.job_dir).resolve(strict=True)
    job = json.loads((directory / "job.json").read_text())
    spec = TrainingSpec(**job["spec"]).validate()
    if file_sha256(spec.dataset_path) != job["dataset_sha256"]:
        raise ValueError("dataset changed after job submission")
    if spec.initial_adapter:
        expected_files = job["initial_artifact"]["files"]
        if any(
            file_sha256(Path(spec.initial_adapter) / name) != digest
            for name, digest in expected_files.items()
        ):
            raise ValueError("initial adapter changed after job admission")
    samples = read_samples(spec.dataset_path)
    torch.set_num_threads(4)
    import numpy as np

    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker requires exactly one assigned visible GPU")
    tokenizer = AutoTokenizer.from_pretrained(spec.model_path, local_files_only=True)
    batches = [encode_sample(tokenizer, sample, spec.max_length) for sample in samples]
    print(json.dumps({"stage": "loading_model", "samples": len(batches)}), flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        spec.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=spec.rank,
            lora_alpha=spec.alpha,
            lora_dropout=0.0,
            target_modules=list(spec.target_modules),
            bias="none",
        ),
    )
    if spec.initial_adapter:
        load_initial_adapter(model, spec.initial_adapter)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(
        ".lora_A." not in name and ".lora_B." not in name for name, _ in trainable
    ):
        raise RuntimeError("only LoRA A/B parameters may be trainable")
    # This immutable v1 is the starting adapter; with no initial_adapter it is
    # freshly initialized, NOT an existing user's expert adapter.
    resuming = job.get("resume_checkpoint") is not None
    if not resuming:
        model.save_pretrained(directory / "v1", safe_serialization=True)
    before = trainable_digest(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=spec.learning_rate
    )
    losses = []
    first_step = 0
    training_identity = identity(spec, job)
    if resuming:
        payload, manifest = read_checkpoint(directory, training_identity)
        if manifest != job["resume_checkpoint"] or payload["sample_count"] != len(
            batches
        ):
            raise ValueError("resume checkpoint/data count changed")
        restore_checkpoint(model, optimizer, payload)
        first_step = payload["step"]
        losses = list(payload["losses"])
        before = payload["initial_digest"]
        print(
            json.dumps(
                {
                    "stage": "checkpoint_restored",
                    "step": first_step,
                    "checkpoint": manifest,
                }
            ),
            flush=True,
        )

    def commit_pause(completed_steps):
        optimizer.zero_grad(set_to_none=True)
        manifest = save_checkpoint(
            directory,
            model,
            optimizer,
            training_identity,
            step=completed_steps,
            sample_count=len(batches),
            losses=losses,
            initial_digest=before,
        )
        atomic_json(
            directory / "worker_outcome.json",
            {"run_id": job["run_id"], "state": "paused", "checkpoint": manifest},
        )
        print(
            json.dumps({"stage": "checkpoint_committed", "checkpoint": manifest}),
            flush=True,
        )

    model.train()
    start = time.monotonic()
    step_interval_seconds = requested_step_interval(
        directory, job["run_id"], spec.step_interval_seconds
    )
    for step in range(first_step, spec.max_steps):
        permit = wait_for_step_permit(
            directory,
            job["run_id"],
            load_url=spec.inference_load_url,
            idle_seconds=spec.permit_idle_seconds,
            poll_seconds=spec.permit_poll_seconds,
            request_timeout_seconds=spec.permit_request_timeout_seconds,
            max_consecutive_errors=spec.permit_max_consecutive_errors,
            stop_requested=lambda: pause_requested(directory, job["run_id"]),
        )
        if not permit["granted"]:
            commit_pause(step)
            return
        print(json.dumps({"stage": "training_permit", **permit}), flush=True)
        step_interval_seconds = requested_step_interval(
            directory, job["run_id"], step_interval_seconds
        )
        step_started = time.monotonic()
        ids, labels = batches[step % len(batches)]
        input_ids = torch.tensor([ids], device="cuda:0")
        targets = torch.tensor([labels], device="cuda:0")
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, labels=targets, use_cache=False)
        loss = output.loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        if any(
            parameter.grad is not None
            and not torch.isfinite(parameter.grad).all().item()
            for _, parameter in trainable
        ):
            raise RuntimeError("nonfinite adapter gradients")
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in trainable], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        progress = {
            "step": step + 1,
            "total_steps": spec.max_steps,
            "loss": value,
            "elapsed_seconds": time.monotonic() - start,
            "step_compute_seconds": time.monotonic() - step_started,
            "step_interval_seconds": step_interval_seconds,
            "permit_wait_seconds": permit["wait_seconds"],
            "permit_busy_observations": permit["busy_observations"],
            "permit_granted_count": permit["granted_count"],
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        }
        atomic_json(directory / "progress.json", progress)
        print(json.dumps({"stage": "training_step", **progress}), flush=True)
        if step + 1 < spec.max_steps and pause_requested(directory, job["run_id"]):
            commit_pause(step + 1)
            return
        # Retain the loaded model while yielding GPU compute between steps.
        # Zero preserves the historical full-speed behavior.
        if step + 1 < spec.max_steps and step_interval_seconds:
            deadline = step_started + step_interval_seconds
            while time.monotonic() < deadline:
                if pause_requested(directory, job["run_id"]):
                    commit_pause(step + 1)
                    return
                step_interval_seconds = requested_step_interval(
                    directory, job["run_id"], step_interval_seconds
                )
                deadline = step_started + step_interval_seconds
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.05, remaining))
    after = trainable_digest(model)
    if before == after:
        raise RuntimeError("training did not change LoRA parameters")
    optimizer.zero_grad(set_to_none=True)
    final_checkpoint = save_checkpoint(
        directory,
        model,
        optimizer,
        training_identity,
        step=spec.max_steps,
        sample_count=len(batches),
        losses=losses,
        initial_digest=before,
    )
    # Inference export remains adapter-only; optimizer/RNG live in checkpoints/.
    model.eval()
    pending = directory / "v2.pending"
    model.save_pretrained(pending, safe_serialization=True)
    if (directory / "v2").exists():
        raise RuntimeError("immutable v2 already exists")
    pending.rename(directory / "v2")
    atomic_json(
        directory / "training_result.json",
        {
            "backend": "transformers_peft",
            "steps": spec.max_steps,
            "losses": losses,
            "weights_changed": before != after,
            "initial_weights_sha256": before,
            "final_weights_sha256": after,
            "trainable_parameters": sum(p.numel() for _, p in trainable),
            "base_parameters_frozen": True,
            "initial_adapter_source": spec.initial_adapter or "fresh_attention_lora",
            "max_length": spec.max_length,
            "step_interval_seconds": spec.step_interval_seconds,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "elapsed_seconds": time.monotonic() - start,
            "quality_evaluation": "not_performed",
            "checkpoint_available": True,
            "final_checkpoint": final_checkpoint,
            "resumed_from_step": first_step,
            "elapsed_seconds_scope": "current_worker_training_segment",
        },
    )
    atomic_json(
        directory / "worker_outcome.json", {"run_id": job["run_id"], "state": "trained"}
    )


if __name__ == "__main__":
    main()

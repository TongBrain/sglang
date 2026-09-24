"""Opt-in single-GPU continuous versus pause/resume test; no production service."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from sglang.srt.lora.training.checkpoint import identity, read_checkpoint, request_pause
from sglang.srt.lora.training.jobs import (
    FileLease,
    JobStore,
    TrainingSpec,
    atomic_json,
    ensure_gpu_idle,
)
from sglang.srt.lora.training.runtime import execute_training, resume_training

REPO = Path(__file__).resolve().parents[3]
ORIGINAL = Path(os.environ.get("SGLANG_LORA_TEST_ORIGINAL_REPO", str(REPO)))
MODEL = os.environ.get("SGLANG_LORA_TEST_MODEL", "/path/to/Qwen1.5-MoE-A2.7B")
GPU = int(os.environ.get("SGLANG_LORA_TEST_GPU", "0"))
CORE = [
    "python/sglang/srt/entrypoints/http_server.py",
    "python/sglang/srt/managers/io_struct.py",
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/tokenizer_control_mixin.py",
    "python/sglang/srt/lora/lora_manager.py",
    "python/sglang/srt/lora/lora_registry.py",
]


def hashes(root):
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in CORE
    }


def same_tree(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            same_tree(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            same_tree(a, b) for a, b in zip(left, right)
        )
    return left == right


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = REPO / "work" / f"checkpoint-live-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "synthetic.jsonl"
    samples = [
        {
            "prompt": "Question: What does LoRA train?\nAnswer:",
            "response": " LoRA updates low-rank factors and freezes base weights.",
        },
        {
            "prompt": "Question: When is a training GPU released?\nAnswer:",
            "response": " After a checkpoint is committed and the worker has exited.",
        },
        {
            "prompt": "Question: What is needed for resume?\nAnswer:",
            "response": " Adapter factors, optimizer, random state and data progress.",
        },
    ]
    dataset.write_text("".join(json.dumps(item) + "\n" for item in samples))
    store = JobStore(output / "jobs")
    spec = TrainingSpec(
        f"continuous-{stamp}",
        f"continuous@{stamp}",
        MODEL,
        str(dataset),
        GPU,
        max_steps=5,
        timeout_seconds=300,
    )
    paused_spec = replace(spec, job_id=f"pause-{stamp}", adapter_name=f"pause@{stamp}")
    before = {"original": hashes(ORIGINAL), "independent_inference": hashes(REPO)}
    result = {
        "passed": False,
        "output": str(output),
        "gpu": GPU,
        "comparison_rule": "exact tensor/optimizer/loss equality; no relaxed tolerance",
        "simultaneous_train_infer": False,
    }
    atomic_json(
        output / "specs.json",
        {"baseline": spec.__dict__, "pause_resume": paused_spec.__dict__},
    )
    try:
        ensure_gpu_idle(GPU, 50000)
        baseline = execute_training(store, spec)
        result["baseline"] = baseline
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(execute_training, store, paused_spec)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if future.done():
                    future.result()
                    raise RuntimeError("training completed before pause request")
                try:
                    job = store.read(paused_spec.job_id)
                except FileNotFoundError:
                    time.sleep(0.05)
                    continue
                if job["state"] == "running":
                    result["pause_request"] = request_pause(store, paused_spec.job_id)
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("job did not enter running before request deadline")
            paused = future.result(timeout=330)
        if (
            paused["state"] != "paused"
            or paused["completed_steps"] != 1
            or not paused["gpu_released"]
        ):
            raise RuntimeError(
                "early pause did not commit exactly one step and release GPU"
            )
        result["paused"] = paused
        # A separate short-lived CUDA process proves another managed task can use the gap.
        with FileLease(store.root / ".gpu-leases" / f"gpu-{GPU}.lock"):
            result["gap_gpu_idle"] = ensure_gpu_idle(GPU, 50000)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU))
            code = "import torch; x=torch.ones(1024, device='cuda'); assert x.sum().item()==1024; torch.cuda.synchronize(); print('gap_cuda_task_completed')"
            result["gap_task"] = subprocess.check_output(
                [sys.executable, "-c", code], env=env, text=True, timeout=40
            ).strip()
            result["gap_gpu_released"] = ensure_gpu_idle(GPU, 50000)
        resumed = resume_training(store, paused_spec.job_id)
        result["resumed"] = resumed
        if (
            resumed["training_result"]["resumed_from_step"] != 1
            or resumed["completed_steps"] != 5
        ):
            raise RuntimeError("resume did not continue from the checkpoint step")
        reference, _ = read_checkpoint(
            store.directory(spec.job_id), identity(spec, baseline)
        )
        restored, _ = read_checkpoint(
            store.directory(paused_spec.job_id), identity(paused_spec, resumed)
        )
        result["exact_comparison"] = {
            key: same_tree(reference[key], restored[key])
            for key in ("factors", "optimizer", "losses", "rng", "step", "next_sample")
        }
        result["weight_fingerprints_equal"] = (
            baseline["training_result"]["final_weights_sha256"]
            == resumed["training_result"]["final_weights_sha256"]
        )
        if (
            not all(result["exact_comparison"].values())
            or not result["weight_fingerprints_equal"]
        ):
            raise RuntimeError("continuous/resumed training are not exactly identical")
        result["passed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["source_unchanged"] = before == {
            "original": hashes(ORIGINAL),
            "independent_inference": hashes(REPO),
        }
        try:
            result["final_gpu"] = ensure_gpu_idle(GPU, 50000)
            result["gpu_released"] = True
        except Exception as error:
            result["gpu_released"] = False
            result["gpu_check_error"] = repr(error)
        result["passed"] = (
            result["passed"] and result["source_unchanged"] and result["gpu_released"]
        )
        atomic_json(output / "result.json", result)
        print(
            json.dumps(
                {"result": str(output / "result.json"), "passed": result["passed"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Bounded subprocess execution. Never terminate unrelated GPU processes."""

import json
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import asdict

from .artifacts import validate_artifact
from .checkpoint import identity, read_checkpoint
from .jobs import (
    FileLease,
    JobStore,
    TrainingSpec,
    atomic_json,
    ensure_gpu_idle,
    file_sha256,
)


def stop_owned_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=20)


def execute_training(store: JobStore, spec: TrainingSpec, *, resume=False):
    if not resume:
        store.create(spec)
    directory = store.directory(spec.job_id)
    process = None
    with store.lock(spec.job_id):
        job = store.read(spec.job_id)
        if resume:
            if (
                job["state"] != "paused"
                or json.loads(json.dumps(asdict(spec))) != job["spec"]
            ):
                raise ValueError("resume requires the original spec of a paused job")
            if file_sha256(spec.dataset_path) != job["dataset_sha256"]:
                raise ValueError("dataset changed before resume")
            payload, manifest = read_checkpoint(directory, identity(spec, job))
            if manifest != job["checkpoint"] or payload["step"] >= spec.max_steps:
                raise ValueError("paused checkpoint changed or has no remaining steps")
        run_id = uuid.uuid4().hex
        try:
            initial = (
                validate_artifact(spec.initial_adapter, spec)
                if spec.initial_adapter
                else None
            )
            if resume and initial != job.get("initial_artifact"):
                raise ValueError("initial adapter changed before resume")
            with FileLease(store.root / ".gpu-leases" / f"gpu-{spec.gpu}.lock"):
                gpu = ensure_gpu_idle(spec.gpu, spec.minimum_free_mib)
                store.transition(
                    spec.job_id,
                    "running",
                    gpu_observation=gpu,
                    initial_artifact=initial,
                    run_id=run_id,
                    resume_checkpoint=manifest if resume else None,
                    error=None,
                    gpu_released=False,
                )
                env = dict(os.environ)
                env.update(
                    CUDA_VISIBLE_DEVICES=str(spec.gpu),
                    PYTHONDONTWRITEBYTECODE="1",
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false",
                    OMP_NUM_THREADS="4",
                    PYTHONUNBUFFERED="1",
                )
                with (directory / "train.log").open(
                    "a" if resume else "x", encoding="utf-8"
                ) as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "sglang.srt.lora.training.peft_worker",
                            "--job-dir",
                            str(directory),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    atomic_json(
                        directory / "process.json",
                        {
                            "pid": process.pid,
                            "command_backend": "peft",
                            "run_id": run_id,
                        },
                    )
                    try:
                        code = process.wait(timeout=spec.timeout_seconds)
                    except BaseException:
                        # Keep the GPU lease until our worker has actually stopped.
                        stop_owned_process(process)
                        process = None
                        raise
                    if code != 0:
                        stop_owned_process(process)
                        process = None
                        raise RuntimeError(
                            f"training worker exited {code}; inspect {directory / 'train.log'}"
                        )
                outcome = json.loads((directory / "worker_outcome.json").read_text())
                if outcome.get("run_id") != run_id:
                    raise RuntimeError("stale/missing worker outcome")
                if outcome.get("state") == "paused":
                    payload, manifest = read_checkpoint(
                        directory, identity(spec, store.read(spec.job_id))
                    )
                    if (
                        manifest != outcome.get("checkpoint")
                        or payload["step"] >= spec.max_steps
                    ):
                        raise RuntimeError("invalid paused worker checkpoint")
                    gpu_after = ensure_gpu_idle(spec.gpu, spec.minimum_free_mib)
                    return store.transition(
                        spec.job_id,
                        "paused",
                        checkpoint=manifest,
                        completed_steps=payload["step"],
                        gpu_after_pause=gpu_after,
                        gpu_released=True,
                    )
                if outcome.get("state") != "trained":
                    raise RuntimeError("invalid worker outcome")
                result = json.loads((directory / "training_result.json").read_text())
                if result.get("steps") != spec.max_steps or not result.get(
                    "weights_changed"
                ):
                    raise RuntimeError(
                        "training worker did not prove completed steps/weight updates"
                    )
                artifact = validate_artifact(directory / "v2", spec)
                gpu_after = ensure_gpu_idle(spec.gpu, spec.minimum_free_mib)
                return store.transition(
                    spec.job_id,
                    "trained",
                    training_result=result,
                    artifact=artifact,
                    gpu_after_training=gpu_after,
                    gpu_released=True,
                    completed_steps=spec.max_steps,
                    checkpoint=result["final_checkpoint"],
                )
        except BaseException as error:
            if process is not None:
                stop_owned_process(process)
            # Admission refusal must leave an intact paused job retryable.
            if store.read(spec.job_id)["state"] in ("queued", "running"):
                store.transition(spec.job_id, "training_failed", error=str(error))
            raise


def resume_training(store: JobStore, job_id):
    spec = TrainingSpec(**store.read(job_id)["spec"]).validate()
    return execute_training(store, spec, resume=True)

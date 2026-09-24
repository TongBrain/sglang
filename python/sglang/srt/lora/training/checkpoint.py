"""Committed single-worker checkpoints, local trusted storage only."""

import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .jobs import atomic_json, file_sha256


def identity(spec, job):
    # GPU placement may change on a future migration; all training semantics stay fixed.
    value = asdict(spec)
    value.pop("gpu")
    value.pop("timeout_seconds")
    value.pop("minimum_free_mib")
    return json.loads(
        json.dumps(
            {
                "spec": value,
                "dataset_sha256": job["dataset_sha256"],
                "model_config_sha256": file_sha256(
                    Path(spec.model_path) / "config.json"
                ),
                "initial_files": (job.get("initial_artifact") or {}).get("files"),
            }
        )
    )


def versions():
    import peft
    import torch
    import transformers

    return {
        "torch": str(torch.__version__),
        "peft": peft.__version__,
        "transformers": transformers.__version__,
    }


def capture_rng():
    import numpy as np
    import torch

    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import numpy as np
    import torch

    cuda = state["cuda"]
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("checkpoint CUDA RNG topology differs")
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def _cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _finite(value):
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def save_checkpoint(
    directory,
    model,
    optimizer,
    training_identity,
    *,
    step,
    sample_count,
    losses,
    initial_digest,
):
    import torch

    if step < 1 or sample_count < 1 or len(losses) != step:
        raise ValueError("invalid checkpoint progress")
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "schema": 1,
        "identity": training_identity,
        "versions": versions(),
        "step": step,
        "next_sample": step % sample_count,
        "sample_count": sample_count,
        "losses": losses,
        "initial_digest": initial_digest,
        "parameter_names": list(trainable),
        "factors": _cpu(trainable),
        "optimizer": _cpu(optimizer.state_dict()),
        "optimizer_kind": "AdamW",
        "scheduler": None,
        "rng": capture_rng(),
    }
    if not trainable or not _finite(payload):
        raise ValueError("nonfinite/empty checkpoint")
    root = Path(directory) / "checkpoints"
    root.mkdir(exist_ok=True)
    name = f"step-{step:06d}-{uuid.uuid4().hex}"
    pending = root / (name + ".pending")
    pending.mkdir()
    with (pending / "state.pt").open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema": 1,
        "name": name,
        "step": step,
        "state_sha256": file_sha256(pending / "state.pt"),
    }
    atomic_json(pending / "commit.json", manifest)
    pending.rename(root / name)
    # atomic_json fsyncs the parent after committing the pointer.
    atomic_json(root / "latest.json", manifest)
    return manifest


def read_checkpoint(directory, expected_identity):
    import torch

    root = Path(directory) / "checkpoints"
    pointer = json.loads((root / "latest.json").read_text())
    name = pointer.get("name", "")
    import re

    if not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{32}", name):
        raise ValueError("invalid checkpoint pointer")
    committed = (root / name).resolve(strict=True)
    if committed.parent != root.resolve() or committed.is_symlink():
        raise ValueError("checkpoint escapes root")
    manifest = json.loads((committed / "commit.json").read_text())
    if (
        manifest != pointer
        or file_sha256(committed / "state.pt") != pointer["state_sha256"]
    ):
        raise ValueError("checkpoint checksum/commit mismatch")
    payload = torch.load(committed / "state.pt", map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or payload.get("identity") != expected_identity:
        raise ValueError("checkpoint training identity mismatch")
    if payload.get("versions") != versions():
        raise ValueError("checkpoint software versions differ")
    step = payload["step"]
    count = payload["sample_count"]
    if (
        type(step) is not int
        or type(count) is not int
        or step < 1
        or count < 1
        or step != pointer["step"]
        or step > expected_identity["spec"]["max_steps"]
        or payload["next_sample"] != step % count
        or len(payload["losses"]) != step
        or payload.get("optimizer_kind") != "AdamW"
        or payload.get("scheduler") is not None
        or not _finite(payload)
    ):
        raise ValueError("invalid checkpoint state")
    return payload, pointer


def restore_checkpoint(model, optimizer, payload):
    import torch

    if payload["versions"] != versions():
        raise ValueError("checkpoint software versions differ")
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if list(trainable) != payload["parameter_names"] or set(trainable) != set(
        payload["factors"]
    ):
        raise ValueError("checkpoint parameter order/set differs")
    if any(
        tuple(trainable[name].shape) != tuple(value.shape)
        or trainable[name].dtype != value.dtype
        for name, value in payload["factors"].items()
    ):
        raise ValueError("checkpoint factor shape/dtype differs")
    # Validate optimizer structure before modifying model factors.
    current = optimizer.state_dict()["param_groups"]
    saved = payload["optimizer"]["param_groups"]
    if current != saved:
        raise ValueError("checkpoint optimizer configuration/order differs")
    expected_ids = [index for group in saved for index in group["params"]]
    if set(payload["optimizer"]["state"]) != set(expected_ids):
        raise ValueError("incomplete optimizer checkpoint")
    for index, parameter in zip(expected_ids, trainable.values()):
        state = payload["optimizer"]["state"][index]
        if (
            set(state) != {"step", "exp_avg", "exp_avg_sq"}
            or state["step"].item() != payload["step"]
            or any(
                tuple(state[key].shape) != tuple(parameter.shape)
                for key in ("exp_avg", "exp_avg_sq")
            )
        ):
            raise ValueError("invalid AdamW moments/step")
    with torch.no_grad():
        for name, value in payload["factors"].items():
            trainable[name].copy_(value.to(trainable[name].device))
    optimizer.load_state_dict(payload["optimizer"])
    # Construction/loading can consume random numbers: restore RNG LAST.
    restore_rng(payload["rng"])


def pause_requested(directory, run_id):
    path = Path(directory) / "pause.request.json"
    if not path.exists():
        return False
    request = json.loads(path.read_text())
    return request.get("run_id") == run_id


def _validate_step_interval(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 60
    ):
        raise ValueError("step interval must be a finite number in [0, 60]")
    return float(value)


def requested_step_interval(directory, run_id, default):
    """Return the latest pace for this run; ignore stale requests from older runs."""
    current = _validate_step_interval(default)
    path = Path(directory) / "pace.request.json"
    if not path.exists():
        return current
    request = json.loads(path.read_text())
    if request.get("run_id") != run_id:
        return current
    return _validate_step_interval(request.get("step_interval_seconds"))


def request_step_interval(store, job_id, step_interval_seconds):
    """Update a running worker's minimum step period without restarting it."""
    interval = _validate_step_interval(step_interval_seconds)
    job = store.read(job_id)
    if job["state"] != "running" or not job.get("run_id"):
        raise ValueError("pace update requires a running managed job")
    request = {
        "job_id": job_id,
        "run_id": job["run_id"],
        "step_interval_seconds": interval,
        "time": time.time(),
    }
    from .jobs import FileLease

    with FileLease(store.directory(job_id) / "pace-request.lock"):
        atomic_json(store.directory(job_id) / "pace.request.json", request)
    return {"accepted": True, **request}


def request_pause(store, job_id):
    # Deliberately does not take job.lock: executor holds it until GPU release.
    # Stale requests cannot affect a resumed worker because each run has a new ID.
    job = store.read(job_id)
    if job["state"] == "paused":
        return {"accepted": True, "already_paused": True, "job_id": job_id}
    if job["state"] != "running" or not job.get("run_id"):
        raise ValueError("pause requires a running managed job")
    request = {"job_id": job_id, "run_id": job["run_id"], "time": time.time()}
    # Serialize request writers, separately from the long-held execution lock.
    from .jobs import FileLease

    with FileLease(store.directory(job_id) / "pause-request.lock"):
        atomic_json(store.directory(job_id) / "pause.request.json", request)
    return {
        "accepted": True,
        "job_id": job_id,
        "run_id": job["run_id"],
        "gpu_released": False,
        "note": "request only; wait for state=paused",
    }

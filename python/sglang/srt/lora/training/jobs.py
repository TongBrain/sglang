"""Durable local jobs and cooperative GPU leases (not a cluster scheduler)."""

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class TrainingSpec:
    job_id: str
    adapter_name: str
    model_path: str
    dataset_path: str
    gpu: int
    initial_adapter: str | None = None
    rank: int = 8
    alpha: int = 16
    target_modules: tuple = ("q_proj", "v_proj")
    max_steps: int = 3
    max_length: int = 96
    learning_rate: float = 0.0002
    seed: int = 42
    timeout_seconds: int = 1200
    minimum_free_mib: int = 50000
    step_interval_seconds: float = 0.0
    inference_load_url: str | None = None
    permit_idle_seconds: float = 0.5
    permit_poll_seconds: float = 0.1
    permit_request_timeout_seconds: float = 1.0
    permit_max_consecutive_errors: int = 3

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.job_id):
            raise ValueError("job_id must be a safe filesystem identifier")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}", self.adapter_name):
            raise ValueError("adapter_name must be an immutable safe name")
        if type(self.gpu) is not int or self.gpu < 0:
            raise ValueError("one nonnegative physical GPU index is required")
        if (
            type(self.rank) is not int
            or type(self.alpha) is not int
            or self.rank not in (4, 8, 16, 32, 64)
            or not 1 <= self.alpha <= 256
        ):
            raise ValueError("unsupported LoRA rank/alpha")
        if (
            not isinstance(self.target_modules, (tuple, list))
            or not self.target_modules
            or not set(self.target_modules) <= {"q_proj", "v_proj"}
            or len(set(self.target_modules)) != len(self.target_modules)
        ):
            raise ValueError(
                "MVP supports attention q_proj/v_proj only, not expert LoRA"
            )
        if (
            type(self.max_steps) is not int
            or type(self.max_length) is not int
            or not 1 <= self.max_steps <= 1000
            or not 8 <= self.max_length <= 512
        ):
            raise ValueError("invalid bounded training steps/sequence length")
        if not 0 < self.learning_rate <= 0.01:
            raise ValueError("invalid learning rate")
        if (
            isinstance(self.step_interval_seconds, bool)
            or not isinstance(self.step_interval_seconds, (int, float))
            or not 0 <= self.step_interval_seconds <= 60
        ):
            raise ValueError("step_interval_seconds must be in [0, 60]")
        if self.inference_load_url is not None:
            parsed = urlsplit(self.inference_load_url)
            query = parse_qs(parsed.query)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path != "/v1/loads"
                or parsed.fragment
                or query.get("include") != ["core"]
            ):
                raise ValueError(
                    "inference_load_url must be a loopback /v1/loads?include=core URL"
                )
        for name, value, lower, upper in (
            ("permit_idle_seconds", self.permit_idle_seconds, 0.0, 10.0),
            ("permit_poll_seconds", self.permit_poll_seconds, 0.02, 5.0),
            (
                "permit_request_timeout_seconds",
                self.permit_request_timeout_seconds,
                0.05,
                10.0,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
        if (
            isinstance(self.permit_max_consecutive_errors, bool)
            or not isinstance(self.permit_max_consecutive_errors, int)
            or not 1 <= self.permit_max_consecutive_errors <= 100
        ):
            raise ValueError("permit_max_consecutive_errors must be in [1, 100]")
        if not 1 <= self.timeout_seconds <= 86400 or self.minimum_free_mib < 1:
            raise ValueError("invalid timeout/memory budget")
        model = Path(self.model_path).resolve(strict=True)
        if not model.is_dir() or not (model / "config.json").is_file():
            raise ValueError("model_path must be a local model with config.json")
        dataset = Path(self.dataset_path).resolve(strict=True)
        if not dataset.is_file():
            raise ValueError("dataset_path must be a local JSONL file")
        if self.initial_adapter and not Path(self.initial_adapter).is_dir():
            raise ValueError("initial_adapter must be a local adapter directory")
        return self


def read_samples(path):
    samples = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict) or any(
                not isinstance(sample.get(key), str) or not sample[key].strip()
                for key in ("prompt", "response")
            ):
                raise ValueError(
                    f"line {number} requires nonempty prompt/response strings"
                )
            samples.append({"prompt": sample["prompt"], "response": sample["response"]})
    if not samples:
        raise ValueError("training dataset is empty")
    return samples


TRANSITIONS = {
    "queued": {"running", "training_failed"},
    "running": {"paused", "trained", "training_failed"},
    "paused": {"running"},
    "trained": {"publishing"},
    "publishing": {"inference_ready", "publish_failed"},
    "publish_failed": {"publishing"},
    "inference_ready": set(),
    "training_failed": set(),
}


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, job_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", job_id):
            raise ValueError("unsafe job_id")
        directory = (self.root / job_id).resolve()
        if directory.parent != self.root:
            raise ValueError("job directory escapes root")
        return directory

    def create(self, spec):
        spec.validate()
        read_samples(spec.dataset_path)
        directory = self.directory(spec.job_id)
        directory.mkdir(exist_ok=False)
        state = {
            "job_id": spec.job_id,
            "state": "queued",
            "spec": asdict(spec),
            "dataset_sha256": file_sha256(spec.dataset_path),
            "events": [{"state": "queued", "time": time.time()}],
        }
        atomic_json(directory / "job.json", state)
        return state

    def read(self, job_id):
        return json.loads((self.directory(job_id) / "job.json").read_text())

    def transition(self, job_id, target, **details):
        state = self.read(job_id)
        if target not in TRANSITIONS[state["state"]]:
            raise ValueError(f"invalid transition {state['state']} -> {target}")
        state.update(details)
        state["state"] = target
        event = {"state": target, "time": time.time()}
        if details.get("error"):
            event["error"] = details["error"]
        state["events"].append(event)
        atomic_json(self.directory(job_id) / "job.json", state)
        return state

    def lock(self, job_id):
        return FileLease(self.directory(job_id) / "job.lock")


class FileLease:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f"lease already held: {self.path}") from None
        return self

    def __exit__(self, *_):
        self.stream.close()
        self.stream = None


def ensure_gpu_idle(gpu, minimum_free_mib):
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    if any(
        line.strip().startswith(gpu_uuid + ",") for line in applications.splitlines()
    ):
        raise RuntimeError(f"GPU {gpu} has an existing compute job; refusing to share")
    free = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    if free < minimum_free_mib:
        raise RuntimeError(f"GPU {gpu} free memory {free} MiB is below budget")
    return {"gpu": gpu, "uuid": gpu_uuid, "free_mib": free}

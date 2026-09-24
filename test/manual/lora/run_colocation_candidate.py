"""Run one bounded full-speed train/inference co-location probe on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

import run_colocation_baseline as baseline
from sglang.benchmark.colocation_gate import compare_runs
from sglang.srt.lora.training.checkpoint import (
    identity,
    read_checkpoint,
    request_pause,
)
from sglang.srt.lora.training.jobs import (
    FileLease,
    JobStore,
    TrainingSpec,
    atomic_json,
    ensure_gpu_idle,
)


BASELINE_RESULT = Path(os.environ.get("SGLANG_LORA_TEST_BASELINE_RESULT", "/path/to/baseline.jsonl"))


def server_command(port: int) -> list[str]:
    return [
        str(baseline.PYTHON),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(baseline.MODEL),
        "--tp-size",
        "1",
        "--ep-size",
        "1",
        "--enable-lora",
        "--lora-paths",
        f"baseline={baseline.ADAPTER}",
        "--max-lora-rank",
        "8",
        "--max-loaded-loras",
        "4",
        "--max-loras-per-batch",
        "4",
        "--lora-page-rank-size",
        "8",
        "--lora-pages",
        "8",
        "--context-length",
        "4096",
        "--max-total-tokens",
        "8192",
        "--max-running-requests",
        "16",
        "--mem-fraction-static",
        "0.70",
        "--attention-backend",
        "fa3",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-max-bs-decode",
        "16",
        "--disable-radix-cache",
        "--random-seed",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def benchmark_command(port: int, output: Path) -> list[str]:
    return [
        str(baseline.PYTHON),
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        str(baseline.MODEL),
        "--dataset-name",
        "random",
        "--dataset-path",
        str(baseline.DATASET),
        "--num-prompts",
        "128",
        "--random-input-len",
        "256",
        "--random-output-len",
        "128",
        "--random-range-ratio",
        "1.0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        "16",
        "--lora-name",
        "baseline",
        "--warmup-requests",
        "16",
        "--flush-cache",
        "--tokenize-prompt",
        "--output-details",
        "--output-file",
        str(output),
        "--disable-tqdm",
        "--seed",
        "42",
    ]


def wait_for_server(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 900
    session = requests.Session()
    session.trust_env = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited during startup; inspect server.log")
        try:
            response = session.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("server readiness deadline exceeded")


def wait_for_training_step(
    process: subprocess.Popen, progress_path: Path, timeout: float = 900
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("training exited before co-location measurement")
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress.get("step", 0) >= 1:
                return progress
        time.sleep(0.1)
    raise TimeoutError("training did not complete its first step")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-interval-seconds", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=13819)
    args = parser.parse_args()
    gpu = int(os.environ.get("SGLANG_LORA_TEST_GPU", "0"))
    port = args.port
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = baseline.REPO / "work" / f"colocation-candidate-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "result.json"
    candidate_path = output / "candidate.jsonl"
    dataset = output / "training.jsonl"
    samples = [
        {
            "prompt": "Question: What is protected during co-location?\nAnswer:",
            "response": " Inference throughput and request correctness are protected.",
        },
        {
            "prompt": "Question: What happens when inference load rises?\nAnswer:",
            "response": " Training checkpoints and pauses before giving the GPU back.",
        },
        {
            "prompt": "Question: What does LoRA optimize?\nAnswer:",
            "response": " Only low-rank adapter factors while base weights stay frozen.",
        },
    ]
    dataset.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )
    store = JobStore(output / "jobs")
    spec = TrainingSpec(
        job_id=f"colocate-{stamp}",
        adapter_name=f"colocate@{stamp}",
        model_path=str(baseline.MODEL),
        dataset_path=str(dataset),
        gpu=gpu,
        max_steps=1000,
        max_length=96,
        step_interval_seconds=args.step_interval_seconds,
        timeout_seconds=1800,
        minimum_free_mib=1,
    )
    report = {
        "state": "starting",
        "kind": (
            "full_speed_training_with_inference"
            if args.step_interval_seconds == 0
            else "paced_training_with_inference"
        ),
        "expected_outcome": "quantify interference; pass is not assumed",
        "gpu": gpu,
        "output_dir": str(output),
        "baseline_result": str(BASELINE_RESULT),
        "throughput_gate_pct": 5.0,
        "inference_workload": {
            "num_prompts": 128,
            "random_input_len": 256,
            "random_output_len": 128,
            "random_range_ratio": 1.0,
            "max_concurrency": 16,
            "warmup_requests": 16,
            "seed": 42,
        },
        "training_spec": spec.__dict__,
    }
    atomic_json(result_path, report)
    server = None
    trainer = None
    server_log = None
    train_log = None

    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=str(baseline.REPO / "python"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    serve = server_command(port)
    bench = benchmark_command(port, candidate_path)
    report.update(server_command=serve, benchmark_command=bench)
    atomic_json(result_path, report)

    try:
        for path in (
            baseline.PYTHON,
            baseline.MODEL,
            baseline.ADAPTER / "adapter_config.json",
            baseline.DATASET,
            BASELINE_RESULT,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))

        with FileLease(baseline.REPO / "work/.gpu-leases" / f"gpu-{gpu}.lock"):
            ensure_gpu_idle(gpu, 50000)
            report["gpu_before"] = baseline.gpu_snapshot(gpu)
            server_log = (output / "server.log").open("x", encoding="utf-8")
            server = subprocess.Popen(
                serve,
                cwd=baseline.REPO,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            report.update(state="server_starting", server_pid=server.pid)
            atomic_json(result_path, report)
            wait_for_server(server, port)
            report.update(
                state="training_starting",
                server_ready_at=time.time(),
                gpu_server_ready=baseline.gpu_snapshot(gpu),
            )

            store.create(spec)
            run_id = uuid.uuid4().hex
            store.transition(
                spec.job_id,
                "running",
                gpu_observation={
                    "mode": "owned_co_location_probe",
                    "gpu": gpu,
                    "server_pid": server.pid,
                },
                initial_artifact=None,
                run_id=run_id,
                resume_checkpoint=None,
                error=None,
                gpu_released=False,
            )
            directory = store.directory(spec.job_id)
            train_log = (directory / "train.log").open("x", encoding="utf-8")
            trainer = subprocess.Popen(
                [
                    str(baseline.PYTHON),
                    "-m",
                    "sglang.srt.lora.training.peft_worker",
                    "--job-dir",
                    str(directory),
                ],
                cwd=baseline.REPO,
                env=env,
                stdout=train_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            atomic_json(
                directory / "process.json",
                {
                    "pid": trainer.pid,
                    "command_backend": "peft",
                    "run_id": run_id,
                    "mode": "owned_co_location_probe",
                },
            )
            report.update(training_pid=trainer.pid)
            atomic_json(result_path, report)
            progress_before = wait_for_training_step(
                trainer, directory / "progress.json"
            )
            report.update(
                state="benchmark_running",
                training_progress_before=progress_before,
                gpu_train_and_infer=baseline.gpu_snapshot(gpu),
            )
            atomic_json(result_path, report)

            with (output / "benchmark.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    bench,
                    cwd=baseline.REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark exited with {completed.returncode}; inspect benchmark.log"
                )
            if trainer.poll() is not None:
                raise RuntimeError("training did not remain active for the benchmark")
            progress_after = json.loads((directory / "progress.json").read_text())
            if progress_after["step"] <= progress_before["step"]:
                raise RuntimeError("training made no progress during inference benchmark")

            pause = request_pause(store, spec.job_id)
            try:
                training_code = trainer.wait(timeout=300)
            except subprocess.TimeoutExpired:
                baseline.stop_process_group(trainer)
                raise RuntimeError("training did not checkpoint and pause in time")
            if training_code != 0:
                raise RuntimeError(
                    f"training exited with {training_code}; inspect train.log"
                )
            outcome = json.loads((directory / "worker_outcome.json").read_text())
            if outcome.get("run_id") != run_id or outcome.get("state") != "paused":
                raise RuntimeError(f"invalid paused outcome: {outcome}")
            payload, manifest = read_checkpoint(
                directory, identity(spec, store.read(spec.job_id))
            )
            if manifest != outcome["checkpoint"] or payload["step"] < 2:
                raise RuntimeError("invalid co-location checkpoint")
            paused = store.transition(
                spec.job_id,
                "paused",
                checkpoint=manifest,
                completed_steps=payload["step"],
                training_process_released=True,
                gpu_shared_with_inference=True,
                gpu_released=False,
            )
            trainer = None

            candidate_rows = [
                json.loads(line)
                for line in candidate_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            baseline_rows = [
                json.loads(line)
                for line in BASELINE_RESULT.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(candidate_rows) != 1 or len(baseline_rows) != 1:
                raise RuntimeError("expected exactly one baseline and candidate row")
            candidate = candidate_rows[0]
            if set(candidate["input_lens"]) != {256}:
                raise RuntimeError("candidate input lengths changed")
            if set(candidate["output_lens"]) != {128}:
                raise RuntimeError("candidate output lengths changed")
            comparison = compare_runs(
                baseline_rows,
                candidate_rows,
                max_throughput_drop_pct=5.0,
                max_p99_ttft_increase_pct=10.0,
                max_p99_tpot_increase_pct=10.0,
                expected_prompts_per_run=128,
            )
            report.update(
                state="completed",
                experiment_passed=True,
                inference_gate_passed=comparison["passed"],
                comparison=comparison,
                training={
                    "progress_before": progress_before,
                    "progress_after": progress_after,
                    "paused": paused,
                    "checkpoint": manifest,
                },
                gpu_after_training_pause=baseline.gpu_snapshot(gpu),
                finished_at=time.time(),
            )
            atomic_json(result_path, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
    except BaseException as error:
        report.update(
            state="failed",
            experiment_passed=False,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(result_path, report)
        raise
    finally:
        if trainer is not None:
            baseline.stop_process_group(trainer)
        if server is not None:
            baseline.stop_process_group(server)
        if train_log is not None:
            train_log.close()
        if server_log is not None:
            server_log.close()
        try:
            report["gpu_after_cleanup"] = baseline.gpu_snapshot(gpu)
            atomic_json(result_path, report)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

"""Run a real fixed-work inference-window loop against paced LoRA training.

This is an opt-in, single-GPU manual experiment. It starts only its own server
and trainer, keeps serving work fixed across windows, applies governor actions,
checkpoints the trainer, and cleans up.
"""

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

import run_colocation_candidate as colocate
from sglang.srt.lora.training.checkpoint import (
    identity,
    read_checkpoint,
    request_pause,
    request_step_interval,
)
from sglang.srt.lora.training.governor import (
    benchmark_row_to_window,
    evaluate_window,
)
from sglang.srt.lora.training.jobs import (
    FileLease,
    JobStore,
    TrainingSpec,
    atomic_json,
    ensure_gpu_idle,
)


baseline = colocate.baseline
REFERENCE_REQUEST_THROUGHPUT = 13.53633893041136


def prompt_count_for_window(window_seconds, concurrency):
    if window_seconds <= 0 or concurrency < 1:
        raise ValueError("window seconds and concurrency must be positive")
    estimate = window_seconds * REFERENCE_REQUEST_THROUGHPUT
    return max(concurrency, round(estimate / concurrency) * concurrency)


def benchmark_command(port, output, num_prompts, concurrency, seed=42):
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
        str(num_prompts),
        "--random-input-len",
        "256",
        "--random-output-len",
        "128",
        "--random-range-ratio",
        "1.0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--lora-name",
        "baseline",
        "--tokenize-prompt",
        "--output-details",
        "--output-file",
        str(output),
        "--disable-tqdm",
        "--seed",
        str(seed),
    ]


def read_one_row(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one benchmark row, found {len(rows)}")
    return rows[0]


def run_benchmark(command, output, log_path, env, timeout=1800):
    started = time.time()
    with Path(log_path).open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=baseline.REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark exited with {completed.returncode}; inspect {log_path}"
        )
    return read_one_row(output), {
        "started_at": started,
        "finished_at": time.time(),
    }


def validate_fixed_workload(row, expected_prompts):
    if len(row.get("input_lens", [])) != expected_prompts:
        raise RuntimeError("benchmark did not retain all input length records")
    if len(row.get("output_lens", [])) != expected_prompts:
        raise RuntimeError("benchmark did not retain all output length records")
    if set(row["input_lens"]) != {256}:
        raise RuntimeError("benchmark input lengths changed")
    if set(row["output_lens"]) != {128}:
        raise RuntimeError("benchmark output lengths changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("SGLANG_LORA_TEST_GPU", "0")))
    parser.add_argument("--port", type=int, default=13820)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--candidate-windows", type=int, default=3)
    parser.add_argument("--initial-step-interval", type=float, default=2.0)
    parser.add_argument("--step-permit", action="store_true")
    parser.add_argument("--permit-idle-seconds", type=float, default=0.5)
    parser.add_argument("--permit-poll-seconds", type=float, default=0.1)
    args = parser.parse_args()
    if not 1 <= args.candidate_windows <= 10:
        raise ValueError("candidate windows must be in [1, 10]")
    if not 0.5 <= args.initial_step_interval <= 10:
        raise ValueError("initial step interval must be in [0.5, 10]")

    concurrency = 16
    num_prompts = prompt_count_for_window(args.window_seconds, concurrency)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = baseline.REPO / "work" / f"colocation-governor-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "result.json"
    dataset = output / "training.jsonl"
    samples = [
        {
            "prompt": "Question: What does LoRA keep frozen?\nAnswer:",
            "response": " The base model weights remain frozen.",
        },
        {
            "prompt": "Question: What does LoRA optimize?\nAnswer:",
            "response": " Low-rank adapter factors are optimized.",
        },
    ]
    dataset.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    store = JobStore(output / "jobs")
    spec = TrainingSpec(
        job_id=f"governor-{stamp}",
        adapter_name=f"governor@{stamp}",
        model_path=str(baseline.MODEL),
        dataset_path=str(dataset),
        gpu=args.gpu,
        max_steps=1000,
        max_length=96,
        step_interval_seconds=args.initial_step_interval,
        inference_load_url=(
            f"http://127.0.0.1:{args.port}/v1/loads?include=core"
            if args.step_permit
            else None
        ),
        permit_idle_seconds=args.permit_idle_seconds,
        permit_poll_seconds=args.permit_poll_seconds,
        timeout_seconds=1800,
        minimum_free_mib=1,
    )
    report = {
        "state": "starting",
        "kind": "real_inference_window_governor",
        "window_contract": {
            "target_seconds": args.window_seconds,
            "mode": "fixed_work_targeting_duration",
            "num_prompts": num_prompts,
            "max_concurrency": concurrency,
            "random_input_len": 256,
            "random_output_len": 128,
            "seed": 42,
            "candidate_windows": args.candidate_windows,
            "step_permit": args.step_permit,
            "permit_idle_seconds": args.permit_idle_seconds,
            "permit_poll_seconds": args.permit_poll_seconds,
            "note": "No benchmark time-stop option; actual duration is recorded.",
        },
        "policy": {
            "max_throughput_drop_pct": 5.0,
            "max_p99_ttft_increase_pct": 10.0,
            "max_p99_tpot_increase_pct": 10.0,
            "warning_ratio": 0.8,
            "healthy_ratio": 0.5,
            "healthy_windows_to_speed_up": 3,
        },
        "training_spec": spec.__dict__,
        "windows": [],
    }
    atomic_json(result_path, report)
    server = None
    trainer = None
    server_log = None
    train_log = None
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
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
    try:
        for path in (
            baseline.PYTHON,
            baseline.MODEL,
            baseline.ADAPTER / "adapter_config.json",
            baseline.DATASET,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))

        with FileLease(baseline.REPO / "work/.gpu-leases" / f"gpu-{args.gpu}.lock"):
            ensure_gpu_idle(args.gpu, 50000)
            report["gpu_before"] = baseline.gpu_snapshot(args.gpu)
            server_log = (output / "server.log").open("x", encoding="utf-8")
            server = subprocess.Popen(
                colocate.server_command(args.port),
                cwd=baseline.REPO,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            report.update(state="server_starting", server_pid=server.pid)
            atomic_json(result_path, report)
            colocate.wait_for_server(server, args.port)
            report.update(
                state="warmup_running",
                server_ready_at=time.time(),
                gpu_server_ready=baseline.gpu_snapshot(args.gpu),
            )
            atomic_json(result_path, report)

            warmup_path = output / "warmup.jsonl"
            warmup_row, warmup_times = run_benchmark(
                benchmark_command(args.port, warmup_path, concurrency, concurrency),
                warmup_path,
                output / "warmup.log",
                env,
            )
            warmup_window = benchmark_row_to_window(warmup_row, concurrency)
            if (
                warmup_window["completed"] != concurrency
                or warmup_window["request_error_count"]
            ):
                raise RuntimeError(f"warmup failed: {warmup_window}")
            report["warmup"] = {"metrics": warmup_window, **warmup_times}

            baseline_path = output / "baseline-window.jsonl"
            report.update(state="baseline_window_running")
            atomic_json(result_path, report)
            baseline_row, baseline_times = run_benchmark(
                benchmark_command(
                    args.port, baseline_path, num_prompts, concurrency
                ),
                baseline_path,
                output / "baseline-window.log",
                env,
            )
            validate_fixed_workload(baseline_row, num_prompts)
            baseline_window = benchmark_row_to_window(baseline_row, num_prompts)
            if (
                baseline_window["completed"] != num_prompts
                or baseline_window["request_error_count"]
            ):
                raise RuntimeError(f"baseline window failed: {baseline_window}")
            duration = float(baseline_row["duration"])
            if not args.window_seconds * 0.75 <= duration <= args.window_seconds * 1.25:
                raise RuntimeError(
                    f"baseline duration {duration:.3f}s is not a usable "
                    f"{args.window_seconds:.3f}s window"
                )
            report["baseline_window"] = {
                "metrics": baseline_window,
                "duration_seconds": duration,
                **baseline_times,
            }
            atomic_json(result_path, report)

            store.create(spec)
            run_id = uuid.uuid4().hex
            store.transition(
                spec.job_id,
                "running",
                gpu_observation={
                    "mode": "real_inference_window_governor",
                    "gpu": args.gpu,
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
                    "mode": "real_inference_window_governor",
                },
            )
            initial_progress = colocate.wait_for_training_step(
                trainer, directory / "progress.json"
            )
            report.update(
                state="candidate_windows_running",
                training_pid=trainer.pid,
                initial_training_progress=initial_progress,
                gpu_train_and_infer=baseline.gpu_snapshot(args.gpu),
            )
            atomic_json(result_path, report)

            governor_state = {
                "step_interval_seconds": initial_progress["step_interval_seconds"],
                "healthy_windows": 0,
            }
            previous_step = initial_progress["step"]
            governor_pause = None
            for index in range(1, args.candidate_windows + 1):
                if trainer.poll() is not None:
                    raise RuntimeError("training exited before candidate window")
                window_path = output / f"candidate-window-{index}.jsonl"
                row, times = run_benchmark(
                    benchmark_command(
                        args.port, window_path, num_prompts, concurrency
                    ),
                    window_path,
                    output / f"candidate-window-{index}.log",
                    env,
                )
                validate_fixed_workload(row, num_prompts)
                current = benchmark_row_to_window(row, num_prompts)
                decision = evaluate_window(
                    baseline_window, current, governor_state
                )
                progress = json.loads((directory / "progress.json").read_text())
                step_delta = progress["step"] - previous_step
                if not args.step_permit and step_delta <= 0:
                    raise RuntimeError(
                        f"training made no progress during window {index}"
                    )
                previous_step = progress["step"]
                permit_status_path = directory / "permit_status.json"
                permit_status = (
                    json.loads(permit_status_path.read_text())
                    if permit_status_path.exists()
                    else None
                )
                action_request = None
                if decision["pause_required"]:
                    action_request = request_pause(store, spec.job_id)
                    governor_pause = action_request
                elif (
                    decision["step_interval_seconds"]
                    != governor_state["step_interval_seconds"]
                ):
                    action_request = request_step_interval(
                        store,
                        spec.job_id,
                        decision["step_interval_seconds"],
                    )
                report["windows"].append(
                    {
                        "index": index,
                        "duration_seconds": float(row["duration"]),
                        "metrics": current,
                        "decision": decision,
                        "action_request": action_request,
                        "training_progress": progress,
                        "training_step_delta": step_delta,
                        "permit_status": permit_status,
                        **times,
                    }
                )
                governor_state = decision
                atomic_json(result_path, report)
                if governor_pause is not None:
                    break

            if governor_pause is None:
                final_pause = request_pause(store, spec.job_id)
                pause_reason = "experiment_complete"
            else:
                final_pause = governor_pause
                pause_reason = "governor_redline"
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
                raise RuntimeError("invalid governor checkpoint")
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
            final_permit_status = None
            permit_status_path = directory / "permit_status.json"
            if permit_status_path.exists():
                final_permit_status = json.loads(permit_status_path.read_text())
            if args.step_permit and (
                final_permit_status is None
                or final_permit_status.get("busy_observations", 0) < 1
                or final_permit_status.get("granted_count", 0) < 1
            ):
                raise RuntimeError(
                    "step permit did not observe both busy inference and an idle grant"
                )
            report.update(
                state="completed",
                control_loop_passed=True,
                inference_sla_passed=all(
                    not window["decision"]["pause_required"]
                    for window in report["windows"]
                ),
                final_pause_request=final_pause,
                pause_reason=pause_reason,
                training={"paused": paused, "checkpoint": manifest},
                final_permit_status=final_permit_status,
                gpu_after_training_pause=baseline.gpu_snapshot(args.gpu),
                finished_at=time.time(),
            )
            atomic_json(result_path, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
    except BaseException as error:
        report.update(
            state="failed",
            control_loop_passed=False,
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
            report["gpu_after_cleanup"] = baseline.gpu_snapshot(args.gpu)
            atomic_json(result_path, report)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

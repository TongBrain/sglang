"""Run one reproducible single-GPU LoRA inference baseline.

This is an opt-in manual benchmark.  It owns the selected GPU for the lifetime
of the server, records the exact commands, and always stops the process it
started.  It does not start a training process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

from sglang.srt.lora.training.jobs import FileLease, atomic_json, ensure_gpu_idle


REPO = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
MODEL = Path(os.environ.get("SGLANG_LORA_TEST_MODEL", "/path/to/Qwen1.5-MoE-A2.7B"))
ADAPTER = Path(os.environ.get("SGLANG_LORA_TEST_ADAPTER", "/path/to/attention-lora-r8"))
DATASET = Path(os.environ.get("SGLANG_LORA_TEST_DATASET", "/path/to/benchmark-dataset.json"))


def gpu_snapshot(gpu: int) -> dict[str, int]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, free, utilization = (int(value.strip()) for value in raw.split(","))
    return {
        "memory_used_mib": used,
        "memory_free_mib": free,
        "utilization_pct": utilization,
    }


def stop_process_group(process: subprocess.Popen, timeout: float = 60) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("SGLANG_LORA_TEST_GPU", "0")))
    parser.add_argument("--port", type=int, default=13819)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = REPO / "work" / f"colocation-baseline-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "result.json"
    benchmark_path = output / "baseline.jsonl"
    report = {
        "state": "starting",
        "kind": "pure_inference_baseline",
        "gpu": args.gpu,
        "output_dir": str(output),
        "workload": {
            "num_prompts": 128,
            "random_input_len": 256,
            "random_output_len": 128,
            "random_range_ratio": 1.0,
            "max_concurrency": 16,
            "warmup_requests": 16,
            "repeats": 1,
            "seed": 42,
        },
        "adapter": str(ADAPTER),
        "model": str(MODEL),
        "dataset": str(DATASET),
    }
    atomic_json(result_path, report)
    server = None
    server_log = None

    server_command = [
        str(PYTHON),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(MODEL),
        "--tp-size",
        "1",
        "--ep-size",
        "1",
        "--enable-lora",
        "--lora-paths",
        f"baseline={ADAPTER}",
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
        str(args.port),
    ]
    benchmark_command = [
        str(PYTHON),
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--model",
        str(MODEL),
        "--dataset-name",
        "random",
        "--dataset-path",
        str(DATASET),
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
        str(benchmark_path),
        "--disable-tqdm",
        "--seed",
        "42",
    ]
    report.update(
        server_command=server_command,
        benchmark_command=benchmark_command,
    )
    atomic_json(result_path, report)

    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(REPO / "python"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_OFFLINE="1",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )

    try:
        for path in (PYTHON, MODEL, ADAPTER / "adapter_config.json", DATASET):
            if not path.exists():
                raise FileNotFoundError(path)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        with FileLease(REPO / "work/.gpu-leases" / f"gpu-{args.gpu}.lock"):
            ensure_gpu_idle(args.gpu, 50000)
            report["gpu_before"] = gpu_snapshot(args.gpu)
            server_log = (output / "server.log").open("x", encoding="utf-8")
            server = subprocess.Popen(
                server_command,
                cwd=REPO,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            report.update(state="server_starting", server_pid=server.pid)
            atomic_json(result_path, report)

            deadline = time.monotonic() + 900
            session = requests.Session()
            session.trust_env = False
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("server exited during startup; inspect server.log")
                try:
                    response = session.get(
                        f"http://127.0.0.1:{args.port}/health", timeout=2
                    )
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("server readiness deadline exceeded")

            report.update(
                state="benchmark_running",
                server_ready_at=time.time(),
                gpu_server_ready=gpu_snapshot(args.gpu),
            )
            atomic_json(result_path, report)
            with (output / "benchmark.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    benchmark_command,
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark exited with {completed.returncode}; inspect benchmark.log"
                )
            rows = [
                json.loads(line)
                for line in benchmark_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != 1:
                raise RuntimeError(f"expected one benchmark row, found {len(rows)}")
            row = rows[0]
            errors = [error for error in row.get("errors", []) if error]
            if row["completed"] != 128 or errors:
                raise RuntimeError(
                    f"incomplete benchmark: completed={row['completed']} errors={len(errors)}"
                )
            if set(row["input_lens"]) != {256}:
                raise RuntimeError(
                    f"unexpected input lengths: {sorted(set(row['input_lens']))}"
                )
            if set(row["output_lens"]) != {128}:
                raise RuntimeError(
                    f"unexpected output lengths: {sorted(set(row['output_lens']))}"
                )
            report.update(
                state="passed",
                finished_at=time.time(),
                metrics={
                    key: row[key]
                    for key in (
                        "duration",
                        "completed",
                        "request_throughput",
                        "output_throughput",
                        "total_throughput",
                        "p99_ttft_ms",
                        "p99_tpot_ms",
                        "p99_itl_ms",
                    )
                },
            )
            atomic_json(result_path, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(result_path, report)
        raise
    finally:
        if server is not None:
            stop_process_group(server)
        if server_log is not None:
            server_log.close()
        try:
            report["gpu_after_cleanup"] = gpu_snapshot(args.gpu)
            atomic_json(result_path, report)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

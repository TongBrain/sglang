"""Opt-in GPU proof that governor decisions reach a running training worker."""

import concurrent.futures
import json
import os
import time
from pathlib import Path

from sglang.srt.lora.training.checkpoint import (
    request_pause,
    request_step_interval,
)
from sglang.srt.lora.training.governor import evaluate_window
from sglang.srt.lora.training.jobs import JobStore, TrainingSpec, atomic_json, ensure_gpu_idle
from sglang.srt.lora.training.runtime import execute_training


REPO = Path(__file__).resolve().parents[3]
MODEL = os.environ.get("SGLANG_LORA_TEST_MODEL", "/path/to/Qwen1.5-MoE-A2.7B")
GPU = int(os.environ.get("SGLANG_LORA_TEST_GPU", "0"))


def wait_for(predicate, description, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {description}")


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = REPO / "work" / f"governor-live-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "training.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "prompt": "Question: Why should training yield to inference?\nAnswer:",
                "response": " Inference service objectives protect user experience.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = TrainingSpec(
        job_id=f"governor-{stamp}",
        adapter_name=f"governor@{stamp}",
        model_path=MODEL,
        dataset_path=str(dataset),
        gpu=GPU,
        max_steps=100,
        max_length=96,
        timeout_seconds=1200,
        step_interval_seconds=2.0,
    )
    store = JobStore(output / "jobs")
    directory = store.directory(spec.job_id)
    baseline = {
        "output_throughput": 100.0,
        "p99_ttft_ms": 100.0,
        "p99_tpot_ms": 10.0,
    }
    report = {
        "passed": False,
        "kind": "governor_control_path_live_test",
        "metric_windows": "controlled_injection_not_live_inference",
        "output_dir": str(output),
        "spec": spec.__dict__,
    }
    atomic_json(output / "result.json", report)

    try:
        report["gpu_before"] = ensure_gpu_idle(GPU, 50000)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(execute_training, store, spec)

            def first_progress():
                if future.done():
                    future.result()
                    raise RuntimeError("training exited before governor action")
                path = directory / "progress.json"
                if not path.exists():
                    return None
                progress = json.loads(path.read_text())
                return progress if progress.get("step", 0) >= 1 else None

            initial = wait_for(first_progress, "first training step")
            state = {
                "step_interval_seconds": initial["step_interval_seconds"],
                "healthy_windows": 0,
            }
            warning_window = {
                "output_throughput": 96.0,
                "p99_ttft_ms": 100.0,
                "p99_tpot_ms": 10.0,
                "completed": 128,
                "expected_completed": 128,
                "request_error_count": 0,
            }
            warning = evaluate_window(baseline, warning_window, state)
            if warning["action"] != "slow_down":
                raise RuntimeError(f"warning did not slow training: {warning}")
            pace = request_step_interval(
                store, spec.job_id, warning["step_interval_seconds"]
            )

            def updated_progress():
                if future.done():
                    future.result()
                    raise RuntimeError("training exited before pace update was observed")
                progress = json.loads((directory / "progress.json").read_text())
                if (
                    progress.get("step", 0) >= 2
                    and progress.get("step_interval_seconds") == 2.5
                ):
                    return progress
                return None

            updated = wait_for(updated_progress, "worker pace update")
            breach_window = {
                **warning_window,
                "output_throughput": 94.0,
            }
            breach = evaluate_window(baseline, breach_window, warning)
            if breach["action"] != "pause":
                raise RuntimeError(f"breach did not pause training: {breach}")
            pause = request_pause(store, spec.job_id)
            paused = future.result(timeout=300)
        if (
            paused["state"] != "paused"
            or not paused["gpu_released"]
            or paused["completed_steps"] < 2
            or paused["checkpoint"]["step"] != paused["completed_steps"]
        ):
            raise RuntimeError(f"invalid final paused state: {paused}")
        report.update(
            passed=True,
            initial_progress=initial,
            warning_decision=warning,
            pace_request=pace,
            updated_progress=updated,
            breach_decision=breach,
            pause_request=pause,
            paused_job=paused,
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            report["gpu_after"] = ensure_gpu_idle(GPU, 50000)
            report["gpu_released"] = True
        except Exception as error:
            report["gpu_released"] = False
            report["gpu_check_error"] = repr(error)
        report["passed"] = report["passed"] and report["gpu_released"]
        atomic_json(output / "result.json", report)
        print(json.dumps({"result": str(output / "result.json"), **report}, indent=2))


if __name__ == "__main__":
    main()

"""Opt-in train->export->serve test, using only one idle GPU sequentially."""

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import test_lora_lifecycle_live as serving

from sglang.srt.lora.training.jobs import (
    FileLease,
    JobStore,
    TrainingSpec,
    atomic_json,
    ensure_gpu_idle,
)
from sglang.srt.lora.training.publish import InferenceClient, publish_job
from sglang.srt.lora.training.runtime import execute_training, stop_owned_process

REPO = Path(__file__).resolve().parents[3]
MODEL = Path(os.environ.get("SGLANG_LORA_TEST_MODEL", "/path/to/Qwen1.5-MoE-A2.7B"))
GPU = int(os.environ.get("SGLANG_LORA_TEST_GPU", "0"))
PORT = 13818


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-adapter")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = REPO / "work" / f"train-serve-{stamp}"
    output.mkdir(exist_ok=False, parents=True)
    dataset = output / "synthetic-smoke.jsonl"
    samples = [
        {
            "prompt": "Question: What does LoRA train?\nAnswer:",
            "response": " LoRA trains low-rank adapter matrices while freezing the base model.",
        },
        {
            "prompt": "Question: How should a new adapter be published?\nAnswer:",
            "response": " Load an immutable version, validate it, run a canary, then switch routing.",
        },
        {
            "prompt": "Question: Can an old request be interrupted during an update?\nAnswer:",
            "response": " Existing requests should finish before their adapter is unloaded.",
        },
    ]
    dataset.write_text("".join(json.dumps(item) + "\n" for item in samples))
    spec = TrainingSpec(
        f"smoke-{stamp}",
        f"smoke@v2-{stamp}",
        str(MODEL),
        str(dataset),
        GPU,
        initial_adapter=args.initial_adapter,
        max_steps=args.steps,
    )
    store = JobStore(output / "jobs")
    atomic_json(output / "spec.json", spec.__dict__)
    result = {
        "passed": False,
        "gpu": GPU,
        "output_dir": str(output),
        "steps": [],
        "dataset_kind": "synthetic_smoke",
        "quality_evaluation": "not_performed",
        "same_gpu_execution": "training_then_inference_not_concurrent",
    }
    original = serving.source_hashes()
    result["original_source_hashes_before"] = original
    server = None
    executor = None
    log = None

    def record(stage, **details):
        result["steps"].append({"stage": stage, "time": time.time(), **details})
        atomic_json(output / "result.json", result)
        print(json.dumps({"stage": stage, **details}, default=str), flush=True)

    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", PORT))
        record("starting_training", spec=spec.__dict__)
        trained = execute_training(store, spec)
        record(
            "training_complete",
            job_state=trained["state"],
            evidence=trained["training_result"],
            artifact=trained["artifact"],
            gpu_released=trained["gpu_after_training"],
        )
        directory = store.directory(spec.job_id)
        serving.BASE = f"http://127.0.0.1:{PORT}"
        serving.PORT = PORT
        with FileLease(store.root / ".gpu-leases" / f"gpu-{GPU}.lock"):
            ensure_gpu_idle(GPU, 50000)
            command = [
                sys.executable,
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
                f"seed={directory / 'v1'}",
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
                "4",
                "--mem-fraction-static",
                "0.70",
                "--attention-backend",
                "fa3",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--disable-radix-cache",
                "--random-seed",
                "1",
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
            ]
            env = dict(os.environ)
            env.update(
                CUDA_VISIBLE_DEVICES=str(GPU),
                PYTHONPATH=str(REPO / "python"),
                PYTHONDONTWRITEBYTECODE="1",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                NO_PROXY="127.0.0.1,localhost",
                no_proxy="127.0.0.1,localhost",
            )
            identity = subprocess.check_output(
                [sys.executable, "-c", "import sglang; print(sglang.__file__)"],
                env=env,
                text=True,
            ).strip()
            assert str(REPO / "python") in identity, identity
            result.update(server_command=command, imported_source=identity)
            log = (output / "server.log").open("x")
            server = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record("starting_inference", pid=server.pid)
            deadline = time.monotonic() + 900
            with requests.Session() as session:
                session.trust_env = False
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(
                            "inference startup failed; inspect server.log"
                        )
                    try:
                        response = session.get(serving.BASE + "/health", timeout=2)
                        if response.status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(2)
                else:
                    raise TimeoutError("inference readiness deadline")
            v1 = f"smoke@v1-{stamp}"
            loaded = serving.api(
                "POST",
                "/load_lora_adapter",
                {"lora_name": v1, "lora_path": str(directory / "v1")},
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
            ids = tokenizer.encode("Explain LoRA briefly.", add_special_tokens=False)
            serving.api(
                "POST", "/prewarm_lora_adapter", {"lora_name": v1, "input_ids": ids}
            )
            record(
                "v1_ready",
                lora_id=loaded["lora_id"],
                initial_adapter=args.initial_adapter or "fresh_attention_lora",
            )
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            started = threading.Event()
            old = executor.submit(serving.long_stream, v1, started)
            assert started.wait(60), "v1 stream failed to start"
            in_flight = serving.api("GET", "/get_lora_adapter_status?lora_name=" + v1)
            assert in_flight["adapters"][v1]["active_requests"] >= 1, in_flight
            published = publish_job(
                store,
                spec.job_id,
                InferenceClient(serving.BASE),
                ids,
                "Explain briefly how LoRA modifies a neural network.",
            )
            assert published["state"] == "inference_ready", published
            assert loaded["lora_id"] != published["publication"]["lora_id"]
            assert not old.done(), "old request ended before new version publication"
            record(
                "v2_inference_ready",
                publication=published["publication"],
                old_request_still_running=True,
            )
            before = serving.api("GET", "/get_lora_adapter_status")
            unload = executor.submit(
                serving.api, "POST", "/unload_lora_adapter", {"lora_name": v1}
            )
            time.sleep(0.3)
            assert (
                not unload.done() and not old.done()
            ), "unload did not wait for old request"
            new = serving.generate(spec.adapter_name)
            assert new["meta_info"]["completion_tokens"] == 32, new
            old_result = old.result(timeout=300)
            unloaded = unload.result(timeout=60)
            after = serving.api("GET", "/get_lora_adapter_status")
            assert (
                v1 not in after["adapters"] and spec.adapter_name in after["adapters"]
            )
            assert (
                after["adapters"][spec.adapter_name]["lora_id"]
                == published["publication"]["lora_id"]
            )
            assert (
                after["workers"][0]["free_pages"]
                == before["workers"][0]["free_pages"] + 1
            )
            record(
                "v1_drained_and_unloaded",
                old_completion_tokens=old_result["meta_info"]["completion_tokens"],
                unload=unloaded,
                v2_completion_tokens=new["meta_info"]["completion_tokens"],
                before=before,
                after=after,
            )
            serving.api(
                "POST", "/unload_lora_adapter", {"lora_name": spec.adapter_name}
            )
            cleaned = serving.api("GET", "/get_lora_adapter_status")
            assert (
                cleaned["workers"][0]["free_pages"]
                == cleaned["workers"][0]["total_pages"]
            )
            record("new_adapter_cleaned", status=cleaned)
            stop_owned_process(server)
            server = None
        result["passed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if server is not None:
            stop_owned_process(server)
        if executor is not None:
            executor.shutdown(wait=True)
        if log is not None:
            log.close()
        result["original_source_hashes_after"] = serving.source_hashes()
        result["original_source_unchanged"] = (
            original == result["original_source_hashes_after"]
        )
        try:
            result["gpu_after"] = ensure_gpu_idle(GPU, 50000)
            result["gpu_released_after_test"] = True
        except Exception as error:
            result.update(gpu_released_after_test=False, gpu_check_error=str(error))
        result["passed"] = bool(
            result["passed"]
            and result["original_source_unchanged"]
            and result["gpu_released_after_test"]
        )
        atomic_json(output / "result.json", result)
        print(
            json.dumps(
                {"passed": result["passed"], "result": str(output / "result.json")}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

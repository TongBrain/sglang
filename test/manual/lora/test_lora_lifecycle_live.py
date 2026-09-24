"""Opt-in GPU lifecycle test; owns only its isolated server process group."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[3]
ORIGINAL = Path(os.environ.get("SGLANG_LORA_TEST_ORIGINAL_REPO", str(REPO)))
MODEL = Path(os.environ.get("SGLANG_LORA_TEST_MODEL", "/path/to/Qwen1.5-MoE-A2.7B"))
CATALOG = Path(os.environ.get("SGLANG_LORA_TEST_CATALOG", "/path/to/structural-adapter-catalog"))
PYTHON = Path(sys.executable)
PORT = 13817
GPU = os.environ.get("SGLANG_LORA_TEST_GPU", "0")
BASE = f"http://127.0.0.1:{PORT}"
SOURCE_FILES = [
    "python/sglang/srt/entrypoints/http_server.py",
    "python/sglang/srt/managers/io_struct.py",
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/tokenizer_control_mixin.py",
    "python/sglang/srt/lora/lora_manager.py",
    "python/sglang/srt/lora/lora_registry.py",
]


def source_hashes():
    return {
        name: hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def gpu_free():
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", GPU, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    return not any(line.startswith(uuid + ",") for line in apps.splitlines())


def api(method, path, body=None, expected=200, timeout=300):
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, BASE + path, json=body, timeout=timeout)
        assert response.status_code == expected, (
            path,
            response.status_code,
            response.text[:2000],
        )
        result = response.json()
        if expected == 200 and isinstance(result, dict) and "success" in result:
            assert result["success"], result
        return result


def generate(name, tokens=32):
    return api(
        "POST",
        "/generate",
        {
            "text": "Explain briefly how LoRA modifies a neural network.",
            "lora_path": name,
            "sampling_params": {
                "max_new_tokens": tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
    )


def long_stream(name, started):
    with requests.Session() as session:
        session.trust_env = False
        with session.post(
            BASE + "/generate",
            json={
                "text": "Explain how low-rank adaptation works in detail.",
                "lora_path": name,
                "stream": True,
                "sampling_params": {
                    "max_new_tokens": 1024,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            stream=True,
            timeout=300,
        ) as response:
            assert response.status_code == 200, response.text[:1000]
            final = None
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                final = json.loads(payload)
                reason = final.get("meta_info", {}).get("finish_reason")
                assert not reason or reason.get("type") != "abort", final
                started.set()
            assert final is not None, "No stream output"
            assert final["meta_info"]["completion_tokens"] == 1024, final["meta_info"]
            assert final["meta_info"]["finish_reason"]["type"] == "length", final[
                "meta_info"
            ]
            return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-graph", action="store_true")
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = REPO / "work" / f"lifecycle-gpu-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "output_dir": str(out),
        "gpu": GPU,
        "tp_size": 1,
        "model": str(MODEL),
        "decode_graph": args.decode_graph,
        "steps": [],
        "passed": False,
    }
    original_before = source_hashes()
    result["original_source_hashes_before"] = original_before
    server = None
    executor = None
    log = None

    def record(stage, **evidence):
        item = {"stage": stage, "time": time.time(), **evidence}
        result["steps"].append(item)
        (out / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps({"stage": stage, "time": item["time"]}, ensure_ascii=False),
            flush=True,
        )

    try:
        assert gpu_free(), "Selected GPU is occupied; will not share an existing job"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", PORT))
        for name in ("r8_000", "r8_001", "r16_000", "r128_000"):
            assert (CATALOG / name / "adapter_model.safetensors").is_file(), name
        command = [
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
            f'seed={CATALOG / "r8_001"}',
            "--max-lora-rank",
            "64",
            "--max-loaded-loras",
            "8",
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
            "full" if args.decode_graph else "disabled",
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
        if args.decode_graph:
            command.extend(["--cuda-graph-max-bs-decode", "4"])
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=GPU,
            PYTHONPATH=str(REPO / "python"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_OFFLINE="1",
            NO_PROXY="127.0.0.1,localhost",
            no_proxy="127.0.0.1,localhost",
        )
        identity = subprocess.check_output(
            [str(PYTHON), "-c", "import sglang; print(sglang.__file__)"],
            env=env,
            text=True,
        ).strip()
        assert str(REPO / "python") in identity, identity
        result["command"] = command
        result["imported_source"] = identity
        log = (out / "server.log").open("w", encoding="utf-8")
        server = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        result["server_pid"] = server.pid
        record("starting_server", pid=server.pid)
        start = time.monotonic()
        last_update = start
        with requests.Session() as session:
            session.trust_env = False
            while time.monotonic() - start < 900:
                if server.poll() is not None:
                    raise RuntimeError(
                        f'Server exited {server.returncode}; see {out / "server.log"}'
                    )
                try:
                    if session.get(BASE + "/health", timeout=3).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() - last_update > 20:
                    print(
                        f"Starting server: {int(time.monotonic() - start)}s elapsed",
                        flush=True,
                    )
                    last_update = time.monotonic()
                time.sleep(2)
            else:
                raise TimeoutError("Server startup exceeded 900 seconds")
        record("server_ready", seconds=round(time.monotonic() - start, 2))
        v1, v2, bad = (
            f"test_user@v1-{stamp}",
            f"test_user@v2-{stamp}",
            f"test_bad@v3-{stamp}",
        )
        load1 = api(
            "POST",
            "/load_lora_adapter",
            {"lora_name": v1, "lora_path": str(CATALOG / "r8_000")},
        )
        id1 = load1["lora_id"]
        status1 = api("GET", "/get_lora_adapter_status?lora_name=" + v1)
        assert (
            status1["adapters"][v1]["loaded"]
            and status1["adapters"][v1]["lora_id"] == id1
        )
        record("v1_loaded", response=load1, status=status1)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        ids = tokenizer.encode("Hello, explain LoRA.", add_special_tokens=False)
        warm1 = api(
            "POST", "/prewarm_lora_adapter", {"lora_name": v1, "input_ids": ids}
        )
        assert warm1["probe_completed"] and warm1["adapters"][v1]["resident"], warm1
        initial = generate(v1)
        assert initial["meta_info"]["completion_tokens"] == 32
        record("v1_prewarm_and_inference", status=warm1, completion=initial)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        started = threading.Event()
        old = executor.submit(long_stream, v1, started)
        assert started.wait(60), "Old request did not begin streaming"
        if old.done():
            old.result()
            raise AssertionError("Old request completed too early to test overlap")
        in_flight = api("GET", "/get_lora_adapter_status?lora_name=" + v1)
        assert in_flight["adapters"][v1]["active_requests"] >= 1, in_flight
        record("old_v1_request_in_flight", status=in_flight)
        load2 = api(
            "POST",
            "/load_lora_adapter",
            {"lora_name": v2, "lora_path": str(CATALOG / "r16_000")},
        )
        id2 = load2["lora_id"]
        assert id1 != id2
        warm2 = api(
            "POST", "/prewarm_lora_adapter", {"lora_name": v2, "input_ids": ids}
        )
        assert warm2["probe_completed"] and warm2["adapters"][v2]["resident"], warm2
        newer = generate(v2)
        assert newer["meta_info"]["completion_tokens"] == 32
        before_unload = api("GET", "/get_lora_adapter_status")
        assert (
            before_unload["adapters"][v1]["active_requests"] >= 1 and not old.done()
        ), before_unload
        record(
            "v2_loaded_and_new_request_completed_while_v1_active",
            response=load2,
            status=before_unload,
            completion=newer,
        )

        unload = executor.submit(api, "POST", "/unload_lora_adapter", {"lora_name": v1})
        time.sleep(0.3)
        assert (
            not old.done() and not unload.done()
        ), "Could not observe unload waiting on the old request"
        during_unload = generate(v2)
        assert during_unload["meta_info"]["completion_tokens"] == 32
        record("v2_serves_while_v1_unload_waits", completion=during_unload)
        old_completion = old.result(timeout=300)
        removed = unload.result(timeout=60)
        assert removed["lora_id"] == id1
        after_unload = api("GET", "/get_lora_adapter_status")
        assert v1 not in after_unload["adapters"]
        assert after_unload["adapters"][v2]["lora_id"] == id2
        old_pages = before_unload["workers"][0]["adapters"][v1]["required_pages"]
        assert (
            after_unload["workers"][0]["free_pages"]
            >= before_unload["workers"][0]["free_pages"] + old_pages
        )
        unknown = api("GET", "/get_lora_adapter_status?lora_name=" + v1, expected=400)
        assert not unknown["success"]
        record(
            "old_request_completed_and_v1_pages_reclaimed",
            old_completion=old_completion,
            unload=removed,
            status=after_unload,
        )

        duplicate = api(
            "POST",
            "/load_lora_adapter",
            {"lora_name": v2, "lora_path": str(CATALOG / "r16_000")},
            expected=400,
        )
        assert not duplicate["success"]
        failures = []
        for _ in range(3):
            rejected = api(
                "POST",
                "/load_lora_adapter",
                {"lora_name": bad, "lora_path": str(CATALOG / "r128_000")},
                expected=400,
            )
            assert not rejected["success"] and "rollback failed" not in (
                rejected.get("error_message") or ""
            )
            failures.append(rejected)
        corrupted = out / "corrupted-adapter"
        corrupted.mkdir()
        (corrupted / "adapter_config.json").write_bytes(
            (CATALOG / "r8_000" / "adapter_config.json").read_bytes()
        )
        (corrupted / "adapter_model.safetensors").write_bytes(
            b"\x04\x00\x00\x00\x00\x00\x00\x00nope"
        )
        corrupt_failures = []
        for _ in range(3):
            rejected = api(
                "POST",
                "/load_lora_adapter",
                {"lora_name": bad, "lora_path": str(corrupted)},
                expected=400,
            )
            assert not rejected["success"] and "rollback failed" not in (
                rejected.get("error_message") or ""
            )
            corrupt_failures.append(rejected)
        after_failure = api("GET", "/get_lora_adapter_status")
        assert bad not in after_failure["adapters"]
        assert all(bad not in worker["adapters"] for worker in after_failure["workers"])
        assert (
            after_failure["workers"][0]["free_pages"]
            == after_unload["workers"][0]["free_pages"]
        )
        assert after_failure["adapters"][v2]["lora_id"] == id2
        healthy = generate(v2)
        assert healthy["meta_info"]["completion_tokens"] == 32
        result["v2_output_tokens_match_after_failures"] = (
            healthy["output_ids"] == newer["output_ids"]
        )
        record(
            "incompatible_rank_and_corrupted_weights_rejected_without_affecting_v2",
            failures=failures,
            corrupt_failures=corrupt_failures,
            duplicate=duplicate,
            status=after_failure,
            completion=healthy,
        )
        api("POST", "/unload_lora_adapter", {"lora_name": v1, "missing_ok": True})
        api("POST", "/unload_lora_adapter", {"lora_name": v2})
        final_status = api("GET", "/get_lora_adapter_status")
        assert set(final_status["adapters"]) == {"seed"}, final_status
        result["passed"] = True
        record("test_passed", final_status=final_status)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        record("test_failed", error=result["error"])
        raise
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=15)
            # The session leader can exit before scheduler children do.
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if log is not None:
            log.close()
        result["original_source_hashes_after"] = source_hashes()
        result["original_source_unchanged"] = (
            result["original_source_hashes_after"] == original_before
        )
        result["gpu_free_after_cleanup"] = gpu_free()
        (out / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "result": str(out / "result.json"),
                    "passed": result["passed"],
                    "original_source_unchanged": result["original_source_unchanged"],
                    "gpu_free_after_cleanup": result["gpu_free_after_cleanup"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

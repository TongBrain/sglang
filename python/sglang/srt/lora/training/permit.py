"""Fail-closed inference-idle permits for co-located LoRA training steps."""

import json
import time
import urllib.request

from .jobs import atomic_json


def parse_load_snapshot(payload):
    """Return a validated aggregate snapshot and whether every DP rank is idle."""
    if not isinstance(payload, dict) or not isinstance(payload.get("loads"), list):
        raise ValueError("load response requires a loads list")
    if not payload["loads"]:
        raise ValueError("load response must contain at least one DP rank")
    running = 0
    waiting = 0
    ranks = []
    for load in payload["loads"]:
        if not isinstance(load, dict):
            raise ValueError("each load snapshot must be a mapping")
        rank = load.get("dp_rank")
        current_running = load.get("num_running_reqs")
        current_waiting = load.get("num_waiting_reqs")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(current_running, bool)
            or not isinstance(current_running, int)
            or current_running < 0
            or isinstance(current_waiting, bool)
            or not isinstance(current_waiting, int)
            or current_waiting < 0
        ):
            raise ValueError("load snapshot has invalid rank/request counts")
        ranks.append(
            {
                "dp_rank": rank,
                "num_running_reqs": current_running,
                "num_waiting_reqs": current_waiting,
            }
        )
        running += current_running
        waiting += current_waiting
    return {
        "idle": running == 0 and waiting == 0,
        "num_running_reqs": running,
        "num_waiting_reqs": waiting,
        "ranks": ranks,
    }


def fetch_load_snapshot(url, timeout_seconds):
    """Fetch through a proxy-free client; URL admission is handled by TrainingSpec."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout_seconds) as response:
        body = response.read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ValueError("load response exceeds 1 MiB")
    return parse_load_snapshot(json.loads(body))


def wait_for_step_permit(
    directory,
    run_id,
    *,
    load_url,
    idle_seconds,
    poll_seconds,
    request_timeout_seconds,
    max_consecutive_errors,
    stop_requested,
    fetch=fetch_load_snapshot,
):
    """Wait until all ranks stay idle; return denied when a pause is requested."""
    if load_url is None:
        return {
            "enabled": False,
            "granted": True,
            "wait_seconds": 0.0,
            "busy_observations": 0,
            "granted_count": 0,
        }
    started = time.monotonic()
    idle_since = None
    consecutive_errors = 0
    busy_observations = 0
    observations = 0
    status_path = directory / "permit_status.json"
    previous = {}
    if status_path.exists():
        try:
            previous = json.loads(status_path.read_text())
        except (OSError, ValueError):
            previous = {}
    if previous.get("run_id") != run_id:
        previous = {}
    granted_count = int(previous.get("granted_count", 0))
    cumulative_busy = int(previous.get("busy_observations", 0))
    last_written_state = None
    last_written_at = 0.0

    def write_status(state, observation=None, error=None, force=False):
        nonlocal last_written_state, last_written_at
        now = time.monotonic()
        if not force and state == last_written_state and now - last_written_at < 1.0:
            return
        value = {
            "run_id": run_id,
            "state": state,
            "time": time.time(),
            "wait_seconds": now - started,
            "observations": observations,
            "busy_observations": cumulative_busy + busy_observations,
            "granted_count": granted_count,
        }
        if observation is not None:
            value["load"] = observation
        if error is not None:
            value["error"] = error
        atomic_json(status_path, value)
        last_written_state = state
        last_written_at = now

    while True:
        if stop_requested():
            write_status("denied_pause", force=True)
            return {
                "enabled": True,
                "granted": False,
                "wait_seconds": time.monotonic() - started,
                "busy_observations": busy_observations,
                "granted_count": granted_count,
            }
        try:
            observation = fetch(load_url, request_timeout_seconds)
            consecutive_errors = 0
            observations += 1
        except Exception as error:
            idle_since = None
            consecutive_errors += 1
            write_status(
                "load_error",
                error=f"{type(error).__name__}: {error}",
                force=True,
            )
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    "inference load endpoint failed repeatedly; permit denied"
                ) from error
            time.sleep(poll_seconds)
            continue

        now = time.monotonic()
        if observation["idle"]:
            if idle_since is None:
                idle_since = now
            if now - idle_since >= idle_seconds:
                granted_count += 1
                write_status("granted", observation, force=True)
                return {
                    "enabled": True,
                    "granted": True,
                    "wait_seconds": now - started,
                    "busy_observations": busy_observations,
                    "granted_count": granted_count,
                    "load": observation,
                }
            write_status("idle_guard", observation)
        else:
            idle_since = None
            busy_observations += 1
            write_status("waiting_busy", observation)
        time.sleep(poll_seconds)

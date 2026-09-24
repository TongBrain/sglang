"""Pure policy for pacing or pausing LoRA training from inference windows."""

from __future__ import annotations

import math


DEFAULT_POLICY = {
    "max_throughput_drop_pct": 5.0,
    "max_p99_ttft_increase_pct": 10.0,
    "max_p99_tpot_increase_pct": 10.0,
    "warning_ratio": 0.8,
    "healthy_ratio": 0.5,
    "healthy_windows_to_speed_up": 3,
    "interval_adjustment_seconds": 0.5,
    "minimum_interval_seconds": 0.5,
    "maximum_interval_seconds": 10.0,
}


def benchmark_row_to_window(row, expected_completed):
    """Project one serving benchmark row into the governor's metric contract."""
    if not isinstance(row, dict):
        raise ValueError("benchmark row must be a mapping")
    if (
        isinstance(expected_completed, bool)
        or not isinstance(expected_completed, int)
        or expected_completed < 1
    ):
        raise ValueError("expected_completed must be a positive integer")
    completed = row.get("completed")
    errors = row.get("errors")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or not isinstance(errors, list)
    ):
        raise ValueError("benchmark row has invalid completion/error fields")
    return {
        "output_throughput": row.get("output_throughput"),
        "p99_ttft_ms": row.get("p99_ttft_ms"),
        "p99_tpot_ms": row.get("p99_tpot_ms"),
        "completed": completed,
        "expected_completed": expected_completed,
        "request_error_count": sum(bool(error) for error in errors),
    }


def _finite_nonnegative(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _positive_metric(mapping, name):
    value = _finite_nonnegative(mapping.get(name), name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_policy(policy=None):
    value = {**DEFAULT_POLICY, **(policy or {})}
    for name in (
        "max_throughput_drop_pct",
        "max_p99_ttft_increase_pct",
        "max_p99_tpot_increase_pct",
        "interval_adjustment_seconds",
        "minimum_interval_seconds",
        "maximum_interval_seconds",
    ):
        value[name] = _finite_nonnegative(value[name], name)
    for name in (
        "max_throughput_drop_pct",
        "max_p99_ttft_increase_pct",
        "max_p99_tpot_increase_pct",
    ):
        if value[name] == 0:
            raise ValueError(f"{name} must be positive")
    for name in ("warning_ratio", "healthy_ratio"):
        value[name] = _finite_nonnegative(value[name], name)
        if value[name] > 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if value["healthy_ratio"] >= value["warning_ratio"]:
        raise ValueError("healthy_ratio must be below warning_ratio")
    if value["minimum_interval_seconds"] > value["maximum_interval_seconds"]:
        raise ValueError("minimum interval cannot exceed maximum interval")
    count = value["healthy_windows_to_speed_up"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("healthy_windows_to_speed_up must be a positive integer")
    return value


def _change_pct(current, baseline, *, degradation_when_lower):
    change = (current / baseline - 1.0) * 100
    return max(0.0, -change if degradation_when_lower else change)


def _exceeds(value, limit):
    return value > limit and not math.isclose(
        value, limit, rel_tol=1e-12, abs_tol=1e-12
    )


def evaluate_window(baseline, current, state, policy=None):
    """Return a deterministic decision; this function performs no side effects."""
    limits = validate_policy(policy)
    base_throughput = _positive_metric(baseline, "output_throughput")
    base_ttft = _positive_metric(baseline, "p99_ttft_ms")
    base_tpot = _positive_metric(baseline, "p99_tpot_ms")
    current_throughput = _finite_nonnegative(
        current.get("output_throughput"), "output_throughput"
    )
    current_ttft = _finite_nonnegative(current.get("p99_ttft_ms"), "p99_ttft_ms")
    current_tpot = _finite_nonnegative(current.get("p99_tpot_ms"), "p99_tpot_ms")
    completed = current.get("completed")
    expected = current.get("expected_completed")
    errors = current.get("request_error_count", 0)
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or isinstance(expected, bool)
        or not isinstance(expected, int)
        or completed < 0
        or expected < 1
        or isinstance(errors, bool)
        or not isinstance(errors, int)
        or errors < 0
    ):
        raise ValueError("invalid request completion/error counts")
    interval = _finite_nonnegative(
        state.get("step_interval_seconds"), "step_interval_seconds"
    )
    if not limits["minimum_interval_seconds"] <= interval <= limits[
        "maximum_interval_seconds"
    ]:
        raise ValueError("state interval is outside policy bounds")
    healthy_windows = state.get("healthy_windows", 0)
    if isinstance(healthy_windows, bool) or not isinstance(healthy_windows, int):
        raise ValueError("healthy_windows must be an integer")

    usage = {
        "throughput_drop_pct": _change_pct(
            current_throughput, base_throughput, degradation_when_lower=True
        ),
        "p99_ttft_increase_pct": _change_pct(
            current_ttft, base_ttft, degradation_when_lower=False
        ),
        "p99_tpot_increase_pct": _change_pct(
            current_tpot, base_tpot, degradation_when_lower=False
        ),
    }
    breaches = []
    if completed != expected or errors:
        breaches.append("request_correctness")
    for metric, limit_name in (
        ("throughput_drop_pct", "max_throughput_drop_pct"),
        ("p99_ttft_increase_pct", "max_p99_ttft_increase_pct"),
        ("p99_tpot_increase_pct", "max_p99_tpot_increase_pct"),
    ):
        if _exceeds(usage[metric], limits[limit_name]):
            breaches.append(metric)
    if breaches:
        return {
            "action": "pause",
            "pause_required": True,
            "step_interval_seconds": interval,
            "healthy_windows": 0,
            "breaches": breaches,
            "usage": usage,
        }

    ratios = [
        usage["throughput_drop_pct"] / limits["max_throughput_drop_pct"],
        usage["p99_ttft_increase_pct"] / limits["max_p99_ttft_increase_pct"],
        usage["p99_tpot_increase_pct"] / limits["max_p99_tpot_increase_pct"],
    ]
    if max(ratios) >= limits["warning_ratio"]:
        return {
            "action": "slow_down",
            "pause_required": False,
            "step_interval_seconds": min(
                limits["maximum_interval_seconds"],
                interval + limits["interval_adjustment_seconds"],
            ),
            "healthy_windows": 0,
            "breaches": [],
            "usage": usage,
        }

    if max(ratios) <= limits["healthy_ratio"]:
        healthy_windows += 1
        if healthy_windows >= limits["healthy_windows_to_speed_up"]:
            return {
                "action": "speed_up",
                "pause_required": False,
                "step_interval_seconds": max(
                    limits["minimum_interval_seconds"],
                    interval - limits["interval_adjustment_seconds"],
                ),
                "healthy_windows": 0,
                "breaches": [],
                "usage": usage,
            }
    else:
        healthy_windows = 0
    return {
        "action": "hold",
        "pause_required": False,
        "step_interval_seconds": interval,
        "healthy_windows": healthy_windows,
        "breaches": [],
        "usage": usage,
    }

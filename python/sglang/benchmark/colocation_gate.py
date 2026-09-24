"""Compare inference benchmark runs for train/inference co-location experiments.

The SGLang serving benchmark already measures request throughput, output-token
throughput, TTFT, TPOT, and per-request errors.  This module deliberately does
not duplicate that measurement code; it turns repeated JSONL results into a
small, deterministic acceptance decision.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


REQUIRED_METRICS = (
    "output_throughput",
    "request_throughput",
    "p99_ttft_ms",
    "p99_tpot_ms",
    "completed",
)


def load_runs(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load every non-empty JSON object from one or more JSONL files."""
    runs: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                runs.append(value)
    if not runs:
        raise ValueError("no benchmark runs were found")
    return runs


def _number(run: dict[str, Any], metric: str, index: int) -> float:
    if metric not in run:
        raise ValueError(f"run {index} is missing required metric {metric!r}")
    value = float(run[metric])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"run {index} has invalid {metric!r}: {run[metric]!r}")
    return value


def summarize_runs(
    runs: list[dict[str, Any]], expected_prompts_per_run: int | None = None
) -> dict[str, Any]:
    """Summarize repeated benchmark runs without hiding failed requests."""
    if expected_prompts_per_run is not None and expected_prompts_per_run <= 0:
        raise ValueError("expected_prompts_per_run must be positive")

    values = {
        metric: [_number(run, metric, index) for index, run in enumerate(runs)]
        for metric in REQUIRED_METRICS
    }
    request_errors = sum(
        1
        for run in runs
        for error in run.get("errors", [])
        if error not in (None, "")
    )
    completion_ok = (
        True
        if expected_prompts_per_run is None
        else all(
            int(run["completed"]) == expected_prompts_per_run for run in runs
        )
    )
    throughput = values["output_throughput"]
    throughput_median = statistics.median(throughput)
    return {
        "run_count": len(runs),
        "output_throughput_median": throughput_median,
        "output_throughput_min": min(throughput),
        "output_throughput_max": max(throughput),
        "output_throughput_relative_range_pct": (
            0.0
            if throughput_median == 0
            else (max(throughput) - min(throughput)) / throughput_median * 100
        ),
        "request_throughput_median": statistics.median(
            values["request_throughput"]
        ),
        "p99_ttft_ms_median": statistics.median(values["p99_ttft_ms"]),
        "p99_tpot_ms_median": statistics.median(values["p99_tpot_ms"]),
        "request_error_count": request_errors,
        "all_runs_completed": completion_ok,
    }


def _relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate / baseline - 1.0) * 100


def _within_upper_limit(value: float, limit: float) -> bool:
    return value <= limit or math.isclose(
        value, limit, rel_tol=1e-12, abs_tol=1e-12
    )


def compare_runs(
    baseline_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
    *,
    max_throughput_drop_pct: float = 5.0,
    max_p99_ttft_increase_pct: float = 10.0,
    max_p99_tpot_increase_pct: float = 10.0,
    expected_prompts_per_run: int | None = None,
) -> dict[str, Any]:
    """Apply the inference-throughput gate chosen for co-location experiments."""
    if not 0 <= max_throughput_drop_pct < 100:
        raise ValueError("max_throughput_drop_pct must be in [0, 100)")
    if max_p99_ttft_increase_pct < 0 or max_p99_tpot_increase_pct < 0:
        raise ValueError("latency increase limits must be nonnegative")

    baseline = summarize_runs(baseline_runs, expected_prompts_per_run)
    candidate = summarize_runs(candidate_runs, expected_prompts_per_run)
    baseline_throughput = baseline["output_throughput_median"]
    candidate_throughput = candidate["output_throughput_median"]
    minimum_throughput = baseline_throughput * (
        1.0 - max_throughput_drop_pct / 100.0
    )
    throughput_pass = candidate_throughput >= minimum_throughput
    ttft_change = _relative_change(
        candidate["p99_ttft_ms_median"], baseline["p99_ttft_ms_median"]
    )
    tpot_change = _relative_change(
        candidate["p99_tpot_ms_median"], baseline["p99_tpot_ms_median"]
    )
    if ttft_change is None or tpot_change is None:
        raise ValueError("baseline p99 TTFT/TPOT must be positive")
    ttft_pass = _within_upper_limit(ttft_change, max_p99_ttft_increase_pct)
    tpot_pass = _within_upper_limit(tpot_change, max_p99_tpot_increase_pct)
    correctness_pass = all(
        summary["all_runs_completed"] and summary["request_error_count"] == 0
        for summary in (baseline, candidate)
    )

    return {
        "passed": throughput_pass and ttft_pass and tpot_pass and correctness_pass,
        "policy": {
            "primary_metric": "median output tokens/s",
            "max_throughput_drop_pct": max_throughput_drop_pct,
            "minimum_candidate_output_throughput": minimum_throughput,
            "max_p99_ttft_increase_pct": max_p99_ttft_increase_pct,
            "max_p99_tpot_increase_pct": max_p99_tpot_increase_pct,
        },
        "checks": {
            "throughput_pass": throughput_pass,
            "p99_ttft_pass": ttft_pass,
            "p99_tpot_pass": tpot_pass,
            "correctness_pass": correctness_pass,
        },
        "baseline": baseline,
        "candidate": candidate,
        "changes_pct": {
            "output_throughput": _relative_change(
                candidate_throughput, baseline_throughput
            ),
            "p99_ttft_ms": ttft_change,
            "p99_tpot_ms": tpot_change,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--max-throughput-drop-pct", type=float, default=5.0)
    parser.add_argument("--max-p99-ttft-increase-pct", type=float, default=10.0)
    parser.add_argument("--max-p99-tpot-increase-pct", type=float, default=10.0)
    parser.add_argument("--expected-prompts-per-run", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = compare_runs(
        load_runs(args.baseline),
        load_runs(args.candidate),
        max_throughput_drop_pct=args.max_throughput_drop_pct,
        max_p99_ttft_increase_pct=args.max_p99_ttft_increase_pct,
        max_p99_tpot_increase_pct=args.max_p99_tpot_increase_pct,
        expected_prompts_per_run=args.expected_prompts_per_run,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

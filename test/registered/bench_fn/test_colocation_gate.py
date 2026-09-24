import json
import tempfile
import unittest
from pathlib import Path

from sglang.benchmark.colocation_gate import compare_runs, load_runs


def run(throughput, *, completed=128, errors=None, ttft=80, tpot=12):
    return {
        "output_throughput": throughput,
        "request_throughput": throughput / 128,
        "p99_ttft_ms": ttft,
        "p99_tpot_ms": tpot,
        "completed": completed,
        "errors": [] if errors is None else errors,
    }


class TestColocationGate(unittest.TestCase):
    def test_exactly_five_percent_drop_passes_and_latency_is_reported(self):
        report = compare_runs(
            [run(98), run(100), run(102)],
            [
                run(94, ttft=84, tpot=12.6),
                run(95, ttft=84, tpot=12.6),
                run(96, ttft=84, tpot=12.6),
            ],
            expected_prompts_per_run=128,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["candidate"]["output_throughput_median"], 95)
        self.assertAlmostEqual(report["changes_pct"]["output_throughput"], -5)
        self.assertAlmostEqual(report["changes_pct"]["p99_ttft_ms"], 5)
        self.assertTrue(report["checks"]["p99_ttft_pass"])
        self.assertTrue(report["checks"]["p99_tpot_pass"])

    def test_more_than_five_percent_drop_fails(self):
        report = compare_runs(
            [run(100), run(100), run(100)],
            [run(94), run(94), run(94)],
            expected_prompts_per_run=128,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["throughput_pass"])

    def test_failed_or_missing_requests_fail_correctness_gate(self):
        report = compare_runs(
            [run(100)],
            [run(100, completed=127, errors=["timeout"])],
            expected_prompts_per_run=128,
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["correctness_pass"])

    def test_ttft_above_ten_percent_fails(self):
        report = compare_runs([run(100)], [run(100, ttft=88.1)])
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["p99_ttft_pass"])

    def test_tpot_above_ten_percent_fails(self):
        report = compare_runs([run(100)], [run(100, tpot=13.21)])
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["p99_tpot_pass"])

    def test_exactly_ten_percent_latency_increase_passes(self):
        report = compare_runs(
            [run(100)], [run(100, ttft=88, tpot=13.2)]
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["p99_ttft_pass"])
        self.assertTrue(report["checks"]["p99_tpot_pass"])

    def test_loads_multiple_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            path.write_text(
                "\n".join(json.dumps(value) for value in [run(99), run(101)]),
                encoding="utf-8",
            )
            self.assertEqual(len(load_runs([path])), 2)


if __name__ == "__main__":
    unittest.main()

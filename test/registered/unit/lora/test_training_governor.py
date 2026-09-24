import unittest

from sglang.srt.lora.training.governor import (
    benchmark_row_to_window,
    evaluate_window,
)


BASELINE = {
    "output_throughput": 100.0,
    "p99_ttft_ms": 100.0,
    "p99_tpot_ms": 10.0,
}


def window(throughput=100.0, ttft=100.0, tpot=10.0, completed=128, errors=0):
    return {
        "output_throughput": throughput,
        "p99_ttft_ms": ttft,
        "p99_tpot_ms": tpot,
        "completed": completed,
        "expected_completed": 128,
        "request_error_count": errors,
    }


class TestTrainingGovernor(unittest.TestCase):
    def test_benchmark_row_projects_real_metrics(self):
        row = {
            "output_throughput": 97.0,
            "p99_ttft_ms": 101.0,
            "p99_tpot_ms": 10.2,
            "completed": 4,
            "errors": ["", "", "timeout", ""],
        }
        self.assertEqual(
            benchmark_row_to_window(row, 4),
            {
                "output_throughput": 97.0,
                "p99_ttft_ms": 101.0,
                "p99_tpot_ms": 10.2,
                "completed": 4,
                "expected_completed": 4,
                "request_error_count": 1,
            },
        )

    def test_benchmark_row_rejects_missing_error_vector(self):
        with self.assertRaisesRegex(ValueError, "completion/error"):
            benchmark_row_to_window({"completed": 1}, 1)

    def state(self, interval=2.0, healthy=0):
        return {"step_interval_seconds": interval, "healthy_windows": healthy}

    def test_any_redline_breach_pauses(self):
        cases = [
            window(throughput=94.99),
            window(ttft=110.01),
            window(tpot=11.01),
            window(completed=127),
            window(errors=1),
        ]
        for current in cases:
            with self.subTest(current=current):
                decision = evaluate_window(BASELINE, current, self.state())
                self.assertEqual("pause", decision["action"])
                self.assertTrue(decision["pause_required"])

    def test_exact_redlines_do_not_pause_but_enter_warning(self):
        decision = evaluate_window(
            BASELINE, window(throughput=95, ttft=110, tpot=11), self.state()
        )
        self.assertEqual("slow_down", decision["action"])
        self.assertEqual(2.5, decision["step_interval_seconds"])

    def test_warning_zone_slows_training(self):
        decision = evaluate_window(BASELINE, window(throughput=96), self.state())
        self.assertEqual("slow_down", decision["action"])
        self.assertEqual(2.5, decision["step_interval_seconds"])

    def test_three_healthy_windows_are_required_to_speed_up(self):
        state = self.state()
        for expected_action in ("hold", "hold", "speed_up"):
            decision = evaluate_window(BASELINE, window(throughput=99), state)
            self.assertEqual(expected_action, decision["action"])
            state = decision
        self.assertEqual(1.5, state["step_interval_seconds"])

    def test_neutral_window_holds_and_resets_healthy_streak(self):
        decision = evaluate_window(
            BASELINE, window(throughput=97), self.state(healthy=2)
        )
        self.assertEqual("hold", decision["action"])
        self.assertEqual(0, decision["healthy_windows"])

    def test_invalid_policy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "healthy_ratio"):
            evaluate_window(
                BASELINE,
                window(),
                self.state(),
                {"healthy_ratio": 0.9, "warning_ratio": 0.8},
            )


if __name__ == "__main__":
    unittest.main()

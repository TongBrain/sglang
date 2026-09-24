import importlib.util
import sys
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_colocation_governor.py")
spec = importlib.util.spec_from_file_location("run_colocation_governor", RUNNER)
runner = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(RUNNER.parent))
spec.loader.exec_module(runner)


class TestColocationGovernorRunner(unittest.TestCase):
    def test_thirty_second_prompt_count_is_aligned_to_concurrency(self):
        self.assertEqual(runner.prompt_count_for_window(30.0, 16), 400)

    def test_invalid_window_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            runner.prompt_count_for_window(0, 16)

    def test_fixed_workload_accepts_expected_shape(self):
        runner.validate_fixed_workload(
            {"input_lens": [256] * 4, "output_lens": [128] * 4}, 4
        )

    def test_fixed_workload_rejects_changed_shape(self):
        with self.assertRaisesRegex(RuntimeError, "input lengths changed"):
            runner.validate_fixed_workload(
                {"input_lens": [255] * 4, "output_lens": [128] * 4}, 4
            )


if __name__ == "__main__":
    unittest.main()

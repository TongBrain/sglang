import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.lora.training.permit import (
    parse_load_snapshot,
    wait_for_step_permit,
)


def snapshot(running=0, waiting=0):
    return {
        "idle": running == 0 and waiting == 0,
        "num_running_reqs": running,
        "num_waiting_reqs": waiting,
        "ranks": [
            {
                "dp_rank": 0,
                "num_running_reqs": running,
                "num_waiting_reqs": waiting,
            }
        ],
    }


class TestTrainingStepPermit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def wait(self, fetch, stop=lambda: False, error_limit=3):
        with patch("sglang.srt.lora.training.permit.time.sleep"):
            return wait_for_step_permit(
                self.directory,
                "run-1",
                load_url="http://127.0.0.1:30000/v1/loads?include=core",
                idle_seconds=0.0,
                poll_seconds=0.1,
                request_timeout_seconds=1.0,
                max_consecutive_errors=error_limit,
                stop_requested=stop,
                fetch=fetch,
            )

    def test_snapshot_requires_every_rank_idle(self):
        value = parse_load_snapshot(
            {
                "loads": [
                    {"dp_rank": 0, "num_running_reqs": 0, "num_waiting_reqs": 0},
                    {"dp_rank": 1, "num_running_reqs": 2, "num_waiting_reqs": 1},
                ]
            }
        )
        self.assertFalse(value["idle"])
        self.assertEqual(2, value["num_running_reqs"])
        self.assertEqual(1, value["num_waiting_reqs"])

    def test_malformed_snapshot_is_rejected(self):
        for payload in ({}, {"loads": []}, {"loads": [{"dp_rank": 0}]}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_load_snapshot(payload)

    def test_busy_then_idle_grants_one_permit(self):
        fetch = Mock(side_effect=[snapshot(running=1), snapshot()])
        result = self.wait(fetch)
        self.assertTrue(result["granted"])
        self.assertEqual(1, result["busy_observations"])
        self.assertEqual(1, result["granted_count"])
        self.assertEqual(
            "granted",
            json.loads((self.directory / "permit_status.json").read_text())["state"],
        )

    def test_stale_run_counters_are_not_reused(self):
        (self.directory / "permit_status.json").write_text(
            json.dumps(
                {
                    "run_id": "old-run",
                    "busy_observations": 99,
                    "granted_count": 99,
                }
            )
        )
        result = self.wait(Mock(return_value=snapshot()))
        self.assertEqual(0, result["busy_observations"])
        self.assertEqual(1, result["granted_count"])

    def test_pause_denies_without_querying_load(self):
        fetch = Mock()
        result = self.wait(fetch, stop=lambda: True)
        self.assertFalse(result["granted"])
        fetch.assert_not_called()

    def test_repeated_endpoint_failure_is_fail_closed(self):
        fetch = Mock(side_effect=OSError("unreachable"))
        with self.assertRaisesRegex(RuntimeError, "permit denied"):
            self.wait(fetch, error_limit=2)
        self.assertEqual(2, fetch.call_count)

    def test_disabled_permit_preserves_old_behavior(self):
        result = wait_for_step_permit(
            self.directory,
            "run-1",
            load_url=None,
            idle_seconds=0.5,
            poll_seconds=0.1,
            request_timeout_seconds=1.0,
            max_consecutive_errors=3,
            stop_requested=lambda: False,
        )
        self.assertTrue(result["granted"])
        self.assertFalse(result["enabled"])


if __name__ == "__main__":
    unittest.main()

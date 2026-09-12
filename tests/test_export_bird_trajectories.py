import json
import unittest

from scripts.export_bird_trajectories import convert


class ExportBirdTrajectoriesTests(unittest.TestCase):
    def test_exports_submission_in_bird_rl_format(self):
        converted = convert([{"instance_idx": 4, "prediction": " SELECT 1 "}])
        self.assertEqual(converted[0]["instance_idx"], 4)
        turn = converted[0]["trajectory"][0]
        payload = json.loads(turn["action"].removeprefix("<tool_call>").removesuffix("</tool_call>"))
        self.assertEqual(payload, {"name": "submit_solution", "arguments": {"sql": "SELECT 1"}})
        self.assertTrue(turn["end_flag"])

    def test_missing_prediction_becomes_failed_empty_trajectory(self):
        self.assertEqual(convert([{"instance_idx": 0, "prediction": None}])[0]["trajectory"], [])

    def test_rejects_duplicate_indices(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            convert([
                {"instance_idx": 1, "prediction": "SELECT 1"},
                {"instance_idx": 1, "prediction": "SELECT 2"},
            ])

    def test_rejects_incomplete_frozen_run(self):
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            convert([{"instance_idx": 0, "prediction": "SELECT 1"}], expected_records=2)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from sql_agent.history import load_runs, summarize


class HistoryTests(unittest.TestCase):
    def test_summary_distinguishes_unscored_from_incorrect(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            rows = [
                {"profile": "base", "model": "base", "prediction": "SELECT 0",
                 "score": {"correct": False}, "tool_calls": 2, "elapsed_seconds": 1},
                {"profile": "base", "model": "base", "prediction": "SELECT 1",
                 "score": {"correct": True}, "tool_calls": 4, "elapsed_seconds": 3},
                {"profile": "rl", "model": "rl", "prediction": "SELECT 1",
                 "score": None, "tool_calls": 1, "elapsed_seconds": 2},
                {"instance_idx": 0, "trajectory": []},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = {row["profile"]: row for row in summarize(load_runs(directory))}
            self.assertEqual(summary["base"]["accuracy"], 0.5)
            self.assertEqual(summary["base"]["avg_tool_calls"], 3)
            self.assertIsNone(summary["rl"]["accuracy"])


if __name__ == "__main__":
    unittest.main()

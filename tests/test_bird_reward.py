import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agenticrl.bird_reward import compute_score, execute


def tool_call(name, sql):
    return '<tool_call>{"name":"%s","arguments":{"sql":"%s"}}</tool_call>' % (name, sql)


class BirdRewardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        db_dir = root / "demo"
        db_dir.mkdir()
        self.db_path = db_dir / "demo.sqlite"
        conn = sqlite3.connect(self.db_path)
        conn.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1), (2), (2);")
        conn.commit()
        conn.close()
        self.env = patch.dict(os.environ, {"BIRD_DB_DIR": str(root)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_correct_submission_receives_full_reward(self):
        result = compute_score("bird/sql_generation", tool_call("submit_solution", "SELECT x FROM t"),
                               "SELECT x FROM t", {"db_id": "demo"})
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["ex_score"], 1)
        self.assertTrue(result["format_valid"])

    def test_wrong_submission_and_unsubmitted_fallback(self):
        wrong = compute_score("bird/sql_generation", tool_call("submit_solution", "SELECT 9"),
                              "SELECT x FROM t", {"db_id": "demo"})
        fallback = compute_score("bird/sql_generation", tool_call("execute_sql", "SELECT 9"),
                                 "SELECT x FROM t", {"db_id": "demo"})
        self.assertEqual(wrong["score"], 0.1)
        self.assertEqual(fallback["score"], 0.05)
        self.assertFalse(fallback["format_valid"])

    def test_database_is_read_only(self):
        with self.assertRaises(sqlite3.Error):
            execute("DELETE FROM t", self.db_path)
        self.assertEqual(execute("SELECT COUNT(*) FROM t", self.db_path), [(3,)])


if __name__ == "__main__":
    unittest.main()

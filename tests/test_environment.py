import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sql_agent.demo import create_demo
from sql_agent.environment import Episode, SQLDatabase, evaluate


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tasks = [json.loads(line) for line in create_demo(self.temp.name).read_text().splitlines()]
        self.path = self.tasks[0]["database"]
        self.db = SQLDatabase(self.path)

    def test_gold_answers(self):
        expected = [[[540.0]], [["北京", 290.0], ["深圳", 250.0]], [[1], [3]], [[2], [5]]]
        for task, rows in zip(self.tasks, expected):
            self.assertEqual(self.db.query(task["gold_sql"])["rows"], rows)

    def test_write_and_attach_denied(self):
        for sql in ["DELETE FROM users", "DROP TABLE users", "PRAGMA user_version=4",
                    "ATTACH DATABASE ':memory:' AS other", "SELECT load_extension('anything')"]:
            with self.subTest(sql=sql), self.assertRaises(sqlite3.Error):
                self.db.query(sql)
        self.assertEqual(self.db.query("SELECT COUNT(*) FROM users")["rows"], [[5]])

    def test_truncation_and_timeout(self):
        self.assertTrue(SQLDatabase(self.path, max_rows=2).query("SELECT * FROM users")["truncated"])
        with self.assertRaises(sqlite3.OperationalError):
            SQLDatabase(self.path, timeout=0).query(
                "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n")

    def test_error_consumes_budget_and_submission_hides_reward(self):
        episode = Episode(self.db, max_calls=1)
        self.assertIn("error", episode.step("execute_sql", {"sql": "SELECT nonexistent FROM users"}))
        self.assertIn("error", episode.step("get_schema", {}))
        self.assertEqual(episode.step("submit_sql", {"sql": "SELECT 0"}), {"submitted": True})
        self.assertIn("error", episode.step("submit_sql", {"sql": "SELECT 1"}))

    def test_result_comparison_preserves_duplicates_and_order(self):
        self.assertFalse(evaluate(self.path, "SELECT city FROM users", "SELECT DISTINCT city FROM users")["correct"])
        asc, desc = "SELECT id FROM users ORDER BY id", "SELECT id FROM users ORDER BY id DESC"
        self.assertTrue(evaluate(self.path, asc, desc)["correct"])
        self.assertFalse(evaluate(self.path, asc, desc, ordered=True)["correct"])

    def test_empty_results_with_different_widths_are_not_equal(self):
        self.assertFalse(evaluate(self.path, "SELECT id,city FROM users WHERE 0", "SELECT id FROM users WHERE 0")["correct"])


if __name__ == "__main__":
    unittest.main()

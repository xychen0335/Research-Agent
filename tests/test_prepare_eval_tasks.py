import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_eval_tasks import convert


class PrepareEvalTasksTests(unittest.TestCase):
    def test_converts_frozen_record_to_existing_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_dir = root / "demo"
            db_dir.mkdir()
            database = db_dir / "demo.sqlite"
            sqlite3.connect(database).close()
            tasks = convert([{
                "question_id": 3,
                "db_id": "demo",
                "question": "Count rows",
                "evidence": "none",
                "SQL": "SELECT COUNT(*) FROM t",
            }], root)
            self.assertEqual(tasks[0]["id"], 3)
            self.assertEqual(tasks[0]["gold_sql"], "SELECT COUNT(*) FROM t")
            self.assertEqual(tasks[0]["database"], str(database.resolve()))
            self.assertNotIn(tasks[0]["gold_sql"], str(tasks[0]["messages"]))
            self.assertIn("Database schema", tasks[0]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()

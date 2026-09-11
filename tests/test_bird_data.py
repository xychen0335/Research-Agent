import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agenticrl.bird_data import rl_record, schema_text, sft_record


class BirdDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        db_dir = root / "demo"
        db_dir.mkdir()
        self.db_root = root
        self.db_path = db_dir / "demo.sqlite"
        conn = sqlite3.connect(self.db_path)
        conn.executescript("CREATE TABLE items(id INTEGER, name TEXT); INSERT INTO items VALUES(1, 'a');")
        conn.commit()
        conn.close()
        self.item = {"question_id": 7, "db_id": "demo", "question": "List names", "evidence": "",
                     "SQL": "SELECT name FROM items"}

    def test_rl_record_separates_prompt_and_ground_truth(self):
        record = rl_record(self.item, self.db_root, {"demo|items|name": "item name"})
        prompt = json.dumps(record["prompt"])
        self.assertNotIn(self.item["SQL"], prompt)
        self.assertEqual(record["reward_model"]["ground_truth"], self.item["SQL"])
        self.assertEqual(record["extra_info"]["tools_kwargs"]["execute_sql"]["create_kwargs"]["db_id"], "demo")
        self.assertIn("CREATE TABLE items", schema_text(self.db_path))

    def test_sft_trajectory_becomes_native_tool_messages(self):
        trajectory = [
            {"thought": "Inspect rows", "action": '<tool_call>{"name":"execute_sql","arguments":{"sql":"SELECT * FROM items"}}</tool_call>',
             "observation": "[(1, 'a')]"},
            {"thought": "Submit", "action": '<tool_call>{"name":"submit_solution","arguments":{"sql":"SELECT name FROM items"}}</tool_call>',
             "end_flag": True},
        ]
        record = sft_record(self.item, trajectory, self.db_root, {})
        roles = [message["role"] for message in record["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(record["messages"][-1]["tool_calls"][0]["function"]["name"], "submit_solution")
        self.assertFalse(record["enable_thinking"])


if __name__ == "__main__":
    unittest.main()

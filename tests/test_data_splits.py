import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_data_splits import build


class DataSplitTests(unittest.TestCase):
    def test_database_disjoint_sizes_and_determinism(self):
        train = [
            {"question_id": db * 10 + item, "db_id": "db_%d" % db, "question": "q", "SQL": "SELECT 1"}
            for db in range(10)
            for item in range(10)
        ]
        test = [
            {"question_id": item, "db_id": "test", "question": "q", "SQL": "SELECT 1"}
            for item in range(7)
        ]
        config = {
            "seed": 42,
            "train": {
                "expected_records": 100,
                "validation_database_fraction": 0.1,
                "sft_train_records": 20,
                "sft_validation_records": 15,
                "rl_train_records": 30,
                "rl_validation_records": 20,
            },
            "test": {"expected_records": 7},
        }
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            manifest = build(config, train, test, Path(first))
            build(config, list(reversed(train)), list(reversed(test)), Path(second))
            val_dbs = set(manifest["validation_databases"])
            rl_train = json.loads((Path(first) / "rl_train.json").read_text())
            rl_val = json.loads((Path(first) / "rl_val.json").read_text())
            self.assertTrue(val_dbs.isdisjoint({row["db_id"] for row in rl_train}))
            self.assertEqual(val_dbs, {row["db_id"] for row in rl_val})
            self.assertEqual(manifest["record_counts"]["rl_train"], 30)
            self.assertEqual(manifest["record_counts"]["rl_val"], 20)
            self.assertEqual((Path(first) / "rl_train.json").read_bytes(),
                             (Path(second) / "rl_train.json").read_bytes())
            self.assertEqual((Path(first) / "frozen_test.json").read_bytes(),
                             (Path(second) / "frozen_test.json").read_bytes())


if __name__ == "__main__":
    unittest.main()

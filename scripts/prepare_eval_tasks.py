"""Convert the frozen BIRD mini-dev split to the local evaluation task format."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agenticrl.bird_data import make_prompt


def ground_truth(row):
    for key in ("SQL", "sql", "gold_sql", "query"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("record %r has no reference SQL" % row.get("question_id", "?"))


def convert(records, database_root):
    database_root = Path(database_root).resolve()
    tasks = []
    for index, row in enumerate(records):
        db_id = row["db_id"]
        database = database_root / db_id / (db_id + ".sqlite")
        if not database.is_file():
            raise FileNotFoundError(str(database))
        tasks.append({
            "instance_idx": index,
            "id": row.get("question_id", index),
            "question_id": row.get("question_id", index),
            "db_id": db_id,
            "question": row["question"],
            "evidence": row.get("evidence", ""),
            "difficulty": row.get("difficulty"),
            "database": str(database),
            "gold_sql": ground_truth(row),
            "ordered": False,
            "messages": make_prompt(row, database_root, {}, max_turns=6),
        })
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data/splits/frozen_test.json"))
    parser.add_argument("--db-dir", default=str(ROOT / "data/databases/mini_dev"))
    parser.add_argument("--output", default=str(ROOT / "data/processed/bird_mini_dev_tasks.jsonl"))
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    tasks = convert(records, args.db_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(tasks), "output": str(output)}))


if __name__ == "__main__":
    main()

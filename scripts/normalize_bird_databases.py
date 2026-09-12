"""Normalize extracted BIRD archives to db_id/db_id.sqlite directories."""

import argparse
import json
import os
from pathlib import Path


def records(path):
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()] if Path(path).suffix == ".jsonl" else json.loads(text)


def normalize(extracted, record_file, destination):
    extracted = Path(extracted).resolve()
    destination = Path(destination).resolve()
    databases = {}
    for path in extracted.rglob("*.sqlite"):
        databases.setdefault(path.stem, []).append(path.resolve())
    required = sorted({row["db_id"] for row in records(record_file)})
    missing = [db_id for db_id in required if db_id not in databases]
    ambiguous = {db_id: paths for db_id, paths in databases.items() if db_id in required and len(paths) != 1}
    if missing or ambiguous:
        raise ValueError("database archive mismatch: missing=%r ambiguous=%r" % (missing, ambiguous))
    destination.mkdir(parents=True, exist_ok=True)
    for db_id in required:
        target_dir = destination / db_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (db_id + ".sqlite")
        relative = os.path.relpath(databases[db_id][0], target_dir)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(relative)
    return len(required)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    count = normalize(args.extracted, args.records, args.destination)
    print(json.dumps({"databases": count, "destination": args.destination}))


if __name__ == "__main__":
    main()

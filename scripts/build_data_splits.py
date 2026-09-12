"""Create deterministic database-disjoint BIRD train and validation subsets."""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_records(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def stable_key(seed, value):
    return hashlib.sha256((str(seed) + ":" + str(value)).encode()).hexdigest()


def take(records, count, seed, label):
    ordered = sorted(
        records,
        key=lambda row: stable_key(
            seed,
            "%s:%s" % (label, json.dumps(row, ensure_ascii=False, sort_keys=True)),
        ),
    )
    if len(ordered) < count:
        raise ValueError("%s requested %d records but only %d are available" % (label, count, len(ordered)))
    return ordered[:count]


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(config, train_records, test_records, output_dir):
    seed = config["seed"]
    expected_train = config["train"]["expected_records"]
    expected_test = config["test"]["expected_records"]
    if len(train_records) != expected_train or len(test_records) != expected_test:
        raise ValueError(
            "record count mismatch: train=%d expected=%d, test=%d expected=%d"
            % (len(train_records), expected_train, len(test_records), expected_test)
        )
    db_ids = sorted({row["db_id"] for row in train_records}, key=lambda value: stable_key(seed, value))
    train_cfg = config["train"]
    configured_val_dbs = train_cfg.get("validation_databases")
    if configured_val_dbs:
        unknown = set(configured_val_dbs) - set(db_ids)
        if unknown:
            raise ValueError("configured validation databases are missing: %r" % sorted(unknown))
        val_db_ids = set(configured_val_dbs)
        val_db_count = len(val_db_ids)
    else:
        val_db_count = max(1, math.ceil(len(db_ids) * train_cfg["validation_database_fraction"]))
        val_db_ids = set(db_ids[:val_db_count])
    minimum_val_records = max(train_cfg["sft_validation_records"], train_cfg["rl_validation_records"])
    records_per_db = {
        db_id: sum(row["db_id"] == db_id for row in train_records)
        for db_id in db_ids
    }
    remaining_db_ids = [db_id for db_id in db_ids if db_id not in val_db_ids]
    for db_id in remaining_db_ids:
        if sum(records_per_db[value] for value in val_db_ids) >= minimum_val_records:
            break
        val_db_ids.add(db_id)
    train_pool = [row for row in train_records if row["db_id"] not in val_db_ids]
    val_pool = [row for row in train_records if row["db_id"] in val_db_ids]
    outputs = {
        "sft_train": take(train_pool, train_cfg["sft_train_records"], seed, "sft_train"),
        "sft_val": take(val_pool, train_cfg["sft_validation_records"], seed, "sft_val"),
        "rl_train": take(train_pool, train_cfg["rl_train_records"], seed, "rl_train"),
        "rl_val": take(val_pool, train_cfg["rl_validation_records"], seed, "rl_val"),
        "frozen_test": sorted(test_records, key=lambda row: row.get("question_id", 0)),
    }
    for name, records in outputs.items():
        write_json(output_dir / (name + ".json"), records)
    manifest = {
        "seed": seed,
        "train_database_count": len(set(row["db_id"] for row in train_pool)),
        "validation_database_count": len(val_db_ids),
        "validation_database_fraction": len(val_db_ids) / len(db_ids),
        "validation_databases": sorted(val_db_ids),
        "pool_record_counts": {"train": len(train_pool), "validation": len(val_pool)},
        "record_counts": {name: len(records) for name, records in outputs.items()},
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    config = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
    raw = ROOT / "data/raw"
    output = ROOT / "data/splits"
    manifest = build(
        config,
        load_records(raw / "bird23-train-filtered/train.jsonl"),
        load_records(raw / "bird-mini-dev/mini_dev_sqlite.json"),
        output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

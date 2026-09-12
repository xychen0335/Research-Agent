"""Download the pinned BIRD records and Qwen model from Hugging Face."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def copy_dataset_file(repo_id, revision, filename, target):
    source = hf_hub_download(repo_id=repo_id, repo_type="dataset", revision=revision, filename=filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true", help="also download the pinned Qwen3.5-4B snapshot")
    args = parser.parse_args()
    data = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
    upstreams = json.loads((ROOT / "configs/upstreams.json").read_text(encoding="utf-8"))
    train_target = ROOT / "data/raw/bird23-train-filtered"
    test_target = ROOT / "data/raw/bird-mini-dev"
    copy_dataset_file(
        data["train"]["repo_id"], data["train"]["revision"], data["train"]["records_file"],
        train_target / "train.jsonl",
    )
    copy_dataset_file(
        data["train"]["repo_id"], data["train"]["revision"], data["train"]["column_meaning_file"],
        train_target / "column_meaning.json",
    )
    copy_dataset_file(
        data["test"]["repo_id"], data["test"]["revision"], data["test"]["records_file"],
        test_target / "mini_dev_sqlite.json",
    )
    (test_target / "column_meaning.json").write_text("{}\n", encoding="utf-8")
    if args.model:
        model = upstreams["model"]
        local_dir = ROOT / "models" / ("Qwen3.5-4B-" + model["revision"][:8])
        snapshot_download(repo_id=model["repo_id"], revision=model["revision"], local_dir=local_dir)
        print("model=" + str(local_dir))
    print("train_records=" + str(train_target / "train.jsonl"))
    print("mini_dev_records=" + str(test_target / "mini_dev_sqlite.json"))


if __name__ == "__main__":
    main()

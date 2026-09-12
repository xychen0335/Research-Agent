"""Measure processed SFT or RL records with the pinned model tokenizer."""

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def percentile(values, fraction):
    if not values:
        return 0
    return sorted(values)[round((len(values) - 1) * fraction)]


def load_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        return parquet.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoProcessor
    from agenticrl.bird_data import TOOLS

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = getattr(processor, "tokenizer", processor)
    reports = []
    failed = False
    for filename in args.input:
        rows = load_rows(filename)
        lengths = []
        overlong = []
        for index, row in enumerate(rows):
            messages = row.get("messages", row.get("prompt"))
            tools = row.get("tools", TOOLS if "prompt" in row else None)
            rendered = processor.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt="prompt" in row,
            )
            length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            lengths.append(length)
            if length > args.max_length:
                overlong.append(index)
        report = {
            "input": str(Path(filename)),
            "records": len(rows),
            "min": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0,
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
            "limit": args.max_length,
            "overlong_count": len(overlong),
            "overlong_indices": overlong[:100],
        }
        reports.append(report)
        failed = failed or bool(overlong)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit("processed records exceed the configured token limit")


if __name__ == "__main__":
    main()

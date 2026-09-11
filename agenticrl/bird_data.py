"""Build native tool-calling SFT and RL records from BIRD data."""

import argparse
import json
import re
import sqlite3
from pathlib import Path


SYSTEM_PROMPT = """You are a SQLite text-to-SQL agent. Solve the user's question against the provided database. Use execute_sql to inspect data and test candidate queries. Call submit_solution exactly once when the SQL is ready. Each turn may call one tool. Only issue read-only queries."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute one read-only SQLite query for exploration. This does not reveal correctness.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "description": "Submit the final SQL and end the episode.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
]


def load_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("BIRD dataset must contain a JSON array or JSONL records")
    return value


def schema_text(db_path, sample_rows=3):
    """Return DDL and small samples without exposing a writable connection."""
    database = Path(db_path).resolve()
    if not database.is_file():
        raise FileNotFoundError(str(database))
    conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    parts = []
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for table, ddl in tables:
            if ddl:
                parts.append(ddl + ";")
            quoted = '"' + table.replace('"', '""') + '"'
            columns = [row[1] for row in conn.execute("PRAGMA table_info(" + quoted + ")")]
            rows = conn.execute("SELECT * FROM " + quoted + " LIMIT ?", (sample_rows,)).fetchall()
            if rows:
                parts.append("Sample rows:")
                parts.append("  " + " | ".join(columns))
                parts.extend("  " + " | ".join("NULL" if v is None else str(v) for v in row) for row in rows)
            parts.append("")
    finally:
        conn.close()
    return "\n".join(parts).strip()


def column_descriptions(db_id, meanings):
    prefix = db_id + "|"
    rows = []
    for key, description in sorted(meanings.items()):
        if key.startswith(prefix) and key.count("|") == 2:
            _, table, column = key.split("|")
            rows.append("- %s.%s: %s" % (table, column, description))
    return "\n".join(rows) if rows else "(none provided)"


def ground_truth(item):
    for key in ("SQL", "sql", "gold_sql", "query"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("record %r has no SQL/ground truth field" % item.get("question_id", "?"))


def make_prompt(item, db_dir, meanings, max_turns=6):
    db_id = item["db_id"]
    db_path = Path(db_dir) / db_id / (db_id + ".sqlite")
    user = """Database schema and sample rows:
{schema}

Column descriptions:
{columns}

Question:
{question}

Evidence:
{evidence}

You have at most {turns} assistant turns. Explore only when useful, then submit the final SQL.""".format(
        schema=schema_text(db_path),
        columns=column_descriptions(db_id, meanings),
        question=item["question"].strip(),
        evidence=(item.get("evidence") or "(none provided)").strip(),
        turns=max_turns,
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def rl_record(item, db_dir, meanings, max_turns=6):
    db_id = item["db_id"]
    return {
        "data_source": "bird/sql_generation",
        "prompt": make_prompt(item, db_dir, meanings, max_turns),
        "agent_name": "bird_sql_agent",
        "reward_model": {"style": "rule", "ground_truth": ground_truth(item)},
        "extra_info": {
            "db_id": db_id,
            "question_id": item.get("question_id"),
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "execute_sql": {"create_kwargs": {"db_id": db_id}},
                "submit_solution": {"create_kwargs": {"db_id": db_id}},
            },
        },
    }


def _parse_tool_call(text):
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text or "", re.DOTALL)
    if not matches:
        return None
    return json.loads(matches[-1].strip().strip("`"))


def sft_record(item, trajectory, db_dir, meanings, max_turns=6):
    messages = make_prompt(item, db_dir, meanings, max_turns)
    for index, turn in enumerate(trajectory):
        call = _parse_tool_call(turn.get("action", ""))
        if not call:
            raise ValueError("trajectory turn %d has no valid <tool_call>" % index)
        call_id = "bird_%s_%d" % (item.get("question_id", "sample"), index)
        thought = (turn.get("thought") or "").strip()
        messages.append({
            "role": "assistant",
            "content": thought or None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                },
            }],
        })
        if call["name"] != "submit_solution":
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(turn.get("observation", "")),
            })
    return {"messages": messages, "tools": TOOLS, "enable_thinking": False}


def write_records(records, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("Parquet output requires `pip install datasets pyarrow`") from exc
        Dataset.from_list(records).to_parquet(str(output))
    else:
        output.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rl", "sft"), required=True)
    parser.add_argument("--data", required=True, help="BIRD JSON/JSONL with question, db_id, and SQL")
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--column-meaning", required=True)
    parser.add_argument("--output", required=True, help=".jsonl or .parquet")
    parser.add_argument("--trajectories", help="BIRD-RL trajectory JSONL, required for SFT")
    parser.add_argument("--evaluation", help="BIRD-RL eval_results.json; only correct trajectories become SFT data")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    items = load_json_or_jsonl(args.data)
    if args.limit is not None:
        items = items[: args.limit]
    meanings = json.loads(Path(args.column_meaning).read_text(encoding="utf-8"))
    if args.mode == "rl":
        records = [rl_record(item, args.db_dir, meanings, args.max_turns) for item in items]
    else:
        if not args.trajectories or not args.evaluation:
            parser.error("--trajectories and --evaluation are required in sft mode")
        trajectories = load_json_or_jsonl(args.trajectories)
        by_index = {row.get("instance_idx", i): row.get("trajectory", []) for i, row in enumerate(trajectories)}
        evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
        correct_indices = {row["instance_idx"] for row in evaluation.get("results", []) if row.get("correct")}
        records = []
        for index, item in enumerate(items):
            if index in correct_indices and by_index.get(index):
                records.append(sft_record(item, by_index[index], args.db_dir, meanings, args.max_turns))
    write_records(records, args.output)
    print(json.dumps({"mode": args.mode, "records": len(records), "output": args.output}))


if __name__ == "__main__":
    main()

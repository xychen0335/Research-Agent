"""Load model profiles and aggregate saved evaluation episodes."""

import json
from pathlib import Path


def load_profiles(path="configs/models.json"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = data.get("profiles", data)
    required = {"label", "base_url", "model"}
    for name, profile in profiles.items():
        missing = required - set(profile)
        if missing:
            raise ValueError("profile %s is missing: %s" % (name, ", ".join(sorted(missing))))
    return profiles


def load_runs(root="outputs"):
    records = []
    for path in sorted(Path(root).glob("**/*.jsonl")) if Path(root).exists() else []:
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not {"model", "elapsed_seconds", "prediction"}.issubset(row):
                        continue
                    row["source_file"] = str(path)
                    row["source_line"] = line_number
                    records.append(row)
        except (OSError, json.JSONDecodeError):
            continue
    return records


def summarize(records):
    grouped = {}
    for row in records:
        name = row.get("profile") or row.get("model") or "unknown"
        stats = grouped.setdefault(name, {"profile": name, "episodes": 0, "scored": 0,
                                          "correct": 0, "tool_calls": 0, "elapsed_seconds": 0.0})
        stats["episodes"] += 1
        stats["tool_calls"] += int(row.get("tool_calls") or 0)
        stats["elapsed_seconds"] += float(row.get("elapsed_seconds") or 0.0)
        score = row.get("score")
        if isinstance(score, dict) and "correct" in score:
            stats["scored"] += 1
            stats["correct"] += int(bool(score["correct"]))
    result = []
    for stats in grouped.values():
        scored = stats["scored"]
        episodes = stats["episodes"]
        stats["accuracy"] = stats["correct"] / scored if scored else None
        stats["avg_tool_calls"] = stats["tool_calls"] / episodes if episodes else None
        stats["avg_elapsed_seconds"] = stats["elapsed_seconds"] / episodes if episodes else None
        result.append(stats)
    return sorted(result, key=lambda row: row["profile"])

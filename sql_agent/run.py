"""Evaluate native tool-calling through an OpenAI-compatible model endpoint."""

import argparse
import copy
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from agenticrl.bird_data import SYSTEM_PROMPT, TOOLS
from .environment import Episode, SQLDatabase, evaluate


def completion(base_url, api_key, payload):
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
        **({"Authorization": "Bearer " + api_key} if api_key else {})})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("Model endpoint returned HTTP " + str(exc.code)) from None


def rollout(task, args):
    episode = Episode(SQLDatabase(task["database"]), args.max_calls)
    if task.get("messages"):
        messages = copy.deepcopy(task["messages"])
    else:
        schema = episode.database.schema()
        evidence = task.get("evidence") or "(none provided)"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Database schema:\n%s\n\nQuestion:\n%s\n\nEvidence:\n%s\n\n"
             "You have at most %d assistant turns. Explore only when useful, then submit the final SQL."
             % (json.dumps(schema, ensure_ascii=False), task["question"], evidence, args.max_calls)},
        ]
    usage, responses = [], []
    started = time.monotonic()
    for _ in range(args.max_calls + 3):
        api_key_env = getattr(args, "api_key_env", "SQL_AGENT_API_KEY")
        response = completion(args.base_url, os.environ.get(api_key_env, ""), {
            "model": args.model, "messages": messages, "tools": TOOLS,
            "tool_choice": "auto", "parallel_tool_calls": False,
            "temperature": 0.0, "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}})
        responses.append(response)
        usage.append(response.get("usage"))
        message = response["choices"][0]["message"]
        # Retain native tool IDs/arguments; omit provider-specific message fields.
        messages.append({k: v for k, v in message.items()
                         if k in {"role", "content", "tool_calls"}})
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": "Use a tool. If the query is ready, call submit_solution."})
            continue
        if len(calls) > 1:
            for call in calls:
                messages.append({"role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps({"error": "Call exactly one tool per response"})})
            continue
        call = calls[0]
        try:
            arguments = json.loads(call["function"]["arguments"])
            observation = episode.step(call["function"]["name"], arguments)
        except (ValueError, TypeError):
            observation = {"error": "Invalid JSON arguments"}
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "content": json.dumps(observation, ensure_ascii=False, default=str)})
        if episode.final_sql is not None:
            break
    # Scoring happens after rollout and never enters the agent's messages.
    score = None
    if task.get("gold_sql"):
        score = evaluate(task["database"], episode.final_sql, task["gold_sql"], task.get("ordered", False))
    result = None
    result_error = None
    if episode.final_sql:
        try:
            result = episode.database.query(episode.final_sql)
        except Exception as exc:
            result_error = str(exc)
    return {"task_id": task.get("id", "interactive"), "instance_idx": task.get("instance_idx"),
            "question": task.get("question"), "db_id": task.get("db_id"),
            "difficulty": task.get("difficulty"), "model": args.model,
            "profile": getattr(args, "profile", None), "mode": "non_thinking",
            "temperature": 0.0, "max_calls": args.max_calls, "max_tokens": args.max_tokens,
            "tool_calls": episode.calls, "elapsed_seconds": time.monotonic() - started,
            "prediction": episode.final_sql, "score": score,
            "query_result": result, "query_error": result_error,
            "messages": messages, "usage": usage, "responses": responses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/generated/tasks.jsonl")
    parser.add_argument("--base-url", default=os.environ.get("SQL_AGENT_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--api-key-env", default="SQL_AGENT_API_KEY")
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.max_calls < 1 or args.max_tokens < 1:
        parser.error("budgets must be positive")
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tasks:
        parser.error("task file is empty")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(args.output or ("outputs/" + stamp + ".jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with output.open("x", encoding="utf-8") as handle:
        for task in tasks:
            result = rollout(task, args)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            correct += int(bool(result["score"] and result["score"]["correct"]))
            print(task["id"], result["score"], "calls=", result["tool_calls"])
    print(json.dumps({"correct": correct, "total": len(tasks), "output": str(output)}))


if __name__ == "__main__":
    main()

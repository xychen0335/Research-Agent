"""Dependency-free execution reward for BIRD SQL tool trajectories."""

import json
import os
import re
import sqlite3
import time
from pathlib import Path


TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)


def calls(response):
    parsed = []
    for raw in TOOL_CALL.findall(response or ""):
        try:
            value = json.loads(raw.strip().strip("`"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def solution(response):
    for call in reversed(calls(response)):
        if call.get("name") != "submit_solution":
            continue
        arguments = call.get("arguments", {})
        sql = arguments.get("sql") if isinstance(arguments, dict) else None
        if isinstance(sql, str) and sql.strip():
            return sql.strip(), True
    for call in reversed(calls(response)):
        if call.get("name") == "execute_sql" and isinstance(call.get("arguments"), dict):
            sql = call["arguments"].get("sql")
            if isinstance(sql, str) and sql.strip():
                return sql.strip(), False
    return None, False


def execute(sql, path, timeout=30.0):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    started = time.monotonic()
    try:
        conn.execute("PRAGMA query_only=ON")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() == "load_extension":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        conn.set_authorizer(authorize)
        conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout), 1000)
        cursor = conn.execute(sql)
        if cursor.description is None:
            raise ValueError("query must return rows")
        return cursor.fetchall()
    finally:
        conn.close()


def normalize(rows):
    return {
        tuple(round(value, 10) if isinstance(value, float) else value for value in row)
        for row in rows
    }


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    predicted_sql, submitted = solution(solution_str)
    result = {
        "score": 0.0,
        "format_valid": bool(predicted_sql and submitted),
        "submitted": submitted,
        "execution_success": False,
        "ex_score": 0,
        "error": "",
    }
    if not predicted_sql:
        result["error"] = "no_sql_tool_call"
        return result
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("ground_truth", "")
    db_id = (extra_info or {}).get("db_id", "")
    database_root = os.environ.get("BIRD_DB_DIR")
    if not database_root:
        result["error"] = "BIRD_DB_DIR_not_set"
        return result
    root = Path(database_root).resolve()
    db_path = (root / db_id / (db_id + ".sqlite")).resolve()
    if root not in db_path.parents or not db_path.is_file():
        result["error"] = "database_not_found"
        return result
    try:
        predicted = execute(predicted_sql, db_path, float(os.environ.get("BIRD_SQL_TIMEOUT", "30")))
        expected = execute(str(ground_truth), db_path, float(os.environ.get("BIRD_SQL_TIMEOUT", "30")))
        result["execution_success"] = True
        result["ex_score"] = int(normalize(predicted) == normalize(expected))
        result["score"] = 1.0 if result["ex_score"] else (0.1 if submitted else 0.05)
    except Exception as exc:
        result["error"] = str(exc)
    return result

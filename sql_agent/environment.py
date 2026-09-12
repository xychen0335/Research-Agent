"""Read-only SQLite tools; gold answers are kept outside the agent context."""

import sqlite3
import time
from collections import Counter
from pathlib import Path


class SQLDatabase:
    def __init__(self, path, timeout=2.0, max_rows=100):
        self.path = Path(path).resolve()
        self.timeout = timeout
        self.max_rows = max_rows

    def query(self, sql):
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("SQL must be a nonempty string")
        if len(sql) > 20000:
            raise ValueError("SQL exceeds the 20,000-character limit")
        conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        started = time.monotonic()
        try:
            conn.execute("PRAGMA query_only=ON")
            # Permit reads, expressions and CTEs; reject writes, ATTACH and PRAGMA.
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                       sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

            def authorize(action, arg1, arg2, database, trigger):
                if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() == "load_extension":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

            conn.set_authorizer(authorize)
            conn.set_progress_handler(
                lambda: int(time.monotonic() - started > self.timeout), 1000)
            cursor = conn.execute(sql)
            if cursor.description is None:
                raise ValueError("Query must return rows")
            rows = cursor.fetchmany(self.max_rows + 1)
            return {"columns": [c[0] for c in cursor.description],
                    "rows": [list(row) for row in rows[:self.max_rows]],
                    "truncated": len(rows) > self.max_rows}
        finally:
            conn.close()

    def schema(self):
        return self.query("SELECT name, sql FROM sqlite_master "
                          "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")

    def sample(self, table):
        names = {row[0] for row in self.schema()["rows"]}
        if table not in names:
            raise ValueError("Unknown table")
        quoted = '"' + table.replace('"', '""') + '"'
        return self.query("SELECT * FROM " + quoted + " LIMIT 5")


class Episode:
    def __init__(self, database, max_calls=6):
        self.database = database
        self.max_calls = max_calls
        self.calls = 0
        self.final_sql = None

    def step(self, name, arguments):
        if self.final_sql is not None:
            return {"error": "Episode already submitted"}
        if not isinstance(arguments, dict):
            return {"error": "Tool arguments must be an object"}
        if name in {"submit_solution", "submit_sql"}:
            sql = arguments.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                return {"error": "Provide a nonempty sql string"}
            self.final_sql = sql
            return {"submitted": True}
        if self.calls >= self.max_calls:
            return {"error": "Tool budget exhausted; submit_solution now"}
        self.calls += 1
        try:
            if name == "get_schema":
                result = self.database.schema()
            elif name == "sample_values":
                result = self.database.sample(arguments["table"])
            elif name == "execute_sql":
                result = self.database.query(arguments["sql"])
            else:
                raise ValueError("Unknown tool")
            return {"result": result, "remaining_calls": self.max_calls - self.calls}
        except (sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            return {"error": str(exc), "remaining_calls": self.max_calls - self.calls}


def evaluate(database_path, prediction, gold_sql, ordered=False):
    """Exact execution comparison for smoke tasks, not official BIRD scoring."""
    database = SQLDatabase(database_path, max_rows=10000)
    gold = database.query(gold_sql)
    if gold["truncated"]:
        raise ValueError("Gold result exceeds evaluator limit; cannot score")
    if not prediction:
        return {"correct": False, "error": "No submission"}
    try:
        actual = database.query(prediction)
    except (sqlite3.Error, ValueError) as exc:
        return {"correct": False, "error": str(exc)}
    if actual["truncated"]:
        return {"correct": False, "error": "Prediction exceeds evaluator limit"}
    left, right = [tuple(r) for r in actual["rows"]], [tuple(r) for r in gold["rows"]]
    equal = left == right if ordered else Counter(left) == Counter(right)
    return {"correct": equal and len(actual["columns"]) == len(gold["columns"])}

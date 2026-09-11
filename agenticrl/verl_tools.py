"""Read-only BIRD SQLite tools for verl's native agent loop."""

import asyncio
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse


def _database_path(db_id):
    root = Path(os.environ["BIRD_DB_DIR"]).resolve()
    candidate = (root / db_id / (db_id + ".sqlite")).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise ValueError("unknown BIRD database: " + db_id)
    return candidate


def _read(sql, path, max_rows, timeout):
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() == "load_extension":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        conn.set_authorizer(authorize)
        started = time.monotonic()
        conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout), 1000)
        cursor = conn.execute(sql)
        if cursor.description is None:
            raise ValueError("query must return rows")
        rows = cursor.fetchmany(max_rows + 1)
        return {
            "columns": [column[0] for column in cursor.description],
            "rows": [list(row) for row in rows[:max_rows]],
            "truncated": len(rows) > max_rows,
        }
    finally:
        conn.close()


class _BirdTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.instances = {}

    async def create(self, instance_id=None, **kwargs):
        instance_id = instance_id or str(uuid4())
        db_id = kwargs.get("create_kwargs", {}).get("db_id")
        if not db_id:
            raise ValueError("create_kwargs.db_id is required")
        self.instances[instance_id] = {"db_id": db_id, "path": _database_path(db_id)}
        return instance_id, ToolResponse(text="ready")

    async def release(self, instance_id, **kwargs):
        self.instances.pop(instance_id, None)


class ReadOnlyBirdSqlTool(_BirdTool):
    async def execute(self, instance_id, parameters, **kwargs):
        sql = parameters.get("sql", "")
        if not isinstance(sql, str) or not sql.strip():
            return ToolResponse(text="Error: sql must be a nonempty string"), 0.0, {}
        state = self.instances[instance_id]
        max_rows = int(os.environ.get("BIRD_SQL_MAX_ROWS", "100"))
        timeout = float(os.environ.get("BIRD_SQL_TIMEOUT", "10"))
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_read, sql, state["path"], max_rows, timeout), timeout=timeout + 1
            )
            return ToolResponse(text=str(result)), 0.0, {"sql_executed": True}
        except Exception as exc:
            return ToolResponse(text="Execution failed: " + str(exc)), 0.0, {"sql_executed": False}


class SubmitBirdSqlTool(_BirdTool):
    async def execute(self, instance_id, parameters, **kwargs):
        sql = parameters.get("sql", "")
        if not isinstance(sql, str) or not sql.strip():
            return ToolResponse(text="Error: sql must be a nonempty string"), 0.0, {}
        agent_data = kwargs.get("agent_data")
        if agent_data is not None:
            agent_data.extra_fields["bird_submitted"] = True
            agent_data.extra_fields["bird_solution_sql"] = sql.strip()
        return ToolResponse(text="Solution accepted; the episode is complete."), 0.0, {"submitted": True}

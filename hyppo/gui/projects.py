from __future__ import annotations

import sqlite3
import uuid


class ProjectStore:
    """Projects live only at the GUI layer; core has no project concept."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # One persistent connection per store: with ":memory:" every fresh
        # sqlite3.connect() would be a *separate* database, so tables created
        # here would be invisible to later calls. check_same_thread=False lets
        # the uvicorn worker threads share it.
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS projects ("
                "id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                "description TEXT, created_order INTEGER)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS ve_defs ("
                "project_id TEXT PRIMARY KEY, payload TEXT)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS iterations ("
                "project_id TEXT, iteration INTEGER, payload TEXT, "
                "PRIMARY KEY (project_id, iteration))"
            )

    def _conn(self) -> sqlite3.Connection:
        # `with conn` commits/rolls back the transaction without closing it.
        return self._connection

    def create(self, name: str, description: str = "") -> str:
        pid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            c.execute(
                "INSERT INTO projects VALUES (?,?,?,?)",
                (pid, name, description, n),
            )
        return pid

    def get(self, pid: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM projects ORDER BY created_order").fetchall()
        return [dict(r) for r in rows]

    def delete(self, pid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM projects WHERE id=?", (pid,))

    def save_ve(self, pid: str, payload: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO ve_defs VALUES (?,?) "
                "ON CONFLICT(project_id) DO UPDATE SET payload=excluded.payload",
                (pid, payload),
            )

    def load_ve(self, pid: str) -> str | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT payload FROM ve_defs WHERE project_id=?", (pid,)
            ).fetchone()
        return row[0] if row else None

    def add_iteration(self, pid: str, payload: str) -> int:
        with self._conn() as c:
            n = c.execute(
                "SELECT COALESCE(MAX(iteration),0) FROM iterations WHERE project_id=?",
                (pid,),
            ).fetchone()[0]
            it = n + 1
            c.execute("INSERT INTO iterations VALUES (?,?,?)", (pid, it, payload))
        return it

    def list_iterations(self, pid: str) -> list[dict]:
        import json as _json

        with self._conn() as c:
            rows = c.execute(
                "SELECT payload FROM iterations WHERE project_id=? "
                "ORDER BY iteration DESC",
                (pid,),
            ).fetchall()
        return [_json.loads(r[0]) for r in rows]

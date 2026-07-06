import sqlite3
import uuid


class ProjectStore:
    """Projects live only at the GUI layer; core has no project concept."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS projects ("
                "id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                "description TEXT, created_order INTEGER)"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
            rows = c.execute(
                "SELECT * FROM projects ORDER BY created_order"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, pid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM projects WHERE id=?", (pid,))

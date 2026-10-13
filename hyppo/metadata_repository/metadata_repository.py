"""MetadataRepository — persistent storage for virtual experiment artifacts.

Implements the repository described in Section 3.1.8 of the dissertation.
Uses SQLite for storage with composite key (hypothesis_id, config_hash) for caching.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from hyppo.core._types import Metrics


@dataclass
class ResultRecord:
    hypothesis_id: str
    config: dict
    metrics: Metrics
    status: str  # SUCCESS, FAILED, SKIPPED
    timestamp: str | None = None


class MetadataRepository:
    """Relational repository for virtual experiment metadata (Section 3.1.8)."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS results (
                hypothesis_id TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'SUCCESS',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (hypothesis_id, config_hash)
            );
            CREATE TABLE IF NOT EXISTS marks (
                kind TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                PRIMARY KEY (kind, hypothesis_id)
            );
            CREATE TABLE IF NOT EXISTS epochs (
                hypothesis_id TEXT PRIMARY KEY,
                epoch INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS graphs (
                graph_id TEXT PRIMARY KEY,
                nodes_json TEXT NOT NULL,
                edges_json TEXT NOT NULL,
                created DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    @staticmethod
    def _config_hash(config: dict) -> str:
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    def save_result(
        self,
        hypothesis_id: str,
        config: dict,
        metrics: Metrics,
        status: str = "SUCCESS",
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO results "
            "(hypothesis_id, config_hash, config_json, metrics_json, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                hypothesis_id,
                self._config_hash(config),
                json.dumps(config),
                json.dumps(metrics),
                status,
            ),
        )
        self._conn.commit()

    def load_result(self, hypothesis_id: str, config: dict) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM results WHERE hypothesis_id = ? AND config_hash = ?",
            (hypothesis_id, self._config_hash(config)),
        ).fetchone()
        if row is None:
            return None
        return {
            "hypothesis_id": row["hypothesis_id"],
            "config": json.loads(row["config_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "status": row["status"],
            "timestamp": row["timestamp"],
        }

    def load_marks(self, kind: str) -> set[str]:
        """Hypothesis ids stored under ``kind`` (e.g. pending stale marks)."""
        rows = self._conn.execute(
            "SELECT hypothesis_id FROM marks WHERE kind = ?", (kind,)
        ).fetchall()
        return {r["hypothesis_id"] for r in rows}

    def save_marks(self, kind: str, ids: set[str]) -> None:
        """Replace the set of hypothesis ids stored under ``kind``."""
        self._conn.execute("DELETE FROM marks WHERE kind = ?", (kind,))
        self._conn.executemany(
            "INSERT INTO marks (kind, hypothesis_id) VALUES (?, ?)",
            [(kind, h) for h in sorted(ids)],
        )
        self._conn.commit()

    def load_epochs(self) -> dict[str, int]:
        """Revision epochs {hypothesis_id: epsilon} (absent means 0)."""
        rows = self._conn.execute("SELECT hypothesis_id, epoch FROM epochs").fetchall()
        return {r["hypothesis_id"]: r["epoch"] for r in rows}

    def save_epochs(self, epochs: dict[str, int]) -> None:
        """Replace the stored revision epochs (zero epochs are not stored)."""
        self._conn.execute("DELETE FROM epochs")
        self._conn.executemany(
            "INSERT INTO epochs (hypothesis_id, epoch) VALUES (?, ?)",
            [(h, e) for h, e in sorted(epochs.items()) if e],
        )
        self._conn.commit()

    def evict(self, hypothesis_id: str, config: dict) -> bool:
        """Remove the stored result of ``hypothesis_id`` under ``config`` (cache
        eviction). Returns True if a record was removed. After evicting a source
        while keeping its dependents the repository is no longer closed under
        sources (theorem thm:safe (iv))."""
        cur = self._conn.execute(
            "DELETE FROM results WHERE hypothesis_id = ? AND config_hash = ?",
            (hypothesis_id, self._config_hash(config)),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def has_result(self, hypothesis_id: str, config: dict) -> bool:
        return self.load_result(hypothesis_id, config) is not None

    def save_graph(
        self, graph_id: str, nodes: set[str], edges: set[tuple[str, str]]
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO graphs "
            "(graph_id, nodes_json, edges_json) VALUES (?, ?, ?)",
            (
                graph_id,
                json.dumps(sorted(nodes)),
                json.dumps(sorted([list(e) for e in edges])),
            ),
        )
        self._conn.commit()

    def find_nearest_graph(
        self, nodes: set[str], edges: set[tuple[str, str]]
    ) -> dict | None:
        """Find nearest graph by Definition 12: d(L, L_j) = |V△V_j| + |E△E_j|."""
        rows = self._conn.execute("SELECT * FROM graphs").fetchall()
        if not rows:
            return None
        best, best_dist = None, float("inf")
        for row in rows:
            stored_nodes = set(json.loads(row["nodes_json"]))
            stored_edges = {tuple(e) for e in json.loads(row["edges_json"])}
            dist = len(nodes ^ stored_nodes) + len(edges ^ stored_edges)
            if dist < best_dist:
                best, best_dist = row, dist
        if best is None:
            return None
        return {
            "graph_id": best["graph_id"],
            "nodes": set(json.loads(best["nodes_json"])),
            "edges": {tuple(e) for e in json.loads(best["edges_json"])},
            "distance": best_dist,
        }

    def close(self) -> None:
        self._conn.close()

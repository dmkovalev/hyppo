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


@dataclass
class ResultRecord:
    hypothesis_id: str
    config: dict
    metrics: dict
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
            CREATE TABLE IF NOT EXISTS lattices (
                lattice_id TEXT PRIMARY KEY,
                nodes_json TEXT NOT NULL,
                edges_json TEXT NOT NULL,
                created DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    @staticmethod
    def _config_hash(config: dict) -> str:
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

    def save_result(self, hypothesis_id: str, config: dict, metrics: dict, status: str = "SUCCESS") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO results (hypothesis_id, config_hash, config_json, metrics_json, status) VALUES (?, ?, ?, ?, ?)",
            (hypothesis_id, self._config_hash(config), json.dumps(config), json.dumps(metrics), status),
        )
        self._conn.commit()

    def load_result(self, hypothesis_id: str, config: dict) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM results WHERE hypothesis_id = ? AND config_hash = ?",
            (hypothesis_id, self._config_hash(config)),
        ).fetchone()
        if row is None:
            return None
        return {"hypothesis_id": row["hypothesis_id"], "config": json.loads(row["config_json"]),
                "metrics": json.loads(row["metrics_json"]), "status": row["status"], "timestamp": row["timestamp"]}

    def has_result(self, hypothesis_id: str, config: dict) -> bool:
        return self.load_result(hypothesis_id, config) is not None

    def save_lattice(self, lattice_id: str, nodes: set[str], edges: set[tuple[str, str]]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO lattices (lattice_id, nodes_json, edges_json) VALUES (?, ?, ?)",
            (lattice_id, json.dumps(sorted(nodes)), json.dumps(sorted([list(e) for e in edges]))),
        )
        self._conn.commit()

    def find_nearest_lattice(self, nodes: set[str], edges: set[tuple[str, str]]) -> dict | None:
        """Find nearest lattice by Definition 12: d(L, L_j) = |V triangle V_j| + |E triangle E_j|."""
        rows = self._conn.execute("SELECT * FROM lattices").fetchall()
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
        return {"lattice_id": best["lattice_id"], "nodes": set(json.loads(best["nodes_json"])),
                "edges": {tuple(e) for e in json.loads(best["edges_json"])}, "distance": best_dist}

    def close(self) -> None:
        self._conn.close()

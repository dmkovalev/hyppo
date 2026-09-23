# examples/research/baseline_filecache/metastore.py
"""Общее хранилище метаинформации (SQLite): сущности, которые правила конвейера
читают в рантайме, не объявляя входом."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

_DDL = ("CREATE TABLE IF NOT EXISTS meta (entity TEXT, token TEXT, version TEXT, "
        "value TEXT, PRIMARY KEY (entity, token))")


def _con(db) -> sqlite3.Connection:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db), timeout=60)
    con.execute(_DDL)
    return con


def write(db, entity: str, token: str, value, version: str = "v1") -> None:
    con = _con(db)
    try:
        con.execute("INSERT OR REPLACE INTO meta VALUES (?,?,?,?)",
                    (entity, token, version, json.dumps(np.asarray(value, float).tolist())))
        con.commit()
    finally:
        con.close()


def read(db, entity: str, token: str) -> tuple[str, np.ndarray]:
    con = _con(db)
    try:
        row = con.execute("SELECT version, value FROM meta WHERE entity=? AND token=?",
                          (entity, token)).fetchone()
    finally:
        con.close()
    if row is None:
        raise RuntimeError(f"{entity}/{token} отсутствует в {db}")
    return row[0], np.array(json.loads(row[1]), float)


def version(db, entity: str, token: str) -> str:
    try:
        return read(db, entity, token)[0]
    except (RuntimeError, sqlite3.OperationalError):
        return "missing"


def _tokens(db, entity: str) -> list[str]:
    con = _con(db)
    try:
        return [r[0] for r in con.execute("SELECT token FROM meta WHERE entity=?", (entity,))]
    finally:
        con.close()


def _bump(v: str) -> str:
    return f"v{int(v[1:]) + 1}"


def revise(db, entity: str) -> dict:
    """Внешняя ревизия сущности (версия +1). Возвращает описание правки."""
    if entity == "gains":
        _, ref = read(db, "gains", "pos")
        r, c = np.unravel_index(int(np.argmax(np.abs(ref))), ref.shape)
        for t in _tokens(db, "gains"):
            v, g = read(db, "gains", t)
            g[r, c] *= 1.5
            write(db, "gains", t, g, _bump(v))
        return {"entity": "gains", "channel": [int(r), int(c)], "factor": 1.5}
    if entity == "tau":
        for t in _tokens(db, "tau"):
            v, x = read(db, "tau", t)
            write(db, "tau", t, x * 2.0, _bump(v))
        return {"entity": "tau", "factor": 2.0}
    if entity == "well_status":
        _, ref = read(db, "gains", "pos")
        j = int(np.argmax(ref.sum(axis=0)))
        v, st = read(db, "well_status", "*")
        st[j] = 0.0
        write(db, "well_status", "*", st, _bump(v))
        return {"entity": "well_status", "shut_producer": j}
    if entity == "corey_ref":
        v, cr = read(db, "corey_ref", "*")
        cr[1] = 0.30
        write(db, "corey_ref", "*", cr, _bump(v))
        return {"entity": "corey_ref", "Sor": 0.30}
    raise ValueError(entity)

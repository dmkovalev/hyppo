"""SharedCache — единый SQLite-кэш для планировщика и раннера.

Раньше кэш существовал в двух несвязанных бэкендах: ``MetadataRepository``
(SQLite, ключ ``(hypothesis_id, config_hash)``) использовал runner, а
``storage.Database`` (cloudpickle-файлы, ключ-строка) — планировщик через
``db.load(key, storage=...)``. Результаты, записанные раннером, планировщик не
видел.

``SharedCache`` наследует репозиторий (API ``has_result/load_result/save_result``
для раннера) и дополнительно реализует протокол ``Database`` (``load/save/
load_all`` для legacy-путей планировщика) поверх ОДНОГО файла SQLite. В связке
с repository-aware путём в ``planner._base`` (``_has_cached_result`` /
``_get_cached_r2`` при наличии ``has_result`` бьют по ключу репозитория) это даёт
общий кэш: то, что вычислил и сохранил runner, планировщик видит как P_e.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from hyppo.metadata_repository.metadata_repository import MetadataRepository


class _Boxed:
    """Обёртка, повторяющая контракт storage.Pickled (атрибут .obj)."""

    def __init__(self, obj):
        self.obj = obj


class SharedCache(MetadataRepository):
    """Единый кэш planner↔runner поверх одного SQLite-файла."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        super().__init__(db_path)
        # generic KV-таблица для протокола Database (решётки, произвольные объекты)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (skey TEXT PRIMARY KEY, blob BLOB NOT NULL)"
        )
        self._conn.commit()

    # ── протокол storage.Database (используется legacy-кодом планировщика) ──
    def load(self, filename: str, storage: str = "") -> _Boxed | None:
        row = self._conn.execute(
            "SELECT blob FROM kv WHERE skey = ?", (f"{storage}/{filename}",)
        ).fetchone()
        return None if row is None else _Boxed(pickle.loads(row[0]))

    def save(self, obj, filename: str, storage: str = "", **_kw) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (skey, blob) VALUES (?, ?)",
            (f"{storage}/{filename}", pickle.dumps(obj)),
        )
        self._conn.commit()

    def load_all(self, storage: str = "") -> list:
        rows = self._conn.execute(
            "SELECT blob FROM kv WHERE skey LIKE ?", (f"{storage}/%",)
        ).fetchall()
        return [_Boxed(pickle.loads(r[0])) for r in rows]

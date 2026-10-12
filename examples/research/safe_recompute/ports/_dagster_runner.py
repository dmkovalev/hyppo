"""Шаг прогона Dagster (запускается интерпретатором .venv-cache, где живёт dagster).

python _dagster_runner.py spec.json  ->  stdout: JSON {"executed": [...], "rounds": [{"status", "selected"}, ...]}

spec: {"workdir", "assets": [{"name", "cmd", "deps", "sources", "code_version", "out"}],
       "sources": [{"name", "kind": "store"|"param", "meta"|"params_file", "entity", "token", "key"}],
       "force": [имена активов, материализуемых вручную]}
Состояние Dagster (события материализации/наблюдения, data provenance) хранится
в постоянном экземпляре DAGSTER_HOME, который задаёт вызывающая сторона.
Только stdlib + dagster: модуль не импортирует пакет комплекса.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from dagster import (AssetKey, DagsterInstance, DataVersion, MaterializeResult, asset, materialize,
                     observable_source_asset)
from dagster._core.definitions.assets.graph.asset_graph import AssetGraph
from dagster._core.definitions.data_version import CachingStaleStatusResolver, StaleStatus
from dagster._core.definitions.observe import observe
from dagster._core.asset_graph_view.asset_graph_view import AssetGraphView, TemporalContext
from dagster._time import get_current_datetime


def _store_version(meta: str, entity: str, token: str) -> str:
    if not Path(meta).exists():
        return "v1"
    con = sqlite3.connect(meta, timeout=60)
    try:
        row = con.execute("SELECT version FROM meta WHERE entity=? AND token=?", (entity, token)).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        con.close()
    return row[0] if row else "v1"


def _source(s: dict):
    if s["kind"] == "store":
        @observable_source_asset(name=s["name"])
        def _obs():
            return DataVersion(_store_version(s["meta"], s["entity"], s["token"]))
    else:
        @observable_source_asset(name=s["name"])
        def _obs():
            return DataVersion(str(json.loads(Path(s["params_file"]).read_text())[s["key"]]))
    return _obs


def _asset(a: dict, workdir: str):
    @asset(name=a["name"], deps=[*a["deps"], *a["sources"]], code_version=a["code_version"])
    def _mat() -> MaterializeResult:
        res = subprocess.run(a["cmd"], cwd=workdir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stdout + res.stderr)
        h = hashlib.sha256((Path(workdir) / a["out"]).read_bytes()).hexdigest()
        return MaterializeResult(data_version=DataVersion(h))
    return _mat


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    sources = [_source(s) for s in spec["sources"]]
    assets = [_asset(a, spec["workdir"]) for a in spec["assets"]]
    instance = DagsterInstance.get()
    names = [a["name"] for a in spec["assets"]]
    force = list(spec.get("force", []))
    executed: list[str] = []
    rounds: list[dict] = []
    # Один проход stale_assets_only не даёт неподвижной точки: потомок становится
    # STALE лишь после того, как родитель материализован с новой data version.
    # Повторяем «наблюдение -> статусы -> материализация STALE/MISSING» до пустого
    # выбора (как последовательные тики сенсора/автоматизации).
    for _ in range(len(names) + 2):
        if sources:
            observe(sources, instance=instance)
        graph = AssetGraph.from_assets([*sources, *assets])
        view = AssetGraphView(temporal_context=TemporalContext(effective_dt=get_current_datetime(),
                                                               last_event_id=None),
                              instance=instance, asset_graph=graph)
        resolver = CachingStaleStatusResolver(instance=instance, asset_graph=graph, loading_context=view)
        status = {n: resolver.get_status(AssetKey(n)).value for n in names}
        # как resolve_stale_or_missing_assets (stale_assets_only в RunRequest)
        selected = [n for n in names if status[n] in (StaleStatus.STALE.value, StaleStatus.MISSING.value)]
        selected += [n for n in force if n not in selected]
        force = []
        rounds.append({"status": status, "selected": selected})
        if not selected:
            break
        result = materialize([*sources, *assets], instance=instance,
                             selection=[AssetKey(n) for n in selected])
        executed += [e.asset_key.path[-1] for e in result.get_asset_materialization_events()]
    else:
        raise RuntimeError("нет неподвижной точки устаревания")
    print(json.dumps({"executed": executed, "rounds": rounds}))


if __name__ == "__main__":
    main()

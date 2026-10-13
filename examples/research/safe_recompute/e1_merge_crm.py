"""E1: пересчитать только граф CRM и влить его в data/e1_eviction.json.

Полный прогон E1 (27 графов) долгий; при изменении common.CRM_EDGES графы nf-core не
меняются. Скрипт запускает e1_eviction.main(graphs_limit=1) (только crm_brugge, те же
steps/seeds и политики), заменяет запись crm_brugge в каждом блоке «политика:ёмкость» и
пересчитывает сводные величины по всем 27 графам ровно так же, как e1_eviction.main:
median_ratio_cascade_to_safe, median_saving_of_safe, saving_iqr, crm_ratio.

Usage: python -m examples.research.safe_recompute.e1_merge_crm [steps] [seeds]
"""
from __future__ import annotations

import statistics
import sys
import tempfile
from pathlib import Path

from examples.research.safe_recompute import e1_eviction
from examples.research.safe_recompute.common import DATA, load_json, save_json


def summarize(per_graph: dict) -> dict:
    ratios = [v["cascade_runs_median"] / v["safe_runs_median"] for v in per_graph.values()]
    saved = [1 - v["safe_runs_median"] / v["cascade_runs_median"] for v in per_graph.values()]
    return {
        "graphs": per_graph,
        "median_ratio_cascade_to_safe": statistics.median(ratios),
        "median_saving_of_safe": statistics.median(saved),
        "saving_iqr": ([q[0], q[2]] if len(saved) >= 2 and (q := statistics.quantiles(saved, n=4))
                       else [saved[0], saved[0]]),
        "crm_ratio": per_graph["crm_brugge"]["cascade_runs_median"] / per_graph["crm_brugge"]["safe_runs_median"],
    }


def main(steps=40, seeds=10, path=None):
    path = Path(path or DATA / "e1_eviction.json")
    full = load_json(path)
    with tempfile.TemporaryDirectory() as tmp:
        crm = e1_eviction.main(steps, seeds, out=Path(tmp) / "crm.json", graphs_limit=1)
    assert set(crm) == set(full), (sorted(crm), sorted(full))
    for tag, block in full.items():
        per_graph = dict(block["graphs"])
        assert "crm_brugge" in per_graph and len(per_graph) == 27, (tag, len(per_graph))
        per_graph["crm_brugge"] = crm[tag]["graphs"]["crm_brugge"]
        full[tag] = summarize(per_graph)
        print(f"{tag}: median cascade/safe = {full[tag]['median_ratio_cascade_to_safe']:.3f}, "
              f"CRM ratio = {full[tag]['crm_ratio']:.3f}")
    save_json(full, path)
    print("written", path)
    return full


if __name__ == "__main__":
    main(*[int(x) for x in sys.argv[1:3]])

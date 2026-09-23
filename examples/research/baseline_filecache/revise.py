# examples/research/baseline_filecache/revise.py
"""CLI внешней ревизии сущности хранилища: python revise.py --meta DB --entity gains"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.research.baseline_filecache import metastore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--entity", required=True, choices=["gains", "tau", "well_status", "corey_ref"])
a = ap.parse_args()
print(json.dumps(metastore.revise(a.meta, a.entity)))

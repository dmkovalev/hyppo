# examples/research/baseline_filecache/config.py
"""Параметры эксперимента «корректно настроенный файловый кэш как базовая линия»."""
from __future__ import annotations

import math
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = Path(os.environ.get("HYPPO_DIAG_ROOT", HERE.parents[3]))  # F:/git-repos/diss
SNAKEMAKE_PY = DIAG / ".venv-cache" / "Scripts" / "python.exe"
HYPPO_PY = HERE.parents[2] / ".venv" / "Scripts" / "python.exe"

# Сетка конвейера Brugge (как в scripts/filecache_control): 2*4*4*4 = 128 OPR.
GRID = {
    "cons": ["uto", "pos"],
    "alphas": ["0.1", "1", "10", "100"],
    "corey": ["1.5", "2", "2.5", "3"],
}
SMALL_GRID = {"cons": ["uto", "pos"], "alphas": ["1"], "corey": ["2"]}

# Сущности общего метахранилища и правила, которые их читают.
ENTITIES = ["gains", "tau", "well_status", "corey_ref"]
PAIRS = [
    ("gains", "fit_liq"),
    ("tau", "fit_liq"),
    ("well_status", "fit_liq"),
    ("well_status", "fit_opr"),
    ("corey_ref", "fit_wct"),
    ("corey_ref", "fit_opr"),
]
PER_TOKEN = {"gains", "tau"}          # хранятся по токену constraints
GLOBAL_TOKEN = "*"                    # для corey_ref, well_status

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
SUBSETS_PER_LEVEL = 5
REVISIONS = ["gains", "tau", "well_status", "corey_ref", "ridge"]  # ridge — контроль


def pair_id(pair: tuple[str, str]) -> str:
    return f"{pair[0]}:{pair[1]}"


def level_size(p: float) -> int:
    return int(math.floor(p * len(PAIRS) + 0.5))


def declaration_subsets(seed: int = 20260923) -> list[tuple[float, tuple[tuple[str, str], ...]]]:
    """17 конфигураций объявлений: по одной для p=0 и p=1, по 5 различных для прочих."""
    rng = random.Random(seed)
    out: list[tuple[float, tuple[tuple[str, str], ...]]] = []
    for p in LEVELS:
        k = level_size(p)
        if k in (0, len(PAIRS)):
            out.append((p, tuple(PAIRS[:k])))
            continue
        seen: set[frozenset] = set()
        while len(seen) < SUBSETS_PER_LEVEL:
            sub = tuple(sorted(rng.sample(PAIRS, k)))
            if frozenset(sub) not in seen:
                seen.add(frozenset(sub))
                out.append((p, sub))
    return out


def tok(x: str) -> str:
    """Токен значения для имён переменных/гипотез: '0.1' -> '0p1'."""
    return x.replace(".", "p")

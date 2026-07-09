"""Serve the precomputed REAL virtual-experiment data.

The JSON is produced by scripts/gui_real_data.py under .venv311 (pywaterflood
CRM on real Brugge/Norne + COA Algorithm 1 + real owlready2 ontology). It is a
dev/demo artifact, not shipped in the package — resolved via HYPPO_REAL_DATA
env override or scripts/gui_real_data.json relative to cwd (repo checkout).
"""

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/real", tags=["real"])

_CACHE: dict | None = None


def _resolve_path() -> Path:
    env = os.environ.get("HYPPO_REAL_DATA")
    return Path(env) if env else Path("scripts/gui_real_data.json")


@router.get("")
def get_real() -> dict:
    global _CACHE
    if _CACHE is None:
        path = _resolve_path()
        if not path.exists():
            raise HTTPException(
                404, "real data not found — run scripts/gui_real_data.py"
            )
        with open(path, encoding="utf-8") as fp:
            _CACHE = json.load(fp)
    return _CACHE

"""Serve the precomputed REAL virtual-experiment data.

The JSON is produced by scripts/gui_real_data.py under .venv311 (pywaterflood
CRM on real Brugge/Norne + COA Algorithm 1 + real owlready2 ontology).
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/real", tags=["real"])

_PATH = Path(__file__).resolve().parents[1] / "real_data.json"
_CACHE: dict | None = None


@router.get("")
def get_real() -> dict:
    global _CACHE
    if _CACHE is None:
        if not _PATH.exists():
            raise HTTPException(
                404, "real_data.json not found — run scripts/gui_real_data.py"
            )
        with open(_PATH, encoding="utf-8") as fp:
            _CACHE = json.load(fp)
    return _CACHE

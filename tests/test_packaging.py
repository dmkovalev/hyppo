"""Packaging metadata sanity checks (dynamic version, py.typed presence)."""

from importlib import metadata
from pathlib import Path

import hyppo


def test_version_matches_hyppo_dunder():
    """Installed distribution version equals hyppo.__version__.

    Falls back to checking the pyproject dynamic-version wiring when the
    package is not installed (e.g. running from a bare source checkout).
    """
    try:
        dist_version = metadata.version("gedanken")
    except metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        assert 'dynamic = ["version"]' in text
        assert "hyppo.__version__" in text
        return
    assert dist_version == hyppo.__version__


def test_py_typed_marker_present():
    """hyppo ships a py.typed marker for downstream type checking."""
    marker = Path(hyppo.__file__).resolve().parent / "py.typed"
    assert marker.exists()

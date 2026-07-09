"""Packaging metadata sanity checks (dynamic version, py.typed presence)."""

import shutil
import subprocess
import zipfile
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


def test_wheel_contains_gui_static(tmp_path):
    """Built wheel must ship the SPA (hyppo/gui/static/index.html).

    Skips when the built SPA is absent from the source tree (no web build
    performed yet) rather than failing on a fresh clone without node/npm.
    """
    repo_root = Path(__file__).resolve().parents[1]
    static_index = repo_root / "hyppo" / "gui" / "static" / "index.html"
    if not static_index.exists():
        return

    uv_bin = shutil.which("uv")
    if uv_bin is None:
        return

    out_dir = tmp_path / "wheel_out"
    subprocess.run(
        [uv_bin, "build", "--wheel", "-o", str(out_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    wheels = list(out_dir.glob("*.whl"))
    assert wheels, "uv build produced no wheel"
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
    assert "hyppo/gui/static/index.html" in names

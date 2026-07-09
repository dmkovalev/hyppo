"""System-level import health checks for the hyppo package.

Guards two invariants for publication:
1. Every submodule under hyppo is importable (no broken imports slip in).
2. hyppo.actions does not pull in hyppo.mcp (no package import cycle).
"""

import importlib
import pkgutil
import subprocess
import sys

import hyppo


def test_all_submodules_importable():
    failures = []
    for module_info in pkgutil.walk_packages(hyppo.__path__, prefix="hyppo."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # noqa: BLE001 - collect all failures, not just the first
            failures.append(f"{module_info.name}: {exc!r}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def test_actions_does_not_import_mcp():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import hyppo.actions; import sys; "
            "assert 'hyppo.mcp' not in sys.modules, "
            "'hyppo.actions must not pull in hyppo.mcp (import cycle)'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

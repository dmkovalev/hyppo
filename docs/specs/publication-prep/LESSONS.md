# LESSONS — publication-prep

## Concurrent executors must not uv-sync a shared venv
- Two agents hit locked/pruned .venv (pywin32/rpds DLLs mid-write); use `uvx --with ... --with-editable .` isolated envs for tooling, reserve `uv sync` for a single owner.

## Empty __init__.py hides broken submodules
- `import hyppo.storage` passed while `storage._base` was unimportable (missing dep); recursive walk_packages import test now guards this permanently.

## Report-time line numbers can misattribute runtime warnings
- "unclosed sqlite at coa/causal.py:85" was owlready2's global world surfacing under coverage; verify the file before planning a fix at a reported location.

## mkdocstrings strict mode vs Google docstrings without type hints
- Params documented without signature annotations are strict-fatal; set `docstring_options.warn_missing_types: false` instead of editing source.

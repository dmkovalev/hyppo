"""VirtualExperimentRunner — executes virtual experiments according to plan.

Implements the runner described in Section 3.1.7 of the dissertation.
Executes models in topological order, retries on failure (k=3),
cascades SKIPPED status to dependent hypotheses.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Status(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class RunResult:
    hypothesis_id: str
    status: Status
    metrics: dict = field(default_factory=dict)
    error: str | None = None


class Runner:
    """Executes virtual experiment plan (Section 3.1.7).

    For each hypothesis in P_ne (topological order):
    - Call model function with config
    - Retry up to max_retries times on failure
    - If still fails: mark FAILED, cascade SKIPPED to dependents
    - Save results to repository
    """

    def __init__(self, repository=None, max_retries: int = 3) -> None:
        self.repository = repository
        self.max_retries = max_retries

    def execute(
        self,
        plan: dict,  # {"p_ne": set[str], "p_e": set[str]}
        models: dict[str, Callable],
        configs: dict[str, dict] | None = None,
        lattice_edges: list[tuple[str, str]] | None = None,
    ) -> dict[str, dict]:
        """Execute plan and return results.

        Args:
            plan: {"p_ne": set of hypothesis IDs to compute, "p_e": set to load from cache}
            models: {hypothesis_id: callable(config) -> {"r2": float, ...}}
            configs: {hypothesis_id: config_dict}
            lattice_edges: list of (parent, child) derived_by edges

        Returns:
            {hypothesis_id: {"status": str, "metrics": dict}}
        """
        results: dict[str, dict] = {}
        p_ne = plan.get("p_ne", set())
        p_e = plan.get("p_e", set())
        configs = configs or {}
        failed_ancestors: set[str] = set()

        # Build dependency graph for cascade
        dependents: dict[str, set[str]] = {}
        if lattice_edges:
            for parent, child in lattice_edges:
                dependents.setdefault(parent, set()).add(child)

        # Load cached results for P_e
        for h_id in p_e:
            if self.repository and self.repository.has_result(h_id, configs.get(h_id, {})):
                cached = self.repository.load_result(h_id, configs.get(h_id, {}))
                results[h_id] = {"status": Status.SUCCESS.value, "metrics": cached.get("metrics", {})}
            else:
                results[h_id] = {"status": Status.SUCCESS.value, "metrics": {}}

        # Execute P_ne in order (caller should provide topological order)
        for h_id in p_ne:
            # Check if ancestor failed
            if h_id in failed_ancestors:
                results[h_id] = {"status": Status.SKIPPED.value, "metrics": {}}
                # Cascade SKIPPED to dependents
                self._cascade_skip(h_id, dependents, failed_ancestors)
                logger.info(f"Skipped {h_id} (ancestor failed)")
                continue

            # Try to execute with retries
            config = configs.get(h_id, {})
            model_fn = models.get(h_id)
            if model_fn is None:
                results[h_id] = {"status": Status.FAILED.value, "metrics": {}, "error": "No model function"}
                self._cascade_skip(h_id, dependents, failed_ancestors)
                continue

            success = False
            last_error = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    metrics = model_fn(config)
                    results[h_id] = {"status": Status.SUCCESS.value, "metrics": metrics}
                    # Save to repository
                    if self.repository:
                        self.repository.save_result(h_id, config, metrics, Status.SUCCESS.value)
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Attempt {attempt}/{self.max_retries} failed for {h_id}: {e}")

            if not success:
                results[h_id] = {"status": Status.FAILED.value, "metrics": {}, "error": last_error}
                self._cascade_skip(h_id, dependents, failed_ancestors)
                logger.error(f"Failed {h_id} after {self.max_retries} attempts")

        return results

    def _cascade_skip(self, h_id: str, dependents: dict[str, set[str]], failed_set: set[str]) -> None:
        """Recursively mark all dependents as needing skip."""
        for dep in dependents.get(h_id, set()):
            failed_set.add(dep)
            self._cascade_skip(dep, dependents, failed_set)

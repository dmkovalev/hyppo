"""VirtualExperimentManager — orchestrates the full lifecycle (Section 3.1.2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import networkx as nx

from hyppo.metadata_repository import MetadataRepository
from hyppo.runner import Runner


class Manager:
    """Orchestrates virtual experiment lifecycle (Section 3.1.2).

    Four stages:
    1. Initialization — create experiment, save to repository
    2. Build lattice — construct hypothesis lattice from structures
    3. Planning — determine P_ne/P_e with caching
    4. Execution — run models, save results
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        r2_threshold: float = 0.7,
        max_retries: int = 3,
    ) -> None:
        self.repository = MetadataRepository(db_path=db_path)
        self.r2_threshold = r2_threshold
        self.runner = Runner(repository=self.repository, max_retries=max_retries)

    def orchestrate(
        self,
        hypotheses: list[str],
        workflow_edges: list[tuple[str, str]],
        models: dict[str, Callable],
        config: dict[str, dict] | None = None,
        structures: dict | None = None,
    ) -> dict[str, dict]:
        """Run full virtual experiment lifecycle.

        Args:
            hypotheses: list of hypothesis IDs
            workflow_edges: DAG edges (parent, child)
            models: {hypothesis_id: callable(config) -> {"r2": float, ...}}
            config: {hypothesis_id: param_dict}
            structures: hypothesis structures for lattice construction (optional)

        Returns:
            {hypothesis_id: {"status": str, "metrics": dict}}
        """
        config = config or {h: {} for h in hypotheses}

        # Stage 1: Initialize — build DAG from hypotheses and edges
        lattice = nx.DiGraph()
        lattice.add_nodes_from(hypotheses)
        lattice.add_edges_from(workflow_edges)

        # Stage 2: Build lattice (use provided edges as lattice)
        # In full implementation, HypothesisLattice.build_lattice() would
        # analyze causal structures. Here we use the workflow DAG directly.

        # Stage 3: Plan — determine P_ne (needs execution) and P_e (cached)
        p_ne: list[str] = []
        p_e: set[str] = set()

        topo_order = list(nx.topological_sort(lattice))
        for h in topo_order:
            if self.repository.has_result(h, config.get(h, {})):
                cached = self.repository.load_result(h, config.get(h, {}))
                if cached is not None:
                    r2 = cached.get("metrics", {}).get("r2")
                    if r2 is not None and r2 < self.r2_threshold:
                        # Prune low-quality hypothesis and descendants
                        continue
                p_e.add(h)
            else:
                p_ne.append(h)

        # Stage 4: Execute
        results = self.runner.execute(
            plan={"p_ne": p_ne, "p_e": p_e},
            models=models,
            configs=config,
            lattice_edges=workflow_edges,
        )

        return results

    def close(self) -> None:
        """Close the underlying repository connection."""
        self.repository.close()

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

        # Stage 1: Initialize
        # Stage 2: Build lattice (Algorithm 1). When causal structures are
        # supplied, run HypothesisGraph.build() to derive the lattice edges from
        # complete causal unions over the workflow-reachable pairs; otherwise fall
        # back to the workflow DAG itself.
        lattice_edges = list(workflow_edges)
        if structures:
            from hyppo.coa import HypothesisGraph

            idx = {h: i for i, h in enumerate(hypotheses)}
            graph = HypothesisGraph()
            for h in hypotheses:
                graph.add(structures.get(h, []))
            for u, v in workflow_edges:
                graph.connect(idx[u], idx[v])
            lattice_edges = [(hypotheses[i], hypotheses[j]) for i, j in graph.build()]

        lattice = nx.DiGraph()
        lattice.add_nodes_from(hypotheses)
        lattice.add_edges_from(lattice_edges)

        # Stage 3: Plan — determine P_ne (needs execution) and P_e (cached)
        p_ne, p_e = self._partition(lattice, config)

        # Stage 4: Execute
        results = self.runner.execute(
            plan={"p_ne": p_ne, "p_e": p_e},
            models=models,
            configs=config,
            lattice_edges=lattice_edges,
        )

        return results

    def _partition(
        self, lattice: nx.DiGraph, config: dict[str, dict]
    ) -> tuple[list[str], set[str]]:
        """Three-way cascading partition of the lattice (Section 3.1.6.2).

        Mirrors ``planner.build_optimal_plan`` semantics exactly, over the
        ``derived_by`` lattice DAG:

        - cache miss => the hypothesis and ALL its descendants need recompute
          (into P_ne, the cascade effect);
        - cached but R2 < threshold => the hypothesis and ALL its descendants
          are pruned from BOTH P_ne and P_e (low-quality branch cut);
        - cached and R2 >= threshold (or R2 absent) => reused (into P_e).

        A single topological pass suffices: a node is settled by the first
        missing/pruned ancestor (``nx.descendants`` is transitive), so it is
        only cache-checked on its own when every ancestor was cached-and-good.

        Returns ``(p_ne, p_e)`` where ``p_ne`` is a topologically ordered list
        (the runner cascades SKIPPED in order) and ``p_e`` is a set.
        """
        topo_order = list(nx.topological_sort(lattice))
        pne: set[str] = set()
        pe: set[str] = set()
        pruned: set[str] = set()
        for h in topo_order:
            if h in pne or h in pruned:
                continue
            cfg_h = config.get(h, {})
            if not self.repository.has_result(h, cfg_h):
                pne.add(h)
                pne.update(nx.descendants(lattice, h))
            else:
                cached = self.repository.load_result(h, cfg_h)
                r2 = cached.get("metrics", {}).get("r2") if cached is not None else None
                if r2 is not None and r2 < self.r2_threshold:
                    pruned.add(h)
                    pruned.update(nx.descendants(lattice, h))
                else:
                    pe.add(h)
        p_ne = [h for h in topo_order if h in pne]
        return p_ne, pe

    def close(self) -> None:
        """Close the underlying repository connection."""
        self.repository.close()

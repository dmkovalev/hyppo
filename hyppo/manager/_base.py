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
    2. Build graph — construct hypothesis graph from structures
    3. Planning — determine P_ne/P_e with caching
    4. Execution — run models, save results
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        r2_threshold: float = 0.7,
        max_retries: int = 3,
        max_workers: int = 1,
        plan_mode: str = "cascade",
    ) -> None:
        """``max_workers`` > 1 runs each topological level of P_ne in parallel
        (see :class:`hyppo.runner.Runner`); 1 keeps sequential execution.

        ``plan_mode``: ``"cascade"`` (default) recomputes Cl(N), every hypothesis
        without an actual result together with all its descendants (Algorithm 4,
        theorem thm:1); ``"safe"`` recomputes exactly N (theorem thm:safe). Both
        are safe; they coincide when the repository is closed under sources, and
        ``"safe"`` is strictly smaller after records of sources were evicted.
        ``"safe"`` relies on the chained keys: a changed model or parameter of a
        source already makes every descendant miss the cache."""
        if plan_mode not in ("cascade", "safe"):
            raise ValueError(f"plan_mode must be 'cascade' or 'safe', got {plan_mode!r}")
        self.plan_mode = plan_mode
        self.repository = MetadataRepository(db_path=db_path)
        # hypotheses made stale by REFUTED/SUPERSEDED ancestors in an earlier run;
        # both sets are persisted in the repository, so a new Manager on the same
        # database keeps the marks that were not worked off yet
        self.pending_stale: set[str] = self.repository.load_marks("pending_stale")
        # hypotheses REFUTED/SUPERSEDED as of the last run (cascade fires on change)
        self.invalid_seen: set[str] = self.repository.load_marks("invalid_seen")
        # revision epochs eps_g (persisted): incremented on every invalidation
        # event of g and chained into the keys of all consumers of g, so records
        # made before a revision of a source are never read again, whatever the
        # parameters (the marks above are only a fast path, see cache_keys)
        self.epochs: dict[str, int] = self.repository.load_epochs()
        self.r2_threshold = r2_threshold
        self.runner = Runner(
            repository=self.repository, max_retries=max_retries, max_workers=max_workers
        )

    def orchestrate(
        self,
        hypotheses: list[str],
        workflow_edges: list[tuple[str, str]],
        models: dict[str, Callable],
        config: dict[str, dict] | None = None,
        structures: dict | None = None,
        model_ids: dict[str, str] | None = None,
        stale: set[str] | None = None,
        competes: dict[str, set[str]] | None = None,
        individuals: dict | None = None,
        ontology=None,
        ranking_weights: dict[str, float] | None = None,
    ) -> dict[str, dict]:
        """Run full virtual experiment lifecycle.

        Args:
            hypotheses: list of hypothesis IDs
            workflow_edges: DAG edges (parent, child)
            models: {hypothesis_id: callable(config) -> {"r2": float, ...}}
            config: {hypothesis_id: param_dict}
            structures: hypothesis structures for graph construction (optional)
                (a list of equations, or {"equations": [...], "inputs": {...}}
                with the declared inputs In(h) of the model, condition У2)
            model_ids: model identifiers for the chained cache keys (def:12)
            stale: hypotheses marked StaleHypothesis by the ontology layer
            competes: {h: competitors} -- for the SUPERSEDED status (Delta AIC)
            ranking_weights: comparable-group criterion weights forwarded to Runner.
            individuals, ontology: optional OWL individuals {h: individual} and
                their ontology; if given, REFUTED/SUPERSEDED statuses are written
                as InvalidHypothesis and the cascade is derived by RL
                materialisation, else it follows the graph descendants

        Stale hypotheses found after this run (descendants of hypotheses that
        became REFUTED or SUPERSEDED) are kept in :attr:`pending_stale` and are
        recomputed by the next ``orchestrate`` call (definition def:12 (4)).

        Returns:
            {hypothesis_id: {"status": str, "metrics": dict}}
        """
        config = config or {h: {} for h in hypotheses}

        # Stage 1: Initialize
        # Stage 2: Build graph (Algorithm 1). When causal structures are
        # supplied, run HypothesisGraph.build() to derive the graph edges from
        # complete causal unions over the workflow-reachable pairs; otherwise fall
        # back to the workflow DAG itself.
        graph_edges = list(workflow_edges)
        if structures:
            from hyppo.coa import HypothesisGraph

            idx = {h: i for i, h in enumerate(hypotheses)}
            graph = HypothesisGraph()
            for h in hypotheses:
                spec = structures.get(h, [])
                if isinstance(spec, dict):  # {"equations": [...], "inputs": {...}} (condition У2)
                    graph.add(spec.get("equations", []), spec.get("inputs"))
                else:
                    graph.add(spec)
            for u, v in workflow_edges:
                graph.connect(idx[u], idx[v])
            graph_edges = [(hypotheses[i], hypotheses[j]) for i, j in graph.build()]

        hyp_graph = nx.DiGraph()
        hyp_graph.add_nodes_from(hypotheses)
        hyp_graph.add_edges_from(graph_edges)

        # Stage 3: Plan — determine P_ne (needs execution) and P_e (cached).
        # Actuality (def:12): chained cache keys cover models/parameters of the
        # hypothesis and all its sources; `stale` (StaleHypothesis markers of
        # the ontology layer) covers revisions that touch neither.
        keys = self.current_cache_keys(hyp_graph, models, config, model_ids)
        stale_now = set(stale or set()) | set(self.pending_stale)
        p_ne, p_e = self._partition(hyp_graph, keys, stale_now)

        # Stage 4: Execute
        results = self.runner.execute(
            plan={"p_ne": p_ne, "p_e": p_e},
            models=models,
            configs=config,
            graph_edges=graph_edges,
            cache_configs=keys,
            competes=competes,
            ranking_weights=ranking_weights,
        )

        # recomputed hypotheses are fresh again
        self.pending_stale -= {
            h for h, r in results.items() if h in p_ne and r.get("status") == "SUCCESS"
        }
        # Stage 4 -> cascade: a REFUTED / SUPERSEDED hypothesis invalidates its
        # descendants (hidden staleness: files, models and configs unchanged)
        invalid = {
            h for h, r in results.items()
            if r.get("epistemic_status") in ("REFUTED", "SUPERSEDED")
        }
        # the cascade fires on the invalidation EVENT (status change), not on
        # every run in which the status persists
        newly_invalid = invalid - self.invalid_seen
        self.invalid_seen = invalid
        for h in newly_invalid:  # revision event of h: new epoch for its consumers
            self.epochs[h] = self.epochs.get(h, 0) + 1
        self.repository.save_epochs(self.epochs)
        self.pending_stale |= self._cascade(hyp_graph, newly_invalid, individuals, ontology)
        self.repository.save_marks("pending_stale", self.pending_stale)
        self.repository.save_marks("invalid_seen", self.invalid_seen)
        return results

    def _cascade(self, hyp_graph, invalid, individuals=None, ontology=None) -> set[str]:
        """Descendants of the invalidated hypotheses. With OWL individuals the
        status is bridged to InvalidHypothesis and the cascade is DERIVED by RL
        materialisation (rule 4); otherwise graph reachability is used (the same
        set, by theorem thm:integrity (IV))."""
        if not invalid:
            return set()
        if individuals is not None and ontology is not None:
            from hyppo.ontology.core_rules import (
                StaleHypothesis,
                apply_epistemic_status,
                materialize_cascade,
            )

            for h in invalid:
                if h in individuals:
                    apply_epistemic_status(individuals[h], "SUPERSEDED")
            materialize_cascade(ontology)
            reach = set().union(*(nx.descendants(hyp_graph, h) for h in invalid if h in hyp_graph))
            return {
                h for h, ind in individuals.items()
                if StaleHypothesis in ind.is_a and h in reach
            }
        out: set[str] = set()
        for h in invalid:
            out |= nx.descendants(hyp_graph, h)
        return out  # a descendant of another invalidated hypothesis is stale too

    @staticmethod
    def cache_keys(
        hyp_graph: nx.DiGraph,
        models: dict[str, Callable] | None = None,
        config: dict[str, dict] | None = None,
        model_ids: dict[str, str] | None = None,
        epochs: dict[str, int] | None = None,
    ) -> dict[str, dict]:
        """Chained cache keys (definition def:12, conditions (1)-(3)) for every
        hypothesis of ``graph``: own config, model identifier (explicit
        ``model_ids`` or the model callable's qualified name) and the keys of the
        direct predecessors. The runner saves and the planner looks up results by
        these keys, so a change of a source's model or parameters misses the
        cache for the source and all its descendants.

        ``epochs``: revision epochs of the hypotheses (see :attr:`epochs`); the
        key of h contains ``(g, K(g), eps_g)`` for every source g. Without
        ``epochs`` (static use) all epochs are 0; to address the records written
        by a Manager after revision events use :meth:`current_cache_keys`."""
        from hyppo.planner.actuality import chained_keys

        if model_ids is None:
            model_ids = {
                h: f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}"
                for h, fn in (models or {}).items()
            }
        topo = list(nx.topological_sort(hyp_graph))
        return chained_keys(topo, hyp_graph.edges(), config or {}, model_ids, epochs)

    def current_cache_keys(
        self,
        hyp_graph: nx.DiGraph,
        models: dict[str, Callable] | None = None,
        config: dict[str, dict] | None = None,
        model_ids: dict[str, str] | None = None,
    ) -> dict[str, dict]:
        """:meth:`cache_keys` with the current revision epochs of this Manager:
        the keys under which ``orchestrate`` reads and writes records now."""
        return self.cache_keys(hyp_graph, models, config, model_ids, self.epochs)

    def _partition(
        self, hyp_graph: nx.DiGraph, config: dict[str, dict], stale: set[str] = frozenset()
    ) -> tuple[list[str], set[str]]:
        """Three-way cascading partition of the graph (Section 3.1.6.2).

        Mirrors ``planner.build_optimal_plan`` semantics exactly, over the
        ``derived_by`` graph DAG:

        - cache miss => the hypothesis and ALL its descendants need recompute
          (plan_mode "cascade"; in plan_mode "safe" only the hypothesis itself,
          its descendants are settled by their own chained keys)
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
        topo_order = list(nx.topological_sort(hyp_graph))
        pne: set[str] = set()
        pe: set[str] = set()
        pruned: set[str] = set()
        # One topological pass, O(|V| + |E|) (proposition prop:plan_lower): a
        # hypothesis inherits the cascade from a recomputed parent ("cascade"
        # mode) and the cut from a pruned parent, else it is checked on its own.
        cascade = self.plan_mode == "cascade"
        for h in topo_order:
            parents = list(hyp_graph.predecessors(h))
            forced_ne = cascade and any(g in pne for g in parents)
            forced_cut = any(g in pruned for g in parents)
            if forced_ne or forced_cut:
                if forced_ne:
                    pne.add(h)
                if forced_cut:
                    pruned.add(h)
                continue
            cfg_h = config.get(h, {})
            if h in stale or not self.repository.has_result(h, cfg_h):
                pne.add(h)
            else:
                cached = self.repository.load_result(h, cfg_h)
                r2 = cached.get("metrics", {}).get("r2") if cached is not None else None
                if r2 is not None and r2 < self.r2_threshold:
                    pruned.add(h)
                else:
                    pe.add(h)
        p_ne = [h for h in topo_order if h in pne]
        return p_ne, pe

    def close(self) -> None:
        """Close the underlying repository connection."""
        self.repository.close()

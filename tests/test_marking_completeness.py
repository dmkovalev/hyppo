"""Completeness of the revision marking for the implementation (prop:marking).

Theorem thm:safe assumes that every revision of a status is marked (the
descendants of a revised hypothesis u carry StaleHypothesis). For the class of
revisions the implementation sees (an epistemic status assigned by the Runner
from R^2 and Delta AIC turns into REFUTED / SUPERSEDED in an ``orchestrate``
run) this is checked here as a property on random DAGs and random run
sequences:

    every strict descendant d of u (graph L) is owed a recomputation from the
    run in which u's status became REFUTED/SUPERSEDED until d is recomputed
    successfully in a later run; while owed, d is never in P_e, and without
    the R^2 cut d is in P_ne of every later run.

The oracle uses only observable results (statuses, plans), not Manager state.
Boundaries (documented, strict xfail): simultaneous nested events (bug: the
descendant that is itself newly invalid is dropped), loss of the in-memory
marks with a new Manager on the same repository.
"""

from __future__ import annotations

import itertools
import random

import networkx as nx
import pytest

from hyppo.manager import Manager

INVALID = ("REFUTED", "SUPERSEDED")


class _PatchedManager(Manager):
    """Minimal fix sketch: strict descendants of the newly invalid set are
    marked even if they are newly invalid themselves (no ``- invalid``)."""

    def _cascade(self, hyp_graph, invalid, individuals=None, ontology=None):
        out: set[str] = set()
        for h in invalid:
            if h in hyp_graph:
                out |= nx.descendants(hyp_graph, h)
        return out


class _World:
    """Random DAG with competitors; deterministic models f(h, version)."""

    def __init__(self, rng: random.Random, n: int):
        self.rng = rng
        self.nodes = [f"h{i}" for i in range(n)]
        self.edges = [
            (self.nodes[i], self.nodes[j])
            for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4
        ]
        # rivals: extra roots r_i competing with a DAG node, plus random DAG pairs
        self.rivals = [f"r{i}" for i in range(rng.randint(1, 3))]
        self.competes: dict[str, set[str]] = {}

        def comp(a, b):
            self.competes.setdefault(a, set()).add(b)
            self.competes.setdefault(b, set()).add(a)

        for r in self.rivals:
            comp(r, rng.choice(self.nodes))
        for _ in range(rng.randint(0, 2)):
            a, b = rng.sample(self.nodes, 2)
            comp(a, b)
        self.all = self.nodes + self.rivals
        self.version = {h: 0 for h in self.all}
        self.table: dict[tuple[str, int], dict | None] = {}
        self.g = nx.DiGraph()
        self.g.add_nodes_from(self.all)
        self.g.add_edges_from(self.edges)

    def _metrics(self, h, v):
        key = (h, v)
        if key not in self.table:
            rng = self.rng
            self.table[key] = None if rng.random() < 0.07 else {
                "r2": rng.choice([0.9, 0.9, 0.5]),
                "aic": rng.choice([0.0, 0.0, 20.0, 40.0]),
            }
        return self.table[key]

    def step(self):
        for h in self.all:  # rivals change more often: hidden revisions
            if self.rng.random() < (0.5 if h in self.rivals else 0.2):
                self.version[h] += 1

    def models(self, calls):
        def mk(h, m):
            def f(cfg):
                calls.add(h)
                if m is None:
                    raise RuntimeError("model failure")
                return dict(m)
            return f

        return {h: mk(h, self._metrics(h, self.version[h])) for h in self.all}

    def model_ids(self):
        return {h: f"{h}@{self.version[h]}" for h in self.all}


def _spy_plan(mgr):
    box = {}
    orig = mgr._partition

    def spy(*a, **kw):
        box["plan"] = orig(*a, **kw)
        return box["plan"]

    mgr._partition = spy
    return box


def _run_sequence(mgr_cls, seed, plan_mode, max_workers, r2_threshold, runs=7,
                  exclude_nested=False):
    """One random world, a sequence of runs; returns counters. Asserts the property."""
    rng = random.Random(seed)
    w = _World(rng, rng.randint(3, 7))
    mgr = mgr_cls(r2_threshold=r2_threshold, max_workers=max_workers,
                  plan_mode=plan_mode, max_retries=1)
    box = _spy_plan(mgr)
    owed: set[str] = set()
    inv_prev: set[str] = set()
    st = dict(events=0, hidden=0, owed_checked=0)
    for run in range(runs):
        if run:
            w.step()
        calls: set[str] = set()
        res = mgr.orchestrate(
            hypotheses=w.all, workflow_edges=w.edges, models=w.models(calls),
            model_ids=w.model_ids(), competes=w.competes,
        )
        p_ne, p_e = box["plan"]
        p_ne = set(p_ne)
        # the property: an owed descendant is never reused from the repository
        assert not owed & p_e, (seed, run, owed & p_e)
        if r2_threshold == 0.0:  # no R^2 cut: owed descendants are planned
            assert owed <= p_ne, (seed, run, owed - p_ne)
        st["owed_checked"] += len(owed)
        ok = {h for h in p_ne if res.get(h, {}).get("status") == "SUCCESS"}
        assert ok <= calls  # planned and successful means actually executed
        owed -= ok
        inv_now = {h for h, r in res.items() if r.get("epistemic_status") in INVALID}
        events = inv_now - inv_prev
        inv_prev = inv_now
        for u in events:
            d = nx.descendants(w.g, u)
            if exclude_nested:
                d -= events
            if d:
                st["events"] += 1
                st["hidden"] += u in p_e  # status changed while u itself reused
            owed |= d
    mgr.close()
    return st


CASES = [
    (mode, workers, thr)
    for mode in ("cascade", "safe")
    for workers in (1, 2)
    for thr in (0.0, 0.7)
]


@pytest.mark.parametrize("plan_mode,max_workers,r2_threshold", CASES)
def test_marking_complete_current_manager(plan_mode, max_workers, r2_threshold):
    """Current implementation: property holds for every event whose descendant
    is not itself newly invalid in the same run (see the nested boundary)."""
    tot = dict(events=0, hidden=0, owed_checked=0)
    for seed in range(60):
        st = _run_sequence(Manager, seed, plan_mode, max_workers, r2_threshold)
        for k in tot:
            tot[k] += st[k]
    assert tot["events"] > 30 and tot["hidden"] > 5 and tot["owed_checked"] > 30, tot


@pytest.mark.parametrize("plan_mode,max_workers,r2_threshold", CASES)
def test_marking_complete_with_fix(plan_mode, max_workers, r2_threshold):
    """With the minimal fix of _cascade the property holds for ALL events."""
    tot = dict(events=0, hidden=0, owed_checked=0)
    for seed in range(60):
        st = _run_sequence(_PatchedManager, seed, plan_mode, max_workers, r2_threshold)
        for k in tot:
            tot[k] += st[k]
    assert tot["events"] > 30 and tot["hidden"] > 5, tot


# ---------------------------------------------------------------------------
# deterministic witnesses of the boundaries
# ---------------------------------------------------------------------------

def _two_rivals_models(aic_b2, aic_a2, calls):
    def mk(h, m):
        def f(cfg):
            calls.append(h)
            return dict(m)
        return f
    return {
        "b": mk("b", {"r2": 0.9, "aic": 10.0}),
        "b2": mk("b2", {"r2": 0.9, "aic": aic_b2}),
        "a": mk("a", {"r2": 0.9, "aic": 10.0}),
        "a2": mk("a2", {"r2": 0.9, "aic": aic_a2}),
        "h": mk("h", {"r2": 0.9}),
    }


def _nested(mgr, aic_b2, aic_a2, version):
    calls: list[str] = []
    res = mgr.orchestrate(
        hypotheses=["b", "b2", "a", "a2", "h"],
        workflow_edges=[("b", "a"), ("a", "h")],
        models=_two_rivals_models(aic_b2, aic_a2, calls),
        model_ids={"b": "b@1", "b2": f"b2@{version}", "a": "a@1",
                   "a2": f"a2@{version}", "h": "h@1"},
        competes={"b": {"b2"}, "b2": {"b"}, "a": {"a2"}, "a2": {"a"}},
    )
    return calls, res


@pytest.mark.parametrize("cls,expect_bug", [(Manager, False), (_PatchedManager, False)])
def test_nested_simultaneous_events(tmp_path, cls, expect_bug):
    """b -> a -> h; b and a are superseded in the SAME run by rivals b2, a2.
    a is a strict descendant of the revised b, built on b's old result, but
    _cascade subtracts the newly invalid set: a is not marked and is reused."""
    mgr = cls(db_path=tmp_path / "n.db")
    _nested(mgr, 50.0, 50.0, 1)                       # b, a SUPPORTED
    _, res = _nested(mgr, -5.0, -5.0, 2)              # rivals revised: both SUPERSEDED
    assert res["b"]["epistemic_status"] == res["a"]["epistemic_status"] == "SUPERSEDED"
    calls, _ = _nested(mgr, -5.0, -5.0, 2)
    if expect_bug:
        assert sorted(calls) == ["h"]                 # a reused: marking incomplete
    else:
        assert sorted(calls) == ["a", "h"]
    mgr.close()


def test_marks_survive_new_manager(tmp_path):
    from tests.test_hidden_stale_witness import _run

    db = tmp_path / "p.db"
    mgr = Manager(db_path=db)
    _run(mgr, aic_b2=50.0, b2_version=1)
    _run(mgr, aic_b2=-5.0, b2_version=2)              # b SUPERSEDED, {a, h} marked
    assert mgr.pending_stale == {"a", "h"}
    mgr.close()
    mgr2 = Manager(db_path=db)                        # new session
    calls, _ = _run(mgr2, aic_b2=-5.0, b2_version=2)
    mgr2.close()
    assert sorted(calls) == ["a", "h"]                # marks persisted in the repository


def test_rehabilitation_is_not_an_event(tmp_path):
    """Boundary of the class: SUPERSEDED -> SUPPORTED (rival revised worse) is a
    status change, but not an invalidation event; no new marks are produced."""
    from tests.test_hidden_stale_witness import _run

    mgr = Manager(db_path=tmp_path / "r.db")
    _run(mgr, aic_b2=50.0, b2_version=1)
    _run(mgr, aic_b2=-5.0, b2_version=2)              # b SUPERSEDED
    _run(mgr, aic_b2=-5.0, b2_version=2)              # a, h recomputed
    calls, res = _run(mgr, aic_b2=50.0, b2_version=3)  # b rehabilitated
    assert res["b"]["epistemic_status"] == "SUPPORTED" and calls == ["b2"]
    assert mgr.pending_stale == set()
    mgr.close()


_k = itertools.count()


def test_ontology_path_needs_derived_by_matching_graph(tmp_path):
    """Ontology path of _cascade returns StaleHypothesis ∩ graph descendants:
    with derived_by missing the edge a -> h, h is not marked (condition of the
    bridge: derived_by of the individuals covers the hypothesis graph)."""
    from hyppo.core._base import Hypothesis, virtual_experiment_onto as onto
    import hyppo.ontology.core_rules as cr  # noqa: F401
    from tests.test_hidden_stale_witness import _run

    k = next(_k)
    with onto:
        ind = {n: Hypothesis(f"mc_{n}_{k}") for n in ("b", "b2", "a", "h")}
        ind["a"].derived_by = [ind["b"]]               # a -> h edge NOT reified
    mgr = Manager(db_path=tmp_path / "o.db")
    _run(mgr, aic_b2=50.0, b2_version=1, individuals=ind, ontology=onto)
    _run(mgr, aic_b2=-5.0, b2_version=2, individuals=ind, ontology=onto)
    assert mgr.pending_stale == {"a"}                  # h lost; graph path gives {a, h}
    mgr.close()

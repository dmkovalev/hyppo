"""Golden claim tests: pin statements made in the papers to the real implementations.

Sources of the claims:
  [SVD]  thesis/papers/sht_dostupnost_v1.tex — журнал «Системы высокой доступности»
  [IIP]  thesis/papers/iip2026_planning_v4.tex — DAMDID (планирование)

Contract: if a test here fails, either the code or the paper text is wrong.
Fix whichever is wrong; only then may the expected values change.

Complexity is pinned by *operation counts* (deterministic), not wall time.
"""
from __future__ import annotations

import itertools
import random

import networkx as nx
import pytest

from hyppo.coa import causal
from hyppo.coa._base import Equation, Structure
from hyppo.coa.graph import HypothesisGraph
from hyppo.lattice_constructor._base import HypothesisLattice


# ---------------------------------------------------------------------------
# Fixtures: the Norne HybridCRM hypothesis set (formulas as in [SVD] §3.9)
# ---------------------------------------------------------------------------

NORNE_HYPS = [
    ("H1", "I_agg = w_ij * I_j"),
    ("H2", "q_f = a_f*q_f_prev + b_f*I_agg"),
    ("H3", "q_s = a_s*q_s_prev + b_s*I_agg"),
    ("H4", "q_c = w_f*q_f + (1-w_f)*q_s"),
    ("H5", "q_liq_phys = J*q_c + q_prim"),
    ("H6", "q_prim = q_prev*exp(-dt*taup)"),
    ("H7", "l_ml = MLP(x_hist)"),
    ("H8", "l = g*q_liq_phys + (1-g)*l_ml"),
    ("H11", "Sw = Sw_prev + (Winj - Wlout)*dt/Vp"),
    ("H12", "krw = ((Sw-Swc)/(1-Swc-Sor))**nw"),
    ("H12b", "kro = ((1-Sw-Sor)/(1-Swc-Sor))**no"),
    ("H13", "fw = 1/(1 + kro*muw/(krw*muo))"),
    ("H14", "o_p = 1 - fw"),
    ("H15", "o = gw*o_p + (1-gw)*o_m"),
    ("GRP", "J = J0 + dJ_grp"),
    ("H19", "q_oil = l * o"),
]

# The 17 derived_by edges of Figure 3 in [SVD] (hand-derivable from the LHS/RHS
# variable flow of the equations above, independently of the implementation).
NORNE_GOLDEN_EDGES = {
    ("H1", "H2"), ("H1", "H3"), ("H2", "H4"), ("H3", "H4"), ("H4", "H5"),
    ("H6", "H5"), ("GRP", "H5"), ("H5", "H8"), ("H7", "H8"), ("H8", "H19"),
    ("H11", "H12"), ("H11", "H12b"), ("H12", "H13"), ("H12b", "H13"),
    ("H13", "H14"), ("H14", "H15"), ("H15", "H19"),
}


class _Hyp:
    def __init__(self, name: str, formula: str):
        self.name = name
        self.structure = Structure([Equation(formula=formula)])

    def __repr__(self):  # pragma: no cover
        return self.name


class _Workflow:
    def __init__(self, tasks):
        self._tasks = tasks

    def get_tasks(self):
        return self._tasks


def _build_lattice(hyps: list[_Hyp]) -> nx.DiGraph:
    wf = _Workflow([[h] for h in hyps])
    return HypothesisLattice(list(hyps), wf).lattice


def _named_edges(g: nx.DiGraph) -> set[tuple[str, str]]:
    return {(u.name, v.name) for u, v in g.edges()}


# ---------------------------------------------------------------------------
# Algorithm 1 — build_lattice  [SVD §3.9, рис. 3]
# ---------------------------------------------------------------------------

def test_alg1_norne_graph_matches_figure():
    """[SVD] «16 гипотез и 17 рёбер зависимостей (глубина графа 5)»,
    «рёбра выведены автоматически из потока переменных уравнений»."""
    hyps = [_Hyp(n, f) for n, f in NORNE_HYPS]
    g = _build_lattice(hyps)
    assert _named_edges(g) == NORNE_GOLDEN_EDGES
    assert g.number_of_nodes() == 16
    assert g.number_of_edges() == 17
    assert nx.is_directed_acyclic_graph(g)
    assert nx.dag_longest_path_length(g) == 5


# ---------------------------------------------------------------------------
# Algorithm 2 — incremental add equals full rebuild  [IIP, лемма 2]
# ---------------------------------------------------------------------------

def _random_formula_hyps(rng: random.Random, n: int) -> list[_Hyp]:
    """Hypothesis i computes x_i from a random subset of earlier outputs."""
    hyps = []
    for i in range(n):
        deps = [f"x{j}" for j in range(i) if rng.random() < 0.4]
        rhs = " + ".join(deps) if deps else f"c{i}"
        hyps.append(_Hyp(f"h{i}", f"x{i} = {rhs}"))
    return hyps


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_alg2_incremental_equals_full_rebuild(seed):
    """Adding the last hypothesis incrementally must yield the same edge set
    as rebuilding the lattice from scratch."""
    rng = random.Random(seed)
    hyps = _random_formula_hyps(rng, 12)

    full = _named_edges(_build_lattice(hyps))

    wf = _Workflow([[h] for h in hyps])
    lat = HypothesisLattice(list(hyps[:-1]), wf)
    lat.add_hypothesis(hyps[-1])
    incremental = _named_edges(lat.lattice)

    assert incremental == full


# ---------------------------------------------------------------------------
# Algorithm 4 — plan == cascade closure oracle  [IIP §4; SVD положение о каскаде]
# ---------------------------------------------------------------------------

def _random_dag(rng: random.Random, n: int, p: float) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def _oracle_plan(n: int, edges, cached) -> set[int]:
    """Independent oracle: P_ne = non-cached nodes plus everything reachable
    from them along derived_by (the cascade closure)."""
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    pne: set[int] = set()
    for h in range(n):
        if h not in cached:
            pne.add(h)
            pne |= nx.descendants(g, h)
    return pne


@pytest.mark.parametrize("seed", range(10))
def test_alg4_plan_matches_reachability_oracle(seed):
    rng = random.Random(seed)
    n = rng.randint(5, 30)
    edges = _random_dag(rng, n, 0.25)
    g = HypothesisGraph.from_edges(n, edges)
    cached = {i for i in range(n) if rng.random() < 0.5}
    assert g.plan(cached) == _oracle_plan(n, edges, cached)


# ---------------------------------------------------------------------------
# Theorem 1 — correctness and minimality of the plan  [IIP теорема 1]
# ---------------------------------------------------------------------------

def _is_valid_plan(p: set[int], n: int, edges, cached) -> bool:
    """A recompute set is correct iff it contains every non-cached hypothesis
    and is closed under derived_by successors (cascade property A2)."""
    succ = {i: set() for i in range(n)}
    for u, v in edges:
        succ[u].add(v)
    if not set(range(n)).difference(cached) <= p:
        return False
    return all(succ[x] <= p for x in p)


@pytest.mark.parametrize("seed", range(20))
def test_theorem1_plan_is_minimal_correct_plan(seed):
    """Theorem 1: the produced plan is (a) correct, (b) contained in every
    correct plan — hence minimal both in cardinality and in total cost for any
    non-negative cost function."""
    rng = random.Random(seed)
    n = rng.randint(3, 6)
    edges = _random_dag(rng, n, 0.4)
    cached = {i for i in range(n) if rng.random() < 0.5}

    plan = HypothesisGraph.from_edges(n, edges).plan(cached)

    assert _is_valid_plan(plan, n, edges, cached)

    # exhaustive check on the small instance: plan ⊆ every valid plan
    universe = list(range(n))
    for r in range(n + 1):
        for subset in itertools.combinations(universe, r):
            p = set(subset)
            if _is_valid_plan(p, n, edges, cached):
                assert plan <= p, (
                    f"valid plan {p} does not contain produced plan {plan}"
                )
    # ⊆-least ⇒ minimal by |P| and by sum of any non-negative costs.


# ---------------------------------------------------------------------------
# Complexity pins (operation counts, deterministic)
# ---------------------------------------------------------------------------

def _chain_graph(n: int) -> HypothesisGraph:
    g = HypothesisGraph()
    for i in range(n):
        g.add([{f"x{i}", f"x{i-1}"}] if i else [{f"x{i}"}])
    for i in range(n - 1):
        g.connect(i, i + 1)
    return g


def test_alg1_complexity_quadratic_pair_enumeration(monkeypatch):
    """[IIP лемма 1] Algorithm 1 costs O(|H|^2 · s · v): the number of causal
    completeness checks equals the number of reachable pairs (n(n-1)/2 on a
    chain), i.e. grows quadratically."""
    counts = {}
    orig = causal.is_complete

    for n in (10, 20, 40):
        calls = 0

        def counting(eqs):
            nonlocal calls
            calls += 1
            return orig(eqs)

        monkeypatch.setattr(causal, "is_complete", counting)
        _chain_graph(n).build()
        monkeypatch.setattr(causal, "is_complete", orig)
        counts[n] = calls
        assert calls == n * (n - 1) // 2

    assert counts[20] / counts[10] == pytest.approx(4, rel=0.15)
    assert counts[40] / counts[20] == pytest.approx(4, rel=0.15)


def test_alg2_complexity_linear_in_hypotheses(monkeypatch):
    """[IIP лемма 2] Algorithm 2 performs exactly |H| causal unions —
    linear, against |H|^2 for a full rebuild."""
    orig = causal.is_complete
    for n in (10, 20, 40):
        g = _chain_graph(n)
        calls = 0

        def counting(eqs):
            nonlocal calls
            calls += 1
            return orig(eqs)

        monkeypatch.setattr(causal, "is_complete", counting)
        g.add_hypothesis([{f"x{n}", f"x{n-1}"}])
        monkeypatch.setattr(causal, "is_complete", orig)
        assert calls == n


class _CountingSet(set):
    counter = 0

    def __iter__(self):
        type(self).counter += len(self)
        return super().__iter__()


def test_alg4_complexity_linear_in_nodes_plus_edges():
    """[IIP] Algorithm 4 runs in O(|V|+|E|): adjacency traversals grow
    linearly with the chain length (ratio ~2 when n doubles, not ~4)."""
    ops = {}
    for n in (200, 400, 800):
        g = HypothesisGraph.from_edges(n, [(i, i + 1) for i in range(n - 1)])
        g._adj = {k: _CountingSet(g._adj.get(k, ())) for k in range(n)}
        _CountingSet.counter = 0
        g.plan(cached=set(range(0, n, 2)))
        ops[n] = _CountingSet.counter
    assert ops[400] / ops[200] == pytest.approx(2, rel=0.25)
    assert ops[800] / ops[400] == pytest.approx(2, rel=0.25)


# ---------------------------------------------------------------------------
# Rule 5 / Algorithm 3 — procedural acyclicity  [SVD §3.2, правило 5]
# ---------------------------------------------------------------------------

def test_rule5_acyclicity_detected_procedurally():
    """[SVD] «ацикличность проверяется процедурно (топологическая сортировка)»."""
    from hyppo.ontology.consistency import _find_cycle_via_kahn

    acyclic = {0: {1}, 1: {2}, 2: set()}
    cyclic = {0: {1}, 1: {2}, 2: {0}}
    assert _find_cycle_via_kahn(acyclic) is None
    assert _find_cycle_via_kahn(cyclic) is not None


# ---------------------------------------------------------------------------
# Algorithm 3 — two-stage consistency check  [IIP alg:consistency; положение 1]
# ---------------------------------------------------------------------------
# Claim [положение 1]: «корректная определённость проверяется статически,
# до запуска моделей, двухэтапным (семантическим и структурным) алгоритмом»;
# stage B checks C3 (ацикличность), C4 (согласованность артефактов),
# C5 (конечность доменов конфигурации).

from hyppo.ontology.consistency import Status, check_consistency


class _FiniteQ:
    finite = True

    def __init__(self, values):
        self.values = values


_GOOD_LATTICE = {0: {1}, 1: {2}, 2: set()}
_GOOD_ARTEFACTS = {
    0: {"in": {"raw"}, "out": {"a"}},
    1: {"in": {"a"}, "out": {"b"}},
    2: {"in": {"b"}, "out": {"c"}},
}


def test_alg3_stage_b_accepts_correct_experiment():
    res = check_consistency(
        None, None, _GOOD_LATTICE, run_hermit=False,
        artefacts=_GOOD_ARTEFACTS, configurations=[_FiniteQ([1, 2])],
    )
    assert res.ok and res.status == Status.OK


def test_alg3_c3_rejects_cycle_with_witness():
    res = check_consistency(
        None, None, {0: {1}, 1: {2}, 2: {0}}, run_hermit=False,
    )
    assert not res.ok and res.status == Status.C3_VIOLATED
    assert res.details["cycle_witness"]


def test_alg3_c4_rejects_edge_without_artefact_flow():
    """[SVD §2.1] ребро согласовано, если Out(m_i) ∩ In(m_j) ≠ ∅."""
    bad = {**_GOOD_ARTEFACTS, 1: {"in": {"a"}, "out": {"zzz"}}}
    res = check_consistency(
        None, None, _GOOD_LATTICE, run_hermit=False, artefacts=bad,
    )
    assert not res.ok and res.status == Status.C4_VIOLATED
    assert tuple(res.details["c4_edge"]) == (1, 2)


def test_alg3_c5_rejects_undeclared_infinite_domain():
    class _NoFlag:
        pass

    res = check_consistency(
        None, None, _GOOD_LATTICE, run_hermit=False,
        artefacts=_GOOD_ARTEFACTS, configurations=[_FiniteQ([1]), _NoFlag()],
    )
    assert not res.ok and res.status == Status.C5_VIOLATED


def test_alg3_structural_checks_run_in_declared_order():
    """Two-stage design: with C3 and C4 both violated, C3 is reported first."""
    res = check_consistency(
        None, None, {0: {1}, 1: {0}}, run_hermit=False,
        artefacts={0: {"in": set(), "out": set()}, 1: {"in": set(), "out": set()}},
    )
    assert res.status == Status.C3_VIOLATED


def _fresh_world():
    from owlready2 import (
        AllDifferent, FunctionalProperty, ObjectProperty, Thing, World,
    )

    w = World()
    onto = w.get_ontology("http://golden.test/alg3_stage_a.owl")
    with onto:
        class GModel(Thing):
            pass

        class GHypothesis(Thing):
            pass

        class g_implemented_by(ObjectProperty, FunctionalProperty):
            domain = [GHypothesis]
            range = [GModel]

        GHypothesis.is_a.append(g_implemented_by.some(GModel))
        m1, m2 = GModel("m1"), GModel("m2")
        AllDifferent([m1, m2])
    return w, onto, GHypothesis, m1, m2


def _hermit_consistent(w, onto) -> bool:
    import owlready2

    try:
        with onto:
            owlready2.sync_reasoner_hermit(w, infer_property_values=False, debug=0)
        return True
    except Exception as exc:
        assert "Inconsistent" in type(exc).__name__, exc
        return False


@pytest.mark.reasoner
def test_alg3_stage_a_hermit_detects_functional_violation():
    """Stage A (semantic, C2): a hypothesis implemented by two *different*
    models violates FunctionalProperty + AllDifferent and must be reported
    inconsistent by HermiT. Isolated World, shared hyppo ontology untouched."""
    pytest.importorskip("owlready2")
    w, onto, GHypothesis, m1, m2 = _fresh_world()
    with onto:
        h = GHypothesis("h_two_models")
        h.g_implemented_by = m1
        # class-level value restriction forces the *other* model:
        # functionality + AllDifferent(m1, m2) => inconsistent
        h.is_a.append(onto.g_implemented_by.value(m2))
    assert not _hermit_consistent(w, onto), (
        "HermiT must flag two distinct models on a functional property (C2)"
    )


@pytest.mark.reasoner
def test_alg3_stage_a_owa_cannot_see_missing_model():
    """[SVD §2.2] Negative golden claim: under OWA a model-less hypothesis is
    *consistent* (the existential may be satisfied by an unknown model) — this
    is exactly why «у гипотезы нет ни одной реализующей модели» is delegated
    to the marker layer (CWA), not to the reasoner."""
    pytest.importorskip("owlready2")
    w, onto, GHypothesis, m1, m2 = _fresh_world()
    with onto:
        GHypothesis("h_bad")  # no implementing model asserted
    assert _hermit_consistent(w, onto), (
        "OWA must NOT flag a model-less hypothesis — marker layer's job"
    )

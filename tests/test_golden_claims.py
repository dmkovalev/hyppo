"""Golden claim tests: pin statements made in the papers to the real implementations.

Sources of the claims:
  [SVD]  thesis/papers/sht_dostupnost_v1.tex — журнал «Системы высокой доступности»

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
from hyppo.graph_constructor._base import GraphConstructor

# ---------------------------------------------------------------------------
# Fixtures: the Norne HybridCRM hypothesis set (formulas as in [SVD] §3.9)
# ---------------------------------------------------------------------------

NORNE_HYPS = [
    ("H1", "I_agg = w_ij * I_j"),
    ("H2", "q_f = a_f*q_f_prev + b_f*I_agg"),
    ("H3", "q_s = a_s*q_s_prev + b_s*I_agg"),
    ("H4", "q_c = w_f*q_f + (1-w_f)*q_s"),
    # Поток работ упорядочен по поставке: H6 (q_prim) и GRP (J) исполняются
    # раньше потребителя H5 (опр. выводимости требует tau(h_a) < tau(h_b)).
    ("H6", "q_prim = q_prev*exp(-dt*taup)"),
    ("GRP", "J = J0 + dJ_grp"),
    ("H5", "q_liq_phys = J*q_c + q_prim"),
    ("H7", "l_ml = w0 + w1*q_c + w2*l_prev + w3*I_mean"),
    ("H8", "l = g*q_liq_phys + (1-g)*l_ml"),
    # Материальный баланс связан с прогнозом жидкости H8 (выход l), а не с сырым
    # замером Wlout: закачка минус спрогнозированный отбор жидкости [SVD §3.9].
    ("H11", "Sw = Sw_prev + (Winj - l)*dt/Vp"),
    ("H12", "krw = ((Sw-Swc)/(1-Swc-Sor))**nw"),
    ("H12b", "kro = ((1-Sw-Sor)/(1-Swc-Sor))**no"),
    ("H13", "fw = 1/(1 + kro*muw/(krw*muo))"),
    ("H14", "o_p = 1 - fw"),
    ("H15", "o = gw*o_p + (1-gw)*o_m"),
    ("H19", "q_oil = l * o"),
]

# The 19 derived_by edges of the CRM graph (dissertation ch. 4; hand-derivable from the LHS/RHS
# variable flow of the equations above, independently of the implementation).
# H8->H11 links the liquid-prediction branch to the material balance (H11 uses l).
NORNE_GOLDEN_EDGES = {
    ("H1", "H2"),
    ("H1", "H3"),
    ("H2", "H4"),
    ("H3", "H4"),
    ("H4", "H5"),
    ("H4", "H7"),
    ("H6", "H5"),
    ("GRP", "H5"),
    ("H5", "H8"),
    ("H7", "H8"),
    ("H8", "H19"),
    ("H8", "H11"),
    ("H11", "H12"),
    ("H11", "H12b"),
    ("H12", "H13"),
    ("H12b", "H13"),
    ("H13", "H14"),
    ("H14", "H15"),
    ("H15", "H19"),
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


def _build_graph(hyps: list[_Hyp]) -> nx.DiGraph:
    wf = _Workflow([[h] for h in hyps])
    return GraphConstructor(list(hyps), wf).hyp_graph


def _named_edges(g: nx.DiGraph) -> set[tuple[str, str]]:
    return {(u.name, v.name) for u, v in g.edges()}


# ---------------------------------------------------------------------------
# Algorithm 1 — build_graph  [SVD §3.9, рис. 3]
# ---------------------------------------------------------------------------


def test_alg1_norne_graph_matches_figure():
    """Гл. 4 диссертации: «16~гипотез ... 19~рёбер при~глубине графа~10»,
    «рёбра выведены автоматически из потока переменных уравнений»."""
    hyps = [_Hyp(n, f) for n, f in NORNE_HYPS]
    g = _build_graph(hyps)
    assert _named_edges(g) == NORNE_GOLDEN_EDGES
    assert g.number_of_nodes() == 16
    assert g.number_of_edges() == 19
    assert nx.is_directed_acyclic_graph(g)
    assert nx.dag_longest_path_length(g) == 10


# ---------------------------------------------------------------------------
# Algorithm 2 — incremental add equals full rebuild  (лемма 2)
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
    as rebuilding the graph from scratch."""
    rng = random.Random(seed)
    hyps = _random_formula_hyps(rng, 12)

    full = _named_edges(_build_graph(hyps))

    wf = _Workflow([[h] for h in hyps])
    lat = GraphConstructor(list(hyps[:-1]), wf)
    lat.add_hypothesis(hyps[-1])
    incremental = _named_edges(lat.hyp_graph)

    assert incremental == full


# ---------------------------------------------------------------------------
# Algorithm 4 — plan == cascade closure oracle  [SVD положение о каскаде]
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
# Theorem 1 — correctness and minimality of the plan  (теорема 1)
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
        g.add([{f"x{i}", f"x{i - 1}"}] if i else [{f"x{i}"}])
    for i in range(n - 1):
        g.connect(i, i + 1)
    return g


def test_alg1_complexity_quadratic_pair_enumeration():
    """(лемма 1) Algorithm 1 costs O(|H|^2 · v): the number of reachable pairs
    enumerated in ``build`` equals n(n-1)/2 on a chain, i.e. grows quadratically.
    Each pair is decided by the cheap Out∩In test — no transitive closure per
    pair (bridge, Theorem thm:build)."""
    counts = {}
    for n in (10, 20, 40):
        g = _chain_graph(n)
        calls = [0]

        class _CountIn(list):
            def __getitem__(self, i):
                calls[0] += 1
                return list.__getitem__(self, i)

        g._in = _CountIn(g._in)  # count In(h_j) accesses = pair enumerations
        g.build()
        counts[n] = calls[0]
        assert calls[0] == n * (n - 1) // 2

    assert counts[20] / counts[10] == pytest.approx(4, rel=0.15)
    assert counts[40] / counts[20] == pytest.approx(4, rel=0.15)


def test_alg2_complexity_linear_in_hypotheses():
    """(лемма 2) Algorithm 2 decides exactly the pairs of the new hypothesis with
    the hypotheses related to it by reachability -- here, appended to a chain of
    n hypotheses, exactly n pairs (linear), against n(n-1)/2 for a full rebuild;
    and the incremental result equals the full rebuild (thm:build (III))."""
    for n in (10, 20, 40):
        g = _chain_graph(n)
        g.build()  # setup: the existing graph
        calls = [0]
        orig = g._pair

        def counting(*a, **k):
            calls[0] += 1
            return orig(*a, **k)

        g._pair = counting
        g.add_hypothesis([{f"x{n}", f"x{n - 1}"}], inputs={f"x{n - 1}"}, preds=[n - 1])
        assert calls[0] == n
        full = _chain_graph(n)
        full.add([{f"x{n}", f"x{n - 1}"}], inputs={f"x{n - 1}"})
        full.connect(n - 1, n)
        assert set(full.build()) == g.hyp_graph


class _CountingSet(set):
    counter = 0

    def __iter__(self):
        type(self).counter += len(self)
        return super().__iter__()


def test_alg4_complexity_linear_in_nodes_plus_edges():
    """Algorithm 4 runs in O(|V|+|E|): adjacency traversals grow
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
# Algorithm 3 — two-stage consistency check  (двухэтапная проверка; положение 1)
# ---------------------------------------------------------------------------
# Claim [положение 1]: «корректная определённость проверяется статически,
# до запуска моделей, двухэтапным (семантическим и структурным) алгоритмом»;
# stage B checks C3 (ацикличность), C4 (согласованность артефактов),
# C5 (конечность доменов конфигурации).

from hyppo.ontology.consistency import (  # noqa: E402 -- colocated with its test section
    Status,
    check_consistency,
)


class _FiniteQ:
    finite = True

    def __init__(self, values):
        self.values = values


_GOOD_GRAPH = {0: {1}, 1: {2}, 2: set()}
_GOOD_ARTEFACTS = {
    0: {"in": {"raw"}, "out": {"a"}},
    1: {"in": {"a"}, "out": {"b"}},
    2: {"in": {"b"}, "out": {"c"}},
}


def test_alg3_stage_b_accepts_correct_experiment():
    res = check_consistency(
        None,
        None,
        _GOOD_GRAPH,
        run_hermit=False,
        artefacts=_GOOD_ARTEFACTS,
        configurations=[_FiniteQ([1, 2])],
    )
    assert res.ok and res.status == Status.OK


def test_alg3_c3_rejects_cycle_with_witness():
    res = check_consistency(
        None,
        None,
        {0: {1}, 1: {2}, 2: {0}},
        run_hermit=False,
    )
    assert not res.ok and res.status == Status.C3_VIOLATED
    assert res.details["cycle_witness"]


def test_alg3_c4_rejects_edge_without_artefact_flow():
    """[SVD §2.1] ребро согласовано, если Out(m_i) ∩ In(m_j) ≠ ∅."""
    bad = {**_GOOD_ARTEFACTS, 1: {"in": {"a"}, "out": {"zzz"}}}
    res = check_consistency(
        None,
        None,
        _GOOD_GRAPH,
        run_hermit=False,
        artefacts=bad,
    )
    assert not res.ok and res.status == Status.C4_VIOLATED
    assert tuple(res.details["c4_edge"]) == (1, 2)


def test_alg3_c5_rejects_undeclared_infinite_domain():
    class _NoFlag:
        pass

    res = check_consistency(
        None,
        None,
        _GOOD_GRAPH,
        run_hermit=False,
        artefacts=_GOOD_ARTEFACTS,
        configurations=[_FiniteQ([1]), _NoFlag()],
    )
    assert not res.ok and res.status == Status.C5_VIOLATED


def test_c7_accepts_grounded_and_rejects_ungrounded_variable():
    """C7: каждая переменная уравнения гипотезы обязана быть объявленным
    доменным термином (SHACL-семантика, замкнутый мир). Опциональная проверка —
    без domain_terms/hypothesis_vars она skipped и не влияет на C1–C5."""
    vocab = {"Sw", "Winj", "l", "fw", "muw", "muo"}
    ok = check_consistency(
        None,
        None,
        _GOOD_GRAPH,
        run_hermit=False,
        domain_terms=vocab,
        hypothesis_vars={0: {"Sw", "Winj", "l"}, 1: {"fw", "muw"}},
    )
    assert ok.ok and ok.status == Status.OK and ok.details["c7"].startswith("passed")

    bad = check_consistency(
        None,
        None,
        _GOOD_GRAPH,
        run_hermit=False,
        domain_terms=vocab,
        hypothesis_vars={0: {"Sw"}, 1: {"fw", "phi_unknown"}},
    )
    assert not bad.ok and bad.status == Status.C7_VIOLATED
    assert bad.details["c7_hypothesis"] == 1
    assert bad.details["c7_ungrounded"] == ["phi_unknown"]

    # без словаря C7 не выполняется (обратная совместимость)
    skip = check_consistency(None, None, _GOOD_GRAPH, run_hermit=False)
    assert skip.details["c7"] == "skipped"


# --- Paper conditions 4 and 5 (§2.4): CRM running example (ch. 2) ---------------
# ids: H1..H16 -> 1..16; 19 edges of Algorithm 1 (chapter 4 graph)
_CRM_EDGES = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (6, 5), (15, 5), (4, 7),
              (5, 8), (7, 8), (8, 9), (9, 10), (9, 11), (10, 12), (11, 12),
              (12, 13), (13, 14), (8, 16), (14, 16)]
_CRM_GRAPH = {i: set() for i in range(1, 17)}
for _a, _b in _CRM_EDGES:
    _CRM_GRAPH[_a].add(_b)
_CRM_TASKS = {"t1": [1], "t2": [2, 3], "t3": [4], "t4": [6], "t5": [15], "t6": [5],
              "t7": [7], "t8": [8], "t9": [9], "t10": [10, 11], "t11": [12],
              "t12": [13], "t13": [14], "t14": [16]}
_CRM_WF = [("t1", "t2"), ("t2", "t3"), ("t3", "t6"), ("t4", "t6"), ("t5", "t6"),
           ("t3", "t7"), ("t6", "t8"), ("t7", "t8"), ("t8", "t9"), ("t9", "t10"),
           ("t10", "t11"), ("t11", "t12"), ("t12", "t13"), ("t13", "t14")]
_CRM_DOM = {i: [0, 1] for i in range(1, 17)}


def _proj(hyp_graph=None, forbidden=None):
    return check_consistency(
        None, None, hyp_graph or _CRM_GRAPH, run_hermit=False,
        task_hypotheses=_CRM_TASKS, workflow_edges=_CRM_WF,
        param_domains=_CRM_DOM, forbidden_combos=forbidden or [],
    )


def test_condition5_accepts_correct_and_rejects_backward_dependency():
    """Условие 5 (alg. consistence): причинный порядок согласован с потоком работ."""
    ok = _proj()
    assert ok.ok and ok.details["cond5"].startswith("passed")
    # модификация 2: баланс H9 (задача t9) читает f_w от H12 (задача t11, потомок t9)
    bad_lat = {k: set(v) for k, v in _CRM_GRAPH.items()}
    bad_lat[12] = {13, 9}
    bad_lat[9] = set()      # без рёбер H9->H10, H9->H11, чтобы не было цикла
    bad = _proj(hyp_graph=bad_lat)
    assert not bad.ok and bad.status == Status.COND5_VIOLATED
    assert bad.details["cond5"]["task"] == "t9"
    assert bad.details["cond5"]["dependency_task"] == "t11"


def test_condition4_accepts_nonempty_and_rejects_empty_projection():
    """Условие 4: pi_t(V) != empty для каждой задачи (CSP-выполнимость)."""
    ok = _proj()
    assert ok.ok and ok.details["cond4"].startswith("passed")
    # запрет обоих значений показателя Кори H10 обнуляет проекцию задачи t10
    empty = _proj(forbidden=[{10: 0}, {10: 1}])
    assert not empty.ok and empty.status == Status.COND4_VIOLATED
    assert empty.details["cond4_task"] == "t10"
    # запрет одной совместной пары (n_w, n_o) не обнуляет проекцию
    sat = _proj(forbidden=[{10: 0, 11: 0}])
    assert sat.ok and sat.details["cond4"].startswith("passed")


def test_conditions_4_5_skipped_when_data_absent():
    """Обратная совместимость: без workflow/доменов условия 4–5 не выполняются."""
    res = check_consistency(None, None, _GOOD_GRAPH, run_hermit=False)
    assert res.details["cond4"] == "skipped" and res.details["cond5"] == "skipped"


def test_alg3_structural_checks_run_in_declared_order():
    """Two-stage design: with C3 and C4 both violated, C3 is reported first."""
    res = check_consistency(
        None,
        None,
        {0: {1}, 1: {0}},
        run_hermit=False,
        artefacts={0: {"in": set(), "out": set()}, 1: {"in": set(), "out": set()}},
    )
    assert res.status == Status.C3_VIOLATED


def _fresh_world():
    from owlready2 import (
        AllDifferent,
        FunctionalProperty,
        ObjectProperty,
        Thing,
        World,
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

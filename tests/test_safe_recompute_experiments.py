import networkx as nx
import pytest

from hyppo.planner import causal_refine as cr

from examples.research.safe_recompute import e2_mincost, e3_causal
from examples.research.safe_recompute.common import DATA, load_json

BASELINES = ("N_read", "N_needed_read", "cascade_read", "full", "best_simple")


def test_e2_ordering_of_plans(tmp_path):
    out = e2_mincost.main(seeds=2, out=tmp_path / "e2.json", graphs_limit=4)
    assert out["rows"]
    for row in out["rows"]:
        assert row["feasible_all"]  # is_feasible для всех планов проверяется в one_case
        for b in BASELINES:
            assert row["cost_mincost"] <= row[f"cost_{b}"] + 1e-9
        assert row["cost_best_simple"] == min(row["cost_N_needed_read"], row["cost_full"])


def test_e2_deterministic_and_summary_consistent(tmp_path):
    a = e2_mincost.main(seeds=2, out=tmp_path / "a.json", graphs_limit=3)
    b = e2_mincost.main(seeds=2, out=tmp_path / "b.json", graphs_limit=3)
    assert a["rows"] == b["rows"]
    assert a["summary"] == e2_mincost.summarize(a["rows"])
    for key, e in a["summary"].items():
        rho, t = key.split(":")
        sel = [x for x in a["rows"] if x["rho"] == float(rho) and x["targets"] == t]
        assert e["n"] == len(sel)
        assert e["frac_N_read_gt_full"] == pytest.approx(
            sum(x["cost_N_read"] > x["cost_full"] + 1e-9 for x in sel) / len(sel))


def test_crm_costs_sum_to_measured_total():
    w = e2_mincost.crm_costs()
    pc = load_json(DATA / "timing_brugge.json")["per_component_ms"]
    assert sum(w.values()) == pytest.approx(sum(pc.values()), rel=1e-6)


def test_needed_plan_examples():
    import networkx as nx
    g = nx.DiGraph([("a", "b")])
    assert e2_mincost.needed_plan(g, {"b"}, {"b"}) == (set(), {"b"})
    g = nx.DiGraph([("a", "b"), ("b", "c")])
    assert e2_mincost.needed_plan(g, {"c"}, {"b"}) == ({"c"}, {"b"})


@pytest.fixture(scope="module")
def e3_out(tmp_path_factory):
    return e3_causal.main(synthetic=200, out=tmp_path_factory.mktemp("e3") / "e3.json")


def test_e3_smoke(e3_out):
    assert e3_out["crm_coarse"]["n_caus_mean"] <= e3_out["crm_coarse"]["n_mean"]
    assert set(e3_out["synthetic"]) == {"0.0", "0.25", "0.5", "0.75"}


def test_e3_projectile_pinned(e3_out):
    ex, al = e3_out["projectile_exact"], e3_out["projectile_aligned"]
    assert (ex["strict_cases"], ex["cases"]) == (7, 24)
    assert len(ex["strict"]) == 7
    assert al["strict_cases"] == 0


def test_e3_deterministic(tmp_path):
    import json
    a = e3_causal.main(synthetic=40, out=tmp_path / "a.json")
    b = e3_causal.main(synthetic=40, out=tmp_path / "b.json")
    assert a == b
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["seed"] == e3_causal.SEED


def test_e3_synthetic_monotone_in_multi_output_share(e3_out):
    syn = [e3_out["synthetic"][str(q)] for q in e3_causal.QS]
    shares = [v["multi_output_share"] for v in syn]
    ratios = [v["mean_ratio"] for v in syn]
    assert shares == sorted(shares) and shares[0] == 0.0
    assert all(r2 <= r1 + 1e-12 for r1, r2 in zip(ratios, ratios[1:]))
    assert ratios[0] == 1.0 and syn[0]["frac_strict"] == 0.0  # одновыходные: N_caus = N
    assert all(0.0 < r <= 1.0 for r in ratios)


def test_e3_crm_graphs_match_common():
    from examples.research.safe_recompute import common
    fine, coarse = e3_causal.norne(False), e3_causal.norne(True)
    assert e3_causal.diss_edges(fine) == set(common.CRM_EDGES)
    assert fine.G.number_of_edges() == len(common.CRM_EDGES) == 18
    g = common.crm_graph()
    assert nx.dag_longest_path_length(g) == 10
    assert {e3_causal.TO_DISS.get(h, h) for h in fine.G} == set(g.nodes)
    # граф компонент = фактор-граф crm_graph() по NORNE_COMP
    comp = {e3_causal.TO_DISS.get(h, h): c for c, hs in e3_causal.NORNE_COMP.items() for h in hs}
    quotient = {(comp[a], comp[b]) for a, b in g.edges if comp[a] != comp[b]}
    assert set(coarse.G.edges) == quotient


def _single_change_sets(inst):
    s, i, e, pmap = inst.to_library()
    for p in sorted(pmap):
        yield p, inst.n_hyp({h for h, _ in pmap[p]}), cr.n_caus(s, i, e, changed_params=pmap[p], stale=set())


def test_e3_crm_fine_n_equals_n_caus():
    rows = list(_single_change_sets(e3_causal.norne(False)))
    assert rows
    for p, n, nc in rows:
        assert n == nc, p


def test_e3_n_caus_subset_of_n_everywhere():
    import random
    insts = [e3_causal.projectile(True), e3_causal.projectile(False),
             e3_causal.norne(False), e3_causal.norne(True)]
    rng = random.Random(1)
    insts += [e3_causal.gen_instance(rng, q) for q in e3_causal.QS for _ in range(50)]
    for inst in insts:
        for p, n, nc in _single_change_sets(inst):
            assert nc <= n, p


from examples.research.safe_recompute import e5_planning_time  # noqa: E402


def test_e5_smoke(tmp_path):
    out = e5_planning_time.main(sizes=[50, 100, 200], reps=2, out=tmp_path / "e5.json")
    assert out["chained"]["slope"] < 1.4
    assert out["direct"]["slope"] > out["chained"]["slope"]

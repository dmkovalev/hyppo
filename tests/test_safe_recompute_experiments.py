import pytest

from examples.research.safe_recompute import e2_mincost
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


from examples.research.safe_recompute import e5_planning_time  # noqa: E402


def test_e5_smoke(tmp_path):
    out = e5_planning_time.main(sizes=[50, 100, 200], reps=2, out=tmp_path / "e5.json")
    assert out["chained"]["slope"] < 1.4
    assert out["direct"]["slope"] > out["chained"]["slope"]

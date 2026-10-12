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


from examples.research.safe_recompute import e6_systems  # noqa: E402


def test_e6_simulated_scaling_smoke(tmp_path):
    out = e6_systems.simulate_corpora(events_per_graph=20, out=tmp_path / "e6.json", graphs_limit=3)
    for system in ("snakemake", "dvc", "hyppo_safe"):
        assert system in out["per_system"]
    assert out["per_system"]["hyppo_safe"]["missed_total"] == 0


def test_e6_brugge_store_and_events():
    from examples.research.baseline_filecache.config import SMALL_GRID
    st = e6_systems.brugge_store(SMALL_GRID)
    assert st.consumers["corey_ref"] == {"wct/2_2.npy", "opr/uto_1_2_2.json", "opr/pos_1_2_2.json"}
    assert st.consumers["ridge"] == {"hyb/uto_1.npy", "hyb/pos_1.npy"}
    # объявлены только corey_ref:fit_wct и well_status:fit_liq
    assert st.visible("corey_ref", "dvc") == {"wct/2_2.npy"}
    assert st.visible("well_status", "snakemake") == {"hyb/uto_1.npy", "hyb/pos_1.npy"}
    assert st.visible("tau", "dagster") == set()
    assert st.visible("tau", "hyppo_safe") == {"hyb/uto_1.npy", "hyb/pos_1.npy"}
    kinds = [e[0] for e in e6_systems.EVENTS]
    assert (kinds.count("param"), kinds.count("evict"), kinds.count("revise")) == (5, 5, 3)
    # три правки ridge без изменения float
    ws = e6_systems.worlds(e6_systems.EVENTS)
    same = [k for k, e in enumerate(e6_systems.EVENTS) if e[0] == "param" and ws[k + 1] == ws[k]]
    assert len(same) == 3


def test_e6_recorded_real_runs_validate_100pct():
    """Записанные реальные прогоны (data/e6_systems.json) воспроизводятся
    симулятором: run_sequence (forced) и native_evict -- 100 % событий."""
    path = DATA / "e6_systems.json"
    if not path.exists():
        pytest.skip("нет data/e6_systems.json")
    data = load_json(path)
    assert data.get("real"), "нет реальных прогонов"
    for grid, real in data["real"].items():
        val = e6_systems.validate_simulators(real)
        for name in e6_systems.REAL_SYSTEMS:
            assert val[name]["forced"]["share"] == 1.0, (grid, name, val[name]["forced"]["rows"])
            assert val[name]["native_evict"]["share"] == 1.0, (grid, name)
            assert val[name]["initial_all"] and not val[name]["initial_wrong"], (grid, name)
        assert real["systems"]["hyppo_safe"]["missed_total"] == 0
        assert real["systems"]["dvc"]["missed_total"] > 0  # необъявленная tau
        assert real["gold"]["delta_exact_equals_rounded"]
        for name in e6_systems.REAL_SYSTEMS:
            rec = real["systems"][name]
            assert rec["wrong_at_end"] == 0 and "miss_spans" in rec
    small = data["real"]["small"]
    assert small["snakemake_triggers_per_event"] == {"checked": 13, "same": 13}
    for name in ("snakemake", "dagster", "dvc"):
        assert all(n["absent_source"] == "measured" for n in small["systems"][name]["native_evict"]), name
    # пропуск tau у инструментов исправлен следующей правкой ridge (одно событие)
    assert small["systems"]["dvc"]["miss_spans"] == [
        {"from_event": 9, "repaired_at_event": 10, "events_wrong": 1, "jobs": 4}]


def test_e6_corpora_deterministic_and_consistent(tmp_path):
    a = e6_systems.simulate_corpora(events_per_graph=30, out=tmp_path / "a.json", graphs_limit=4)
    b = e6_systems.simulate_corpora(events_per_graph=30, out=tmp_path / "b.json", graphs_limit=4)
    assert a["per_system"] == b["per_system"]
    ps = a["per_system"]
    for p in ("0.0", "0.5", "1.0"):
        assert ps["hyppo_safe"]["by_p"][p]["missed_total"] == 0
        assert ps["hyppo_cascade"]["by_p"][p]["missed_total"] == 0
        # cascade-режим комплекса не меньше safe
        assert ps["hyppo_cascade"]["by_p"][p]["recomputed_total"] >= ps["hyppo_safe"]["by_p"][p]["recomputed_total"]
    # пропуски внешних систем не растут с долей объявленных пар; при p=1 их нет
    for s in ("snakemake", "dvc", "dagster"):
        m = [ps[s]["by_p"][p]["missed_total"] for p in ("0.0", "0.5", "1.0")]
        assert m[0] >= m[1] >= m[2] == 0, (s, m)
    assert ps["snakemake"]["native_evict"]["executed"] <= ps["snakemake"]["native_evict"]["evict_events"]
    assert ps["dvc"]["native_evict"]["executed"] == 0
    fav = a["favourable_worlds"]["by_p"]["1.0"]["dvc"]
    assert fav["fewer_safe"] > 0 and fav["fewer_with_miss"] == 0
    assert fav["fewer_safe_by_kind"]["param"] == fav["fewer_safe"]


def test_e6_run_hyppo_small(tmp_path):
    """Комплекс (Manager, режим safe) на малой сетке: evict -- только узел,
    правка ridge -- fit_liq и потомки, необъявленная ревизия видна."""
    from examples.research.baseline_filecache.config import SMALL_GRID
    from examples.research.safe_recompute import ports
    if not (ports.brugge_run_dir() / "brugge_perwell.npz").exists():
        pytest.skip("нет данных Brugge (HYPPO_BRUGGE_RUN)")
    evs = [("evict", "hyb/uto_1.npy"), ("param", "ridge", "1.0"), ("revise", "tau")]
    res = e6_systems.run_hyppo(tmp_path / "h", SMALL_GRID, evs)
    hyb = {"hyb/uto_1.npy", "hyb/pos_1.npy"}
    opr = {"opr/uto_1_2_2.json", "opr/pos_1_2_2.json"}
    assert len(res["initial"]) == 7
    assert res["after"] == [{"hyb/uto_1.npy"}, hyb | opr, hyb | opr]
    assert res["outputs"][1] == res["outputs"][0]


def test_e6_derive_fields_miss_spans():
    ev = lambda w, m: {"wrong": w, "missed": m}
    rec = {"events": [ev([], []), ev(["a", "b"], ["a", "b"]), ev(["b"], []), ev(["b", "c"], ["c"])]}
    e6_systems.derive_fields(rec)
    assert rec["wrong_at_end"] == 2
    assert rec["miss_spans"] == [
        {"from_event": 1, "repaired_at_event": 2, "events_wrong": 1, "jobs": 1},
        {"from_event": 1, "repaired_at_event": None, "events_wrong": 3, "jobs": 1},
        {"from_event": 3, "repaired_at_event": None, "events_wrong": 1, "jobs": 1},
    ]


def test_e6_validate_repeated_evict_uses_event_positions():
    """Повтор одного и того же evict с разным forced-результатом (между ними --
    необъявленная ревизия, узел устарел): нативная строка системы без нативного
    режима (bazel) сверяется со своим событием, а не с первым вхождением."""
    j, o = "hyb/uto_1.npy", "opr/uto_1_2_2.json"
    hyb_opr = ["hyb/pos_1.npy", j, "opr/pos_1_2_2.json", o]
    events = [["evict", j], ["revise", "tau"], ["evict", j]]
    real = {"grid": "small", "events": events, "declared": list(e6_systems.DECLARED),
            "gold": {"delta_exact": [[], hyb_opr, []]},
            "systems": {"bazel": {
                "initial_all": True, "initial_wrong": [],
                "events": [{"executed": [j], "wrong": []}, {"executed": [], "wrong": hyb_opr},
                           {"executed": [j, o], "wrong": ["hyb/pos_1.npy", "opr/pos_1_2_2.json"]}],
                "native_evict": [{"event": events[0], "executed": [j], "absent": []},
                                 {"event": events[2], "executed": [j, o], "absent": []}]}}}
    val = e6_systems.validate_simulators(real)["bazel"]
    assert val["forced"]["share"] == 1.0 and val["native_evict"]["share"] == 1.0
    real["systems"]["bazel"]["native_evict"][1]["event"] = ["evict", o]
    with pytest.raises(ValueError):
        e6_systems.validate_simulators(real)


def test_e6_native_by_kind_and_sensitivity(tmp_path):
    c = e6_systems.simulate_corpora(events_per_graph=30, graphs_limit=3)
    assert c["meta"]["p_output_changed"] == 0.7 and c["meta"]["weights"] == {"param": 6, "evict": 3, "revise": 1}
    for s, v in c["per_system"].items():
        for p, x in v["by_p"].items():
            f, n = x["by_kind"], x["by_kind_native"]
            for k in ("param", "revise"):
                assert (f[k]["recomputed"], f[k]["missed"]) == (n[k]["recomputed"], n[k]["missed"]), (s, p, k)
            if s == "dvc":
                assert n["evict"]["recomputed"] == 0 and n["evict"]["absent"] == 0
            if s == "dagster":
                assert n["evict"]["absent"] == n["evict"]["events"]
    sens = e6_systems.sensitivity(events_per_graph=30, graphs_limit=3, p_values=(0.5, 1.0))
    a, b = sens["by_P"]["0.5"], sens["by_P"]["1.0"]
    # при P=1 выход всегда меняется: благоприятных миров нет, cutoff = hyppo_safe на param
    assert b["dvc_fewer_safe"]["1.0"] == 0 and a["dvc_fewer_safe"]["1.0"] > 0
    for P in (a, b):
        assert all(x["missed"] == 0 for x in P["per_system"]["hyppo_safe"].values())

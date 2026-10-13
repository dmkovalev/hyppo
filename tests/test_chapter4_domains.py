"""Chapter 4 domain experiments (BGM, HCP): the graph is derived by Algorithm 1,
the experiment passes the consistency check, the ontology rules fire as
reported, and reasoning time is measured."""

from __future__ import annotations

import shutil

import pytest

from hyppo.adapters import bgm_adapter as B
from hyppo.adapters import hcp_adapter as N

from examples.research.chapter4._domain_run import (
    cascade,
    derive_graph,
    edge_diff,
    ic_conditions_1_3,
    reset_abox,
    structural_check,
    time_rl_materialisation,
)

needs_java = pytest.mark.skipif(shutil.which("java") is None, reason="HermiT needs Java")

BGM_EXPECTED_EDGES = sorted([
    ("age_cells", "local_volume_density"),
    ("density_law", "local_volume_density"),
    ("SFR", "star_birthrate"),
    ("IMF", "star_birthrate"),
    ("age_cells", "star_birthrate"),
    ("age_cells", "evolutionary_tracks"),
    ("local_volume_density", "evolutionary_tracks"),
    ("star_birthrate", "local_luminosity"),
    ("local_volume_density", "local_luminosity"),
    ("evolutionary_tracks", "local_luminosity"),
    ("evolutionary_tracks", "velocity_dispersion"),
    ("local_luminosity", "dynamical_self_consistency"),
    ("evolutionary_tracks", "dynamical_self_consistency"),
    ("velocity_dispersion", "dynamical_self_consistency"),
])


@pytest.fixture
def clean_abox():
    reset_abox()
    yield
    reset_abox()


# ---------------------------------------------------------------- BGM graph
def test_bgm_structures_satisfy_u0():
    for h in B.BGM_HYPOTHESES:
        h.check_u0()
        assert h.outputs and h.inputs
    assert set(B.BGM_VARIABLES) >= set().union(*(h.variables for h in B.BGM_HYPOTHESES))


def test_bgm_graph_is_derived_by_algorithm1():
    g = derive_graph(B.BGM_HYPOTHESES, B.BGM_TASKS, B.BGM_TASK_EDGES)
    assert len(g.nodes) == 10
    assert g.edges == BGM_EXPECTED_EDGES
    assert g.depth == 4
    assert g.hypothesis_graph_agrees  # HypothesisGraph.build gives the same edges
    assert g.competes == [] and g.order_conflicts == []
    sig = g.signatures
    assert set(sig["age_cells"]["out"]) & set(sig["star_birthrate"]["in"]) == {"x_c", "d"}


def test_bgm_diff_with_dissertation_graph():
    g = derive_graph(B.BGM_HYPOTHESES, B.BGM_TASKS, B.BGM_TASK_EDGES)
    d = edge_diff(g.edges, B.DISSERTATION_EDGES)
    assert d["only_expected"] == []  # every drawn edge is derived
    assert d["only_derived"] == [("age_cells", "evolutionary_tracks"),
                                 ("evolutionary_tracks", "local_luminosity")]


def test_bgm_dissertation_workflow_violates_condition5():
    gd = derive_graph(B.BGM_HYPOTHESES, B.DISSERTATION_TASKS, B.BGM_TASK_EDGES)
    res = structural_check(B.BGM_HYPOTHESES, B.DISSERTATION_TASKS, B.BGM_TASK_EDGES,
                           gd.edges, B.BGM_VARIABLES, engine="skip")
    assert not res["ok"] and res["status"].startswith("condition 5")
    v = res["details"]["cond5"]
    assert (v["hypothesis"], v["dependency"]) == ("local_luminosity", "evolutionary_tracks")


def test_bgm_cascade_4_of_10():
    g = derive_graph(B.BGM_HYPOTHESES, B.BGM_TASKS, B.BGM_TASK_EDGES)
    tail = ["dynamical_self_consistency", "local_luminosity", "star_birthrate"]
    assert cascade(g.graph, "IMF") == sorted(["IMF", *tail])
    assert cascade(g.graph, "SFR") == sorted(["SFR", *tail])


def test_bgm_configuration_space_and_invariants():
    assert B.CONFIG_SPACE_SIZE == 1188
    assert sum(len(lv) - 1 for _, lv in B.CONFIG_AXES.values()) == 18
    assert all(not v for v in B.nominal_invariant_checks().values())
    with pytest.raises(ValueError):
        B.IMFParams(alpha=2.35, m_lo=150.0, m_up=120.0)
    with pytest.raises(ValueError):
        B.AgeBinsParams(edges=[0.0, 1.0, 0.5])


def test_bgm_rival_imfs_compete():
    specs, tasks = B.rival_hypotheses()
    g = derive_graph(specs, tasks, B.BGM_TASK_EDGES)
    assert g.competes == [("IMF_Haywood", "IMF_Kroupa"), ("IMF_Haywood", "IMF_Salpeter"),
                          ("IMF_Kroupa", "IMF_Salpeter")]
    assert all(("IMF_" + r, "star_birthrate") in g.edges for r in ("Kroupa", "Haywood", "Salpeter"))


# ---------------------------------------------------------------- HCP graph
def test_hcp_graph_is_derived_by_algorithm1():
    for h in N.HCP_HYPOTHESES:
        h.check_u0()
    g = derive_graph(N.HCP_HYPOTHESES, N.HCP_TASKS, N.HCP_TASK_EDGES)
    assert g.edges == [("h_atlas", "h_conn"), ("h_conn", "h_group")]
    assert g.depth == 2 and g.hypothesis_graph_agrees
    assert edge_diff(g.edges, N.DISSERTATION_EDGES)["only_derived"] == []
    assert [len(cascade(g.graph, h)) for h in g.nodes] == [3, 2, 1]
    assert N.CONFIG_SPACE_SIZE == 48


def test_hcp_enumeration_with_reuse():
    from examples.research.chapter4.hcp_experiment import enumerate_configurations

    e = enumerate_configurations()
    assert e["configurations"] == 48
    assert e["model_runs_with_reuse"] == {"h_atlas": 4, "h_conn": 16, "h_group": 48}
    assert e["invariant_violations"] == []


def test_hcp_invariants_reject_bad_outputs():
    import numpy as np

    P = N.parcellation("aal", 400)
    N.ParcellationParams(P=P)
    with pytest.raises(ValueError):
        N.ParcellationParams(P=2 * P)
    with pytest.raises(ValueError):
        N.ConnectivityParams(C=np.array([[1.0, 2.0], [2.0, 1.0]]), method="pearson")
    with pytest.raises(ValueError):
        N.GroupParams(sizes={"men": 1})


# ------------------------------------------------- consistency (conditions)
@pytest.mark.parametrize("domain", ["bgm", "hcp"])
def test_consistency_passes(domain, clean_abox):
    if domain == "bgm":
        from examples.research.chapter4.bgm_experiment import builder

        specs, tasks, tedges, vocab = B.BGM_HYPOTHESES, B.BGM_TASKS, B.BGM_TASK_EDGES, B.BGM_VARIABLES
    else:
        from examples.research.chapter4.hcp_experiment import builder

        specs, tasks, tedges, vocab = N.HCP_HYPOTHESES, N.HCP_TASKS, N.HCP_TASK_EDGES, N.HCP_VARIABLES
    g = derive_graph(specs, tasks, tedges)
    ab = builder(specs, tasks, g.edges)()
    res = structural_check(specs, tasks, tedges, g.edges, vocab)  # Stage A: OWL 2 RL
    assert res["ok"], res
    for key in ("c3", "c4", "c5", "c7", "cond4", "cond5"):
        assert res["details"][key].startswith("passed"), (key, res["details"][key])
    ic = ic_conditions_1_3(ab.ve, ab.task, list(ab.hyp.values()))
    assert ic["condition1"]["ok"] and ic["condition2"]["ok"] and ic["condition3"]["ok"]
    t = time_rl_materialisation(repeats=1)
    assert 0 < t["median_s"] < 60  # measured, not asserted tightly


# ---------------------------------------------------------- rules firing
@needs_java
def test_bgm_rules_firing_table():
    from examples.research.chapter4.bgm_experiment import run

    res = run(time_reasoning=False)
    try:
        assert res["rules_fired_life_cycle"] == [1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 17, 18, 19]
        assert res["rules_fired_fault_only"] == [6, 9, 13, 15]
        assert res["rules_not_fired"] == [11, 14]
        for c in res["change"].values():
            assert c["rule4_match"] and c["rule7_match"] and c["plan_matches_marking"]
        assert res["rivals"]["phase2_after_ranking"]["preferred"] == ["IMF_Kroupa"]
        assert res["consistency"]["structural"]["ok"]
    finally:
        reset_abox()


@needs_java
def test_hcp_rules_firing_table():
    from examples.research.chapter4.hcp_experiment import run

    res = run(time_reasoning=False, execute=False)
    try:
        assert res["rules_fired_life_cycle"] == [1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 17, 18, 19]
        assert res["rules_fired_fault_only"] == [6, 9, 13, 14, 15]
        assert res["rules_not_fired"] == [11]
        for c in res["change"].values():
            assert c["rule4_match"] and c["rule7_match"] and c["plan_matches_marking"]
        assert res["rivals"]["phase2_after_ranking"]["preferred"] == ["h_conn_gp"]
    finally:
        reset_abox()

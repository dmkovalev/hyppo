import networkx as nx

from examples.research.safe_recompute.simulators import (
    Event, SYSTEMS, simulate, simulate_sequence)

G = nx.DiGraph([("a", "b"), ("b", "c")])


def run(system, ev):
    return simulate(system, G, ev)


def test_param_change_output_changes_everyone_cascades():
    ev = Event("param", "a", output_changed=True)
    for s in SYSTEMS:
        assert run(s, ev).recomputed == {"a", "b", "c"}, s


def test_early_cutoff_systems_win_in_favourable_world():
    ev = Event("param", "a", output_changed=False)
    assert run("dvc", ev).recomputed == {"a"}
    assert run("dagster", ev).recomputed == {"a"}
    assert run("bazel", ev).recomputed == {"a"}
    assert run("snakemake", ev).recomputed == {"a", "b", "c"}
    assert run("hyppo_safe", ev).recomputed == {"a", "b", "c"}


def test_eviction_of_source():
    ev = Event("evict", "a", output_changed=False)
    assert run("snakemake", ev).recomputed == {"a", "b", "c"}
    assert run("hyppo_cascade", ev).recomputed == {"a", "b", "c"}
    assert run("hyppo_safe", ev).recomputed == {"a"}
    assert run("dvc", ev).recomputed == {"a"}


def test_status_revision_is_invisible_to_file_systems():
    ev = Event("revise", "a", output_changed=True)
    for s in ("snakemake", "dvc", "dagster", "nextflow", "bazel"):
        r = run(s, ev)
        assert r.recomputed == set() and r.missed == {"b", "c"}, s
    for s in ("hyppo_cascade", "hyppo_safe"):
        r = run(s, ev)
        assert r.recomputed == {"b", "c"} and r.missed == set(), s


def test_mid_chain_cutoff():
    ev = Event("param", "a", output_changed=True)
    oc = lambda n, e: n == "a"  # выход b не меняется
    assert simulate_sequence("dvc", G, [ev], oc)[0].recomputed == {"a", "b"}
    assert simulate_sequence("snakemake", G, [ev], oc)[0].recomputed == {"a", "b", "c"}
    assert simulate_sequence("hyppo_safe", G, [ev], oc)[0].recomputed == {"a", "b", "c"}


def test_evict_with_output_changes():
    ev = Event("evict", "a", output_changed=True)
    assert simulate_sequence("dvc", G, [ev], lambda n, e: True)[0].recomputed == {"a", "b", "c"}
    assert simulate_sequence("dvc", G, [ev], lambda n, e: False)[0].recomputed == {"a"}
    assert simulate_sequence("hyppo_safe", G, [ev], lambda n, e: True)[0].recomputed == {"a"}


def test_sequence_returns_one_result_per_event():
    evs = [Event("param", "b", True), Event("revise", "a", True)]
    rs = simulate_sequence("bazel", G, evs, lambda n, e: True)
    assert [r.recomputed for r in rs] == [{"b", "c"}, set()]
    assert rs[1].missed == {"b", "c"}


import random

import pytest

from examples.research.safe_recompute.simulators import EARLY_CUTOFF

FILE_BASED = ("snakemake", "dvc", "dagster", "nextflow", "nextflow_deep", "bazel")


def test_nextflow_standard_cascades_deep_cuts_off():
    ev = Event("param", "a", output_changed=False)
    assert run("nextflow", ev).recomputed == {"a", "b", "c"}
    assert run("nextflow_deep", ev).recomputed == {"a"}
    assert "nextflow" not in EARLY_CUTOFF and "nextflow_deep" in EARLY_CUTOFF


def test_revise_invisible_incl_nextflow_deep():
    r = run("nextflow_deep", Event("revise", "a", True))
    assert r.recomputed == set() and r.missed == {"b", "c"}


def test_diamond_changed_branch_recomputes_join():
    d = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    ev = Event("param", "a", True)
    oc = lambda n, e: n != "b"
    for s in EARLY_CUTOFF:
        assert simulate_sequence(s, d, [ev], oc)[0].recomputed == {"a", "b", "c", "d"}, s


def test_mid_chain_cutoff_all_systems():
    ev = Event("param", "a", True)
    oc = lambda n, e: n == "a"
    for s in EARLY_CUTOFF:
        assert simulate_sequence(s, G, [ev], oc)[0].recomputed == {"a", "b"}, s
    assert simulate_sequence("hyppo_cascade", G, [ev], oc)[0].recomputed == {"a", "b", "c"}


def test_empty_events():
    assert simulate_sequence("dvc", G, [], lambda n, e: True) == []


def test_value_errors():
    with pytest.raises(ValueError):
        simulate("nope", G, Event("param", "a", True))
    with pytest.raises(ValueError):
        simulate("dvc", G, Event("bogus", "a", True))
    with pytest.raises(ValueError, match="not in the graph"):
        simulate("dvc", G, Event("param", "zzz", True))


def test_simulate_equals_sequence_on_random_dags():
    rng = random.Random(0)
    for _ in range(10):
        n = 8
        g = nx.DiGraph()
        g.add_nodes_from(str(i) for i in range(n))
        g.add_edges_from((str(i), str(j)) for i in range(n) for j in range(i + 1, n)
                         if rng.random() < 0.3)
        for kind in ("param", "evict", "revise"):
            for ch in (True, False):
                ev = Event(kind, str(rng.randrange(n)), ch)
                for s in SYSTEMS:
                    assert simulate(s, g, ev) == simulate_sequence(
                        s, g, [ev], lambda x, e: e.output_changed)[0]


def test_evict_sequence_cascade_systems():
    ev = Event("evict", "a", False)
    for s in ("snakemake", "hyppo_cascade"):
        assert simulate_sequence(s, G, [ev], lambda n, e: False)[0].recomputed == {"a", "b", "c"}

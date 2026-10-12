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

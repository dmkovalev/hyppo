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


# ---------- последовательная модель с хранилищем (Э6) ----------
from examples.research.safe_recompute.simulators import (  # noqa: E402
    NATIVE_EVICT, Store, native_evict, run_sequence)

# a -> b -> c, a -> d; сущность e читают b и d, объявлена только пара (e, b)
G2 = nx.DiGraph([("a", "b"), ("b", "c"), ("a", "d")])
ST = Store(consumers={"e": frozenset({"b", "d"}), "k": frozenset({"a"})},
           declared=frozenset({("e", "b")}), params=frozenset({"k"}))


def test_declared_revision_detected_with_tool_closure():
    ev = Event("revise", "e", True)
    delta = [{"b", "c", "d"}]
    for s in ("snakemake", "dagster_eager"):
        r = run_sequence(s, G2, [ev], delta, ST)[0]
        assert r.recomputed == {"b", "c"} and r.missed == {"d"}, s
    for s in ("dvc", "dagster"):
        r = run_sequence(s, G2, [ev], delta, ST)[0]
        assert r.recomputed == {"b", "c"} and r.missed == {"d"}, s
    # c не меняется -> обрыв у cutoff-систем, но c всё равно пересчитан (b изменился)
    r = run_sequence("dvc", G2, [ev], [{"b", "d"}], ST)[0]
    assert r.recomputed == {"b", "c"} and r.wrong == {"d"}


def test_undeclared_revision_missed_hyppo_sees_all_consumers():
    st = Store(consumers=ST.consumers, declared=frozenset(), params=ST.params)
    ev = Event("revise", "e", True)
    for s in ("snakemake", "dvc", "dagster"):
        r = run_sequence(s, G2, [ev], None, st)[0]
        assert r.recomputed == set() and r.missed == {"b", "c", "d"}, s
    for s in ("hyppo_safe", "hyppo_cascade"):
        r = run_sequence(s, G2, [ev], None, st)[0]
        assert r.recomputed == {"b", "c", "d"} and r.missed == set(), s


def test_param_entity_always_visible():
    ev = Event("param", "k", False)
    assert run_sequence("dvc", G2, [ev], None, ST)[0].recomputed == {"a"}
    assert run_sequence("snakemake", G2, [ev], None, ST)[0].recomputed == {"a", "b", "c", "d"}


def test_dagster_eager_has_no_cutoff():
    ev = Event("param", "a", False)
    assert simulate("dagster", G, ev).recomputed == {"a"}
    assert simulate("dagster_eager", G, ev).recomputed == {"a", "b", "c"}
    assert run_sequence("dagster_eager", G, [ev])[0].recomputed == {"a", "b", "c"}


def test_state_persists_and_stale_node_breaks_cutoff():
    """Пропущенная ревизия оставляет b, c неверными; последующая правка параметра
    b без изменения выхода (как ridge "3" после tau в Э6): DVC пересчитывает b,
    его выход изменился относительно записанного (b был неверен) -> пересчёт c,
    всё снова верно. Правка a без изменения выхода b не трогает: b, c неверны."""
    st = Store(consumers={"e": frozenset({"b"})}, declared=frozenset())
    evs = [Event("revise", "e", True), Event("param", "b", False)]
    r = run_sequence("dvc", G, evs, None, st)
    assert r[0].missed == {"b", "c"} and r[0].wrong == {"b", "c"}
    assert r[1].recomputed == {"b", "c"} and r[1].wrong == set() and r[1].missed == set()
    # без пропуска та же правка обрывается на b
    assert run_sequence("dvc", G, evs[1:], None, st)[0].recomputed == {"b"}
    r = run_sequence("dvc", G, [evs[0], Event("param", "a", False)], None, st)
    assert r[1].recomputed == {"a"} and r[1].wrong == {"b", "c"} and r[1].missed == set()


def test_recomputed_child_of_wrong_parent_is_wrong():
    st = Store(consumers={"e": frozenset({"b", "c"})}, declared=frozenset({("e", "c")}))
    r = run_sequence("dvc", G, [Event("revise", "e", True)], None, st)[0]
    assert r.recomputed == {"c"} and r.wrong == {"b", "c"}


def test_hyppo_safe_evict_only_node_and_forced_cutoff():
    ev = Event("evict", "a", False)
    assert run_sequence("hyppo_safe", G, [ev])[0].recomputed == {"a"}
    assert run_sequence("hyppo_cascade", G, [ev])[0].recomputed == {"a", "b", "c"}
    assert run_sequence("dvc", G, [ev])[0].recomputed == {"a"}
    assert run_sequence("snakemake", G, [ev])[0].recomputed == {"a", "b", "c"}


def test_run_sequence_legacy_revise_matches_simulate():
    for s in SYSTEMS:
        for n in G2:
            ev = Event("revise", n, True)
            a, b = simulate(s, G2, ev), run_sequence(s, G2, [ev])[0]
            assert a.recomputed == b.recomputed and a.missed == b.missed, (s, n)


def test_native_evict_variants():
    assert native_evict("dvc", "b", G).absent == set() and native_evict("dvc", "b", G).recomputed == set()
    for s in ("dagster", "dagster_eager"):
        r = native_evict(s, "c", G)
        assert r.recomputed == set() and r.absent == {"c"}, s
    # Snakemake с целью all: промежуточный не восстанавливается, конечный -- да
    assert native_evict("snakemake", "b", G).absent == {"b"}
    assert native_evict("snakemake", "c", G).recomputed == {"c"}
    for s in SYSTEMS:
        if s not in NATIVE_EVICT:
            assert native_evict(s, "b", G) is None, s


def test_run_sequence_errors():
    with pytest.raises(ValueError):
        run_sequence("nope", G, [])
    with pytest.raises(ValueError, match="unknown store entity"):
        run_sequence("dvc", G2, [Event("revise", "zzz", True)], None, ST)
    with pytest.raises(ValueError, match="length"):
        run_sequence("dvc", G, [Event("param", "a", True)], [])

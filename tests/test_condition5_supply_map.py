"""Condition 5 as a supply map (§2.4): every producer of an input of a hypothesis
lies in a strict ancestor task. Covers the gap of the edge-only check: supplies
between parallel tasks or inside one task are never edges of L, yet they break
the execution order and allow cross-hypothesis cycles (lemma lem:acyclic)."""

from hyppo.ontology.consistency import Status, check_consistency


def _check(tasks, wf_edges, sigs, lattice=None):
    return check_consistency(
        None, None, lattice or {}, run_hermit=False,
        task_hypotheses=tasks, workflow_edges=wf_edges, artefacts=sigs,
    )


SIG_OK = {0: {"in": set(), "out": {"x"}}, 1: {"in": {"x"}, "out": {"y"}}}


def test_supply_along_the_workflow_passes():
    res = _check({"t1": [0], "t2": [1]}, [("t1", "t2")], SIG_OK)
    assert res.ok and res.details["cond5"].startswith("passed")


def test_supply_between_parallel_tasks_is_rejected():
    res = _check({"t0": [], "t1": [0], "t2": [1]}, [("t0", "t1"), ("t0", "t2")], SIG_OK)
    assert not res.ok and res.status == Status.COND5_VIOLATED
    assert res.details["cond5"]["hypothesis"] == 1 and res.details["cond5"]["dependency"] == 0


def test_supply_inside_one_task_is_rejected():
    res = _check({"t1": [0, 1]}, [], SIG_OK)
    assert not res.ok and res.status == Status.COND5_VIOLATED


def test_counter_supply_order_conflict_is_rejected():
    # 0 (task t1) needs z which 1 produces only later (task t2)
    sigs = {0: {"in": {"z"}, "out": {"x"}}, 1: {"in": {"x"}, "out": {"z"}}}
    res = _check({"t1": [0], "t2": [1]}, [("t1", "t2")], sigs)
    assert not res.ok and res.status == Status.COND5_VIOLATED
    assert res.details["cond5"]["task"] == "t1"


def test_projectile_h3bad_is_rejected():
    """Running example: h3^bad (task t4) needs theta*, produced by h6 in t7."""
    tasks = {"t3": [5], "t4": [6], "t7": [9]}
    wf = [("t3", "t4"), ("t4", "t7")]
    sigs = {5: {"in": set(), "out": {"traj"}},
            6: {"in": {"traj", "theta_star"}, "out": {"R"}},
            9: {"in": {"R", "theta"}, "out": {"theta_star"}}}
    res = _check(tasks, wf, sigs)
    assert not res.ok and res.status == Status.COND5_VIOLATED
    assert res.details["cond5"]["dependency_task"] == "t7"


def test_mutual_supply_reports_merge_candidate():
    """A data cycle between two hypotheses of parallel tasks is reported as a
    strongly connected component to merge into one composite hypothesis."""
    sigs = {0: {"in": {"y", "p"}, "out": {"x"}}, 1: {"in": {"x"}, "out": {"y"}}}
    res = _check({"t0": [], "t1": [0], "t2": [1]}, [("t0", "t1"), ("t0", "t2")], sigs)
    assert not res.ok and res.status == Status.COND5_VIOLATED
    cyc = res.details["cond5"]["supply_cycles"]
    assert cyc == [{"hypotheses": [0, 1],
                    "merged_signature": {"in": ["p"], "out": ["x", "y"]}}]

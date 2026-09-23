import numpy as np

from examples.research.baseline_filecache import config, dag, experiment, hyppo_side, metastore
from examples.research.baseline_filecache.config import SMALL_GRID, GRID


def test_pairs_and_levels():
    assert len(config.PAIRS) == 6
    assert config.level_size(0.0) == 0
    assert config.level_size(0.25) == 2
    assert config.level_size(0.5) == 3
    assert config.level_size(0.75) == 5
    assert config.level_size(1.0) == 6


def test_declaration_subsets_deterministic_and_sized():
    a = config.declaration_subsets(seed=7)
    b = config.declaration_subsets(seed=7)
    assert a == b
    sizes = {}
    for p, subset in a:
        sizes.setdefault(p, []).append(len(subset))
        assert set(subset) <= set(config.PAIRS)
    assert sizes[0.0] == [0]
    assert sizes[1.0] == [6]
    for p in (0.25, 0.5, 0.75):
        assert len(sizes[p]) == 5
        assert set(sizes[p]) == {config.level_size(p)}
    # подмножества одного уровня различны
    for p in (0.25, 0.5, 0.75):
        subs = [frozenset(s) for q, s in a if q == p]
        assert len(set(subs)) == 5


def test_jobs_count_full_grid():
    jobs = dag.jobs(GRID)
    assert len(jobs) == 2 + 8 + 16 + 128


def test_parents_small_grid():
    par = dag.parents(SMALL_GRID)
    assert par["hyb/uto_1.npy"] == {"p1/uto.npy"}
    assert par["opr/pos_1_2_2.json"] == {"hyb/pos_1.npy", "wct/2_2.npy"}
    assert par["p1/uto.npy"] == set()


def test_wrong_set_propagates_through_files():
    g = SMALL_GRID
    S = {"hyb/uto_1.npy", "opr/uto_1_2_2.json"}
    # hyb не пересчитан, opr пересчитан -> opr читает устаревший hyb -> оба неверны
    E = {"opr/uto_1_2_2.json"}
    assert dag.wrong_set(g, S, E) == {"hyb/uto_1.npy", "opr/uto_1_2_2.json"}
    # всё пересчитано -> ничего неверного
    assert dag.wrong_set(g, S, S) == set()


def _descendant_jobs(grid, roots_rule_filter):
    return {j for j in dag.jobs(grid) if roots_rule_filter(j)}


def test_plan_matches_expected_cascade_small_grid():
    g = SMALL_GRID
    model = hyppo_side.build(g)
    hyb = {j for j in dag.jobs(g) if j.startswith("hyb/")}
    opr = {j for j in dag.jobs(g) if j.startswith("opr/")}
    wct = {j for j in dag.jobs(g) if j.startswith("wct/")}
    assert hyppo_side.plan_jobs(model, "gains") == hyb | opr
    assert hyppo_side.plan_jobs(model, "tau") == hyb | opr
    assert hyppo_side.plan_jobs(model, "well_status") == hyb | opr
    assert hyppo_side.plan_jobs(model, "corey_ref") == wct | opr
    assert hyppo_side.plan_jobs(model, "ridge") == hyb | opr


def test_no_manual_dependency_records():
    model = hyppo_side.build(SMALL_GRID)
    assert model.extra_declarations == 0


def test_metastore_roundtrip_and_revise(tmp_path):
    db = tmp_path / "meta.sqlite"
    g = np.array([[0.2, 0.9], [0.4, 0.1]])
    metastore.write(db, "gains", "pos", g)
    metastore.write(db, "gains", "uto", g * 0.5)
    metastore.write(db, "tau", "pos", np.ones((2, 2)))
    metastore.write(db, "tau", "uto", np.ones((2, 2)))
    metastore.write(db, "corey_ref", "*", np.array([0.25, 0.25]))
    metastore.write(db, "well_status", "*", np.ones(2))
    assert metastore.version(db, "gains", "pos") == "v1"
    metastore.revise(db, "gains")
    v, g2 = metastore.read(db, "gains", "pos")
    assert v == "v2" and g2[0, 1] == 0.9 * 1.5 and g2[1, 0] == 0.4
    metastore.revise(db, "well_status")
    _, st = metastore.read(db, "well_status", "*")
    assert st.tolist() == [0.0, 1.0]  # продюсер с наибольшим суммарным gain по строке; строки = продюсеры
    metastore.revise(db, "corey_ref")
    _, cr = metastore.read(db, "corey_ref", "*")
    assert cr.tolist() == [0.25, 0.30]
    metastore.revise(db, "tau")
    _, t = metastore.read(db, "tau", "uto")
    assert np.all(t == 2.0)


def test_summarize_and_checks_with_stub():
    g = SMALL_GRID
    hyb = {j for j in dag.jobs(g) if j.startswith("hyb/")}
    opr = {j for j in dag.jobs(g) if j.startswith("opr/")}
    wct = {j for j in dag.jobs(g) if j.startswith("wct/")}
    truth = {"gains": hyb | opr, "tau": hyb | opr, "well_status": hyb | opr,
             "corey_ref": wct | opr, "ridge": hyb | opr}
    model = hyppo_side.build(g)

    def stub_e(declared, revision):
        # файловый кэш: пересчитывает прямых потребителей объявленных пар + файловых потомков
        if revision == "ridge":
            return hyb | opr
        direct = {"fit_liq": hyb, "fit_wct": wct, "fit_opr": opr}
        hit = set()
        for ent, rule in declared:
            if ent == revision:
                hit |= direct[rule]
        if hit & hyb:
            hit |= opr
        if hit & wct:
            hit |= opr
        return hit

    from examples.research.baseline_filecache.config import PAIRS
    trials = experiment.run_trials(g, truth, model, stub_e, subsets=[(0.0, ()), (1.0, tuple(PAIRS))])
    experiment.self_checks(trials)                 # не должно падать
    summ = experiment.summarize(trials)
    assert summ[1.0]["snakemake_missed_mean"] == 0
    assert summ[0.0]["snakemake_missed_mean"] > 0
    assert all(v["hyppo_missed_max"] == 0 for v in summ.values())

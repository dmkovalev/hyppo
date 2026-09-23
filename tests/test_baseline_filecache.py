from examples.research.baseline_filecache import config, dag
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

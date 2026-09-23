from examples.research.baseline_filecache import config


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

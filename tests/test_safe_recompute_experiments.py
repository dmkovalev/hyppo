from examples.research.safe_recompute import e1_eviction


def test_e1_smoke(tmp_path):
    out = e1_eviction.main(steps=4, seeds=1, out=tmp_path / "e1.json", graphs_limit=3)
    for tag, row in out.items():
        assert row["median_ratio_cascade_to_safe"] >= 1.0
    assert "lru:0.75" in out and "random:0.75" in out

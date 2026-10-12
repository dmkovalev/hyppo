from examples.research.safe_recompute import e5_planning_time


def test_e5_smoke(tmp_path):
    out = e5_planning_time.main(sizes=[50, 100, 200], reps=2, out=tmp_path / "e5.json")
    assert out["chained"]["slope"] < 1.4
    assert out["direct"]["slope"] > out["chained"]["slope"]

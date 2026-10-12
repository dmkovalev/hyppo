from hyppo.planner.mincost import is_feasible
from examples.research.safe_recompute import e2_mincost


def test_e2_ordering_of_plans(tmp_path):
    out = e2_mincost.main(seeds=2, out=tmp_path / "e2.json", graphs_limit=4)
    assert out["rows"]
    for row in out["rows"]:
        assert row["cost_mincost"] <= row["cost_N_read_rest"] + 1e-9
        assert row["cost_mincost"] <= row["cost_cascade_read_rest"] + 1e-9
        # N+чтение может превысить полный пересчёт при дорогом чтении (rho=1)
        assert row["cost_mincost"] <= row["cost_full"] + 1e-9
        assert row["feasible_all"]

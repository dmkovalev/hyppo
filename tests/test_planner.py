"""Tests for VirtualExperimentPlanner."""

import networkx as nx
import pytest

from hyppo.planner._base import ExecutionPlan, Planner, build_optimal_plan


# ---------------------------------------------------------------------------
# Helpers: lightweight fakes for lattice, configuration, and Database
# ---------------------------------------------------------------------------

class FakeModel:
    def __init__(self, name: str):
        self.name = name


class FakeHypothesis:
    """Minimal hypothesis stub with name and model list."""

    def __init__(self, name: str, models: list[FakeModel] | None = None):
        self.name = name
        self.is_implemented_by_model = models or [FakeModel(f"model_{name}")]


class FakeConfiguration:
    """Single-parameter configuration stub."""

    def __init__(self, tag: str = "default"):
        self.parameters = [tag]


class FakeLattice:
    """Wraps a networkx DiGraph as a HypothesisLattice-like object."""

    def __init__(self, graph: nx.DiGraph):
        self.lattice = graph
        self.hypotheses = list(graph.nodes())


class _FakePickled:
    """Minimal stand-in for storage._base.Pickled (avoids cloudpickle dep)."""

    def __init__(self, obj):
        self.obj = obj


class FakeDatabase:
    """In-memory fake of the old pickle-based Database for planner tests."""

    def __init__(self):
        self._results: dict[str, object] = {}
        self._lattices: list = []

    # --- results storage ---
    def load(self, key: str, storage: str = "") -> object | None:
        return self._results.get(f"{storage}/{key}")

    def save(self, obj, key: str, storage: str = "", **kw) -> None:
        self._results[f"{storage}/{key}"] = _FakePickled(obj)

    def load_all(self, storage: str = "") -> list:
        return self._lattices

    # --- convenience for tests ---
    def put_result(self, hypothesis: FakeHypothesis, config: FakeConfiguration, obj: dict) -> None:
        """Store a fake cached result for the given hypothesis + config."""
        for cm in (config.parameters or [config]):
            for model in hypothesis.is_implemented_by_model:
                key = f"{hypothesis.name}__{model.name}__{str(cm)}"
                self._results[f"results/{key}"] = _FakePickled(obj)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plan_all_new():
    """All hypotheses without cache -> all in needs_execution (P_ne)."""
    h1 = FakeHypothesis("h1")
    h2 = FakeHypothesis("h2")
    g = nx.DiGraph()
    g.add_node(h1)
    g.add_node(h2)
    g.add_edge(h1, h2)

    lattice = FakeLattice(g)
    db = FakeDatabase()
    config = FakeConfiguration()

    plan = build_optimal_plan(config, lattice, db)

    assert h1 in plan.needs_execution
    assert h2 in plan.needs_execution
    assert len(plan.cached) == 0


def test_plan_with_cache():
    """Hypothesis with cached result -> in cached (P_e)."""
    h1 = FakeHypothesis("h1")
    h2 = FakeHypothesis("h2")
    g = nx.DiGraph()
    g.add_node(h1)
    g.add_node(h2)
    # h1 -> h2 dependency
    g.add_edge(h1, h2)

    lattice = FakeLattice(g)
    db = FakeDatabase()
    config = FakeConfiguration()

    # Cache results for both hypotheses
    db.put_result(h1, config, {"r2": 0.9})
    db.put_result(h2, config, {"r2": 0.85})

    plan = build_optimal_plan(config, lattice, db)

    assert h1 in plan.cached
    assert h2 in plan.cached
    assert len(plan.needs_execution) == 0


def test_plan_prune_low_r2():
    """R2 < threshold -> hypothesis and dependents excluded from plan."""
    h1 = FakeHypothesis("h1")
    h2 = FakeHypothesis("h2")
    h3 = FakeHypothesis("h3")
    g = nx.DiGraph()
    g.add_nodes_from([h1, h2, h3])
    g.add_edge(h1, h2)
    g.add_edge(h2, h3)

    lattice = FakeLattice(g)
    db = FakeDatabase()
    config = FakeConfiguration()

    # h1 has good R2, h2 has low R2 -> h2 and h3 should be pruned
    db.put_result(h1, config, {"r2": 0.9})
    db.put_result(h2, config, {"r2": 0.5})
    db.put_result(h3, config, {"r2": 0.95})

    plan = build_optimal_plan(config, lattice, db, r2_threshold=0.7)

    assert h1 in plan.cached
    # h2 pruned (R2=0.5 < 0.7), h3 pruned as dependent
    assert h2 not in plan.cached
    assert h2 not in plan.needs_execution
    assert h3 not in plan.cached
    assert h3 not in plan.needs_execution


def test_planner_class_delegates():
    """Planner class wraps build_optimal_plan correctly."""
    h1 = FakeHypothesis("h1")
    g = nx.DiGraph()
    g.add_node(h1)

    lattice = FakeLattice(g)
    db = FakeDatabase()
    config = FakeConfiguration()

    planner = Planner(db=db, r2_threshold=0.7)
    plan = planner.plan(config, lattice)

    # No cache -> h1 in needs_execution
    assert h1 in plan.needs_execution


def test_plan_empty_lattice():
    """Empty lattice produces empty plan."""
    g = nx.DiGraph()
    lattice = FakeLattice(g)
    db = FakeDatabase()
    config = FakeConfiguration()

    plan = build_optimal_plan(config, lattice, db)

    assert len(plan.needs_execution) == 0
    assert len(plan.cached) == 0


def test_shared_cache_planner_sees_runner_saves():
    """SharedCache: результат, записанный API раннера (save_result), планировщик
    видит как кэш (P_e) — planner и runner делят ОДИН SQLite-кэш."""
    from hyppo.metadata_repository import SharedCache

    h1, h2 = FakeHypothesis("h1"), FakeHypothesis("h2")
    g = nx.DiGraph(); g.add_edge(h1, h2)
    lattice = FakeLattice(g)
    config = FakeConfiguration()
    cache = SharedCache(":memory:")

    # холодный план: кэш пуст → обе в needs_execution
    p0 = build_optimal_plan(config, lattice, cache)
    assert h1 in p0.needs_execution and h2 in p0.needs_execution and not p0.cached

    # runner «вычислил» и записал результаты (тот же config, что видит planner)
    cfg_key = {"parameters": ["default"]}
    cache.save_result("h1", cfg_key, {"r2": 0.9})
    cache.save_result("h2", cfg_key, {"r2": 0.88})

    # тёплый план: планировщик видит записи раннера как кэш
    p1 = build_optimal_plan(config, lattice, cache)
    assert h1 in p1.cached and h2 in p1.cached and not p1.needs_execution

    # низкий R² из кэша раннера → отсечение ветви планировщиком
    cache.save_result("h2", cfg_key, {"r2": 0.4})
    p2 = build_optimal_plan(config, lattice, cache, r2_threshold=0.7)
    assert h1 in p2.cached and h2 not in p2.cached and h2 not in p2.needs_execution


def test_per_hypothesis_config_invalidates_single_branch():
    """По-гипотезные конфигурации: смена конфигурации ОДНОЙ гипотезы даёт промах
    только по ней и пересчёт её замыкания вниз — одним вызовом build_optimal_plan."""
    from hyppo.metadata_repository import SharedCache

    h1, h2, h3 = FakeHypothesis("h1"), FakeHypothesis("h2"), FakeHypothesis("h3")
    g = nx.DiGraph(); g.add_edge(h1, h2); g.add_edge(h2, h3)
    lattice = FakeLattice(g)
    cache = SharedCache(":memory:")

    base = {"v": 1}
    per = {"h1": base, "h2": base, "h3": base}
    for name in ("h1", "h2", "h3"):
        cache.save_result(name, per[name], {"r2": 0.9})

    # всё в кэше → пусто на пересчёт
    p0 = build_optimal_plan(None, lattice, cache, per_hypothesis_configs=per)
    assert {h1, h2, h3} == p0.cached and not p0.needs_execution

    # меняем конфигурацию ТОЛЬКО h2 → промах по h2, пересчёт h2 и потомка h3
    per2 = {"h1": base, "h2": {"v": 2}, "h3": base}
    p1 = build_optimal_plan(None, lattice, cache, per_hypothesis_configs=per2)
    assert p1.needs_execution == {h2, h3}
    assert p1.cached == {h1}


def test_plan_rejects_cyclic_graph():
    """Cyclic hypothesis graph must raise ValueError (Algorithm 4 precondition)."""
    h1 = FakeHypothesis("h1")
    h2 = FakeHypothesis("h2")
    h3 = FakeHypothesis("h3")
    g = nx.DiGraph()
    g.add_nodes_from([h1, h2, h3])
    g.add_edge(h1, h2)
    g.add_edge(h2, h3)
    g.add_edge(h3, h1)  # cycle: h1 -> h2 -> h3 -> h1

    lattice = FakeLattice(g)
    db = FakeDatabase()
    config = FakeConfiguration()

    with pytest.raises(ValueError, match="cycle"):
        build_optimal_plan(config, lattice, db)

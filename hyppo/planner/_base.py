"""Планировщик виртуальных экспериментов.

Реализация Алгоритма 4 из диссертации: «Построение плана виртуального эксперимента».
Алгоритм принимает конфигурацию эксперимента и решетку гипотез, определяет,
какие гипотезы требуют пересчета моделей, а какие могут быть взяты из кеша
репозитория метаинформации.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Set

import networkx as nx

if TYPE_CHECKING:
    from hyppo.lattice_constructor._base import HypothesisLattice
    from hyppo.storage._base import Database

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Результат планирования виртуального эксперимента.

    Attributes:
        needs_execution: Pne -- множество гипотез, модели которых требуют пересчета.
        cached: Pe -- множество гипотез, для которых можно использовать
            ранее вычисленные результаты.
    """

    needs_execution: Set = field(default_factory=set)
    cached: Set = field(default_factory=set)


def _find_nearest_lattice(
    lattice: "HypothesisLattice",
    db: "Database",
) -> Optional["HypothesisLattice"]:
    """Поиск ближайшей ранее вычисленной решетки в репозитории метаинформации.

    Ближайшей считается решетка, имеющая наибольшее пересечение
    множества гипотез с текущей решеткой.

    Args:
        lattice: Текущая решетка гипотез.
        db: Экземпляр базы данных (репозиторий метаинформации).

    Returns:
        Ближайшая решетка из репозитория или None, если репозиторий пуст.
    """
    stored_objects = db.load_all(storage="lattices")
    if not stored_objects:
        return None

    current_hypotheses = set(lattice.hypotheses)
    best_lattice = None
    best_overlap = -1

    for pickled in stored_objects:
        if pickled is None:
            continue
        candidate = pickled.obj
        candidate_hypotheses = set(candidate.hypotheses)
        overlap = len(current_hypotheses & candidate_hypotheses)
        if overlap > best_overlap:
            best_overlap = overlap
            best_lattice = candidate

    return best_lattice


def _cfg_dict(configuration) -> dict:
    """Привести конфигурацию к dict для ключа общего кэша (config_hash).
    Согласовано с runner, который пишет результаты по (hypothesis_id, config)."""
    if isinstance(configuration, dict):
        return configuration
    params = getattr(configuration, "parameters", None)
    if params is not None:
        return {"parameters": [str(p) for p in params]}
    return {"repr": str(configuration)}


def _config_for(hypothesis, configuration, per_hypothesis_configs) -> object:
    """Разрешить конфигурацию для конкретной гипотезы.

    Если задан ``per_hypothesis_configs`` (отображение id гипотезы → конфигурация),
    возвращается индивидуальная конфигурация гипотезы (при отсутствии — общий
    ``configuration`` как значение по умолчанию). Иначе — общий ``configuration``
    для всех гипотез (прежнее поведение)."""
    if per_hypothesis_configs is None:
        return configuration
    name = getattr(hypothesis, "name", str(hypothesis))
    return per_hypothesis_configs.get(name, configuration)


def _has_cached_result(
    hypothesis,
    configuration,
    db: "Database",
) -> bool:
    """Проверка наличия кешированного результата для гипотезы при данной конфигурации.

    Для каждого набора параметров cm из конфигурации C и для каждой модели m,
    реализующей гипотезу h, проверяется, существует ли вычисленный результат
    h -> m(cm) в репозитории метаинформации.

    Args:
        hypothesis: Проверяемая гипотеза.
        configuration: Конфигурация эксперимента (набор параметров).
        db: Экземпляр базы данных.

    Returns:
        True, если для ВСЕХ комбинаций параметров и моделей существует
        кешированный результат; False -- если хотя бы для одной комбинации
        результат отсутствует.
    """
    # Быстрый путь: репозиторий с ключом (гипотеза, конфигурация) — ОБЩИЙ кэш
    # planner↔runner (SharedCache / MetadataRepository). Так планировщик видит
    # результаты, записанные раннером, в одном SQLite. Legacy Database (только
    # load/save) идёт по старому пути ниже.
    if hasattr(db, "has_result") and hasattr(db, "load_result"):
        return db.has_result(
            getattr(hypothesis, "name", str(hypothesis)), _cfg_dict(configuration)
        )

    # Получаем модели, реализующие гипотезу.
    # В онтологии is_implemented_by_model -- ObjectProperty, возвращающее список
    # связанных объектов Model. Owlready2 делает его вызываемым для получения
    # инстансов, поэтому пробуем вызвать как метод; если атрибут уже является
    # списком, используем его напрямую.
    models_attr = getattr(hypothesis, "is_implemented_by_model", None)
    if callable(models_attr):
        models = models_attr()
    else:
        models = models_attr
    if not models:
        return False

    # Получаем наборы параметров из конфигурации.
    # Configuration -- самостоятельная сущность; параметры хранятся как список
    # внутри неё (например, configuration.parameters).  Если такого атрибута нет,
    # используем саму конфигурацию как единственный набор параметров.
    param_sets = getattr(configuration, "parameters", None)
    if not param_sets:
        param_sets = [configuration]

    for cm in param_sets:
        for model in models:
            # Формируем ключ кеша: идентификатор гипотезы + модели + параметров
            cache_key = (
                f"{getattr(hypothesis, 'name', str(hypothesis))}"
                f"__{getattr(model, 'name', str(model))}"
                f"__{str(cm)}"
            )
            result = db.load(cache_key, storage="results")
            if result is None:
                return False

    return True


def _get_cached_r2(
    hypothesis,
    configuration,
    db: "Database",
) -> Optional[float]:
    """Извлечь R² из кешированного результата для гипотезы (если есть).

    Returns:
        R² (float) или None, если результат не найден или R² не записан.
    """
    # Быстрый путь: общий кэш planner↔runner (см. _has_cached_result).
    if hasattr(db, "has_result") and hasattr(db, "load_result"):
        rec = db.load_result(
            getattr(hypothesis, "name", str(hypothesis)), _cfg_dict(configuration)
        )
        if rec:
            return rec.get("metrics", {}).get("r2")
        return None

    models_attr = getattr(hypothesis, "is_implemented_by_model", None)
    if callable(models_attr):
        models = models_attr()
    else:
        models = models_attr
    if not models:
        return None

    param_sets = getattr(configuration, "parameters", None)
    if not param_sets:
        param_sets = [configuration]

    # Возвращаем R² первого найденного результата
    for cm in param_sets:
        for model in models:
            cache_key = (
                f"{getattr(hypothesis, 'name', str(hypothesis))}"
                f"__{getattr(model, 'name', str(model))}"
                f"__{str(cm)}"
            )
            result = db.load(cache_key, storage="results")
            if result is not None and hasattr(result, "obj"):
                obj = result.obj
                if isinstance(obj, dict) and "r2" in obj:
                    return obj["r2"]
    return None


def build_optimal_plan(
    configuration,
    lattice: "HypothesisLattice",
    db: "Database",
    r2_threshold: float = 0.7,
    *,
    per_hypothesis_configs: "Mapping | None" = None,
) -> ExecutionPlan:
    """Построение оптимального плана виртуального эксперимента (Алгоритм 4).

    Алгоритм обходит решетку гипотез сверху вниз (начиная с вершин
    без входящих ребер). Для каждой гипотезы проверяется наличие
    кешированного результата в репозитории метаинформации:

    - Если результата нет -- гипотеза и все зависимые от неё гипотезы
      добавляются в множество Pne (требуют пересчета).
    - Если результат есть -- гипотеза добавляется в множество Pe
      (повторное вычисление не требуется).
    - Отсечение по R² (Раздел 3.1.6.2): если R² кешированного результата
      гипотезы ниже порога r2_threshold, гипотеза и все зависимые от неё
      исключаются из плана (не попадают ни в Pne, ни в Pe).

    Это позволяет значительно сократить время исполнения виртуального
    эксперимента за счет повторного использования ранее вычисленных
    результатов и отсечения низкокачественных ветвей.

    Args:
        configuration: Конфигурация эксперимента (C) -- набор параметров.
        lattice: Решетка гипотез (L) -- объект HypothesisLattice с графом
            зависимостей между гипотезами.
        db: Экземпляр базы данных (репозиторий метаинформации) для поиска
            ранее вычисленных результатов.
        r2_threshold: Минимальный порог R² для включения гипотезы в план.
            Гипотезы с R² < порога и все их потомки исключаются (Раздел 3.1.6.2).
            По умолчанию 0.7.
        per_hypothesis_configs: Необязательное отображение id гипотезы →
            конфигурация. Если задано, кэш каждой гипотезы проверяется по её
            собственной конфигурации (при отсутствии — по общей ``configuration``).
            Это позволяет инвалидировать кэш ОТДЕЛЬНОЙ гипотезы: смена её
            конфигурации даёт промах только по ней и пересчёт её замыкания вниз.

    Returns:
        ExecutionPlan с двумя множествами:
            - needs_execution (Pne): гипотезы, требующие пересчета моделей.
            - cached (Pe): гипотезы с доступными кешированными результатами.
    """
    plan = ExecutionPlan()

    # Шаг 1: Поиск ближайшей ранее вычисленной решетки в репозитории
    nearest_lattice = _find_nearest_lattice(lattice, db)

    # Строим рабочий граф на основе ТЕКУЩЕЙ решетки (чтобы гарантировать
    # полноту: Pne U Pe = V(L)).  Ближайшая решетка используется только
    # для обогащения кеша -- гипотезы, присутствующие в nearest_lattice,
    # с большей вероятностью имеют кешированные результаты, но сам обход
    # всегда идёт по текущей решетке.
    graph: nx.DiGraph = lattice.lattice

    # Множество гипотез из ближайшей решетки (для возможной дополнительной
    # логики; на данный момент кеш-проверка через _has_cached_result уже
    # обращается к репозиторию напрямую).
    _nearest_hypotheses: Set = set()
    if nearest_lattice is not None:
        _nearest_hypotheses = set(nearest_lattice.hypotheses)

    # Шаг 2: Инициализация множеств
    # Pne -- гипотезы, требующие пересчета
    # Pe  -- гипотезы, не требующие пересчета
    # F   -- обработанные вершины
    # pruned -- гипотезы, отсеченные по R² (Раздел 3.1.6.2)
    pne: Set = set()
    pe: Set = set()
    finished: Set = set()
    pruned: Set = set()

    # Шаг 3: Множество необработанных вершин V(L)
    remaining = set(graph.nodes())

    # Пустая решетка -- пустой план
    if not remaining:
        return plan

    # Шаг 4: Обход решетки сверху вниз
    # Начинаем с гипотез, не имеющих входящих ребер (корневые гипотезы)
    while remaining:
        # Выбираем вершины без входящих ребер среди необработанных
        # (топологический порядок: сначала обрабатываем «верхние» гипотезы)
        roots = [
            h
            for h in remaining
            if all(pred in finished for pred in graph.predecessors(h))
        ]

        if not roots:
            raise ValueError(
                "Hypothesis graph contains a cycle — "
                "precondition of Algorithm 4 violated "
                "(Theorem 1 requires a DAG). "
                "Run check_consistency() before build_optimal_plan()."
            )

        for h in roots:
            if h in finished:
                continue

            # Отсечение по R² (Раздел 3.1.6.2): если предок был отсечен,
            # текущая гипотеза тоже отсекается.
            if h in pruned:
                finished.add(h)
                continue

            # Конфигурация именно этой гипотезы (по-гипотезная или общая)
            cfg_h = _config_for(h, configuration, per_hypothesis_configs)

            # Проверяем наличие кешированного результата для данной конфигурации
            if not _has_cached_result(h, cfg_h, db):
                # Результата нет: помечаем гипотезу и все зависимые (потомки в графе)
                # как требующие пересчета
                dependents = nx.descendants(graph, h)
                pne.add(h)
                pne.update(dependents)
                finished.add(h)
                finished.update(dependents)
            else:
                # Результат есть: проверяем R² перед включением в Pe
                r2 = _get_cached_r2(h, cfg_h, db)
                if r2 is not None and r2 < r2_threshold:
                    # R² ниже порога: отсекаем гипотезу и все зависимые
                    dependents = nx.descendants(graph, h)
                    pruned.add(h)
                    pruned.update(dependents)
                    finished.add(h)
                    finished.update(dependents)
                else:
                    # Повторное вычисление не требуется
                    pe.add(h)
                    finished.add(h)

        # Обновляем множество необработанных вершин
        remaining -= finished

    plan.needs_execution = pne
    plan.cached = pe

    return plan


class Planner:
    """Объектная обёртка над build_optimal_plan для удобства использования.

    Пример::

        planner = Planner(db=my_db, r2_threshold=0.8)
        plan = planner.plan(configuration, lattice)
    """

    def __init__(self, db: "Database", r2_threshold: float = 0.7) -> None:
        self.db = db
        self.r2_threshold = r2_threshold

    def plan(
        self,
        configuration,
        lattice: "HypothesisLattice",
        r2_threshold: float | None = None,
        per_hypothesis_configs: "Mapping | None" = None,
    ) -> ExecutionPlan:
        """Построить план виртуального эксперимента.

        Args:
            configuration: Конфигурация эксперимента.
            lattice: Решетка гипотез.
            r2_threshold: Порог R² для отсечения (переопределяет значение из __init__).
            per_hypothesis_configs: Необязательное отображение id гипотезы →
                конфигурация (см. build_optimal_plan).

        Returns:
            ExecutionPlan с множествами needs_execution и cached.
        """
        threshold = r2_threshold if r2_threshold is not None else self.r2_threshold
        return build_optimal_plan(
            configuration,
            lattice,
            self.db,
            r2_threshold=threshold,
            per_hypothesis_configs=per_hypothesis_configs,
        )

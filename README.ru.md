# gedanken

[![CI](https://github.com/dmkovalev/hyppo-ref/actions/workflows/ci.yml/badge.svg)](https://github.com/dmkovalev/hyppo-ref/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)

> 🇬🇧 [English version](README.md)

> gedanken — платформа для виртуальных (мысленных) экспериментов над
> решётками гипотез

Референсная реализация платформы управления виртуальными экспериментами,
описанной в диссертации. Дистрибутив публикуется как `gedanken` на PyPI;
путь импорта остаётся `hyppo`.

Сайт документации: <https://dmkovalev.github.io/hyppo-ref/> (заглушка —
публикуется после выката workflow развёртывания `mkdocs`).

## Что это?

Научные и инженерные модели обычно исследуются как конкурирующие семейства
гипотез — альтернативные уравнения для одного и того же явления, каждое со
своими допущениями и стоимостью (пере)вычисления. `gedanken` строит решётку
зависимостей над такими гипотезами (Алгоритм 1), планирует минимальный
набор моделей, которые нужно пересчитать после изменения (Алгоритм 4),
запускает и кэширует результаты, а также отслеживает эпистемический статус
каждой гипотезы (конкурирующая, вытесненная, подтверждённая) по мере
накопления свидетельств.

## Установка

```bash
pip install gedanken
```

Для локальной разработки:

```bash
uv sync --all-extras
```

Это устанавливает `hyppo` (распространяется как `gedanken`) со всеми
опциональными экстрами (`gui`, `mcp`, `coa`, `gp`, `data`, `docs`, `dev`).

## Быстрый старт

```python
from hyppo.coa._base import Equation, Structure
from hyppo.lattice_constructor._base import HypothesisLattice
from hyppo.planner._base import Planner
from hyppo.storage._base import Database


class Hypothesis:
    def __init__(self, name, formula):
        self.name = name
        self.structure = Structure([Equation(formula=formula)])


class Workflow:
    def __init__(self, tasks):
        self._tasks = tasks

    def get_tasks(self):
        return self._tasks


h1 = Hypothesis("H1", "q = a*p")     # q — выход H1
h2 = Hypothesis("H2", "wct = q*2")   # wct зависит от q -> H2 derived_by H1

workflow = Workflow([[h1], [h2]])
lattice = HypothesisLattice([h1, h2], workflow)
G = lattice.lattice
print(G.number_of_nodes(), G.number_of_edges())  # 2 1

Database.set_root(".hyppo_demo_db")
plan = Planner(db=Database).plan(configuration={}, lattice=lattice)
print(sorted(h.name for h in plan.needs_execution))  # ['H1', 'H2']
```

## Запуск тестов

```bash
pytest tests/ -v
```

## GUI

Запуск веб-GUI локально:

```bash
pip install "gedanken[gui]"
hyppo-gui
```

Открывает браузер по адресу http://127.0.0.1:8787 со встроенной демонстрацией
Norne (16 гипотез, заводнение нефтяного пласта). Пройдите полный жизненный
цикл виртуального эксперимента: задать гипотезы -> граф -> план пересчёта ->
запуск -> сравнение -> итерация.

## Архитектура

| Компонент | Модуль | Назначение |
|-----------|--------|------------|
| Ядро | `hyppo.core` | OWL-онтология + эпистемический статус гипотез |
| Менеджер | `hyppo.manager` | Оркестрация жизненного цикла виртуальных экспериментов |
| HypothesisGenerator | `hyppo.generator` | Базовая генерация гипотез (линейная регрессия + опциональное генетическое программирование) |
| COAConstructor | `hyppo.coa` | Каузальное упорядочивание систем уравнений |
| LatticesConstructor | `hyppo.lattice_constructor` | Алгоритм 1 — решётка гипотез |
| Planner | `hyppo.planner` | Минимальный план пересчёта (Алгоритм 4) |
| Runner | `hyppo.runner` | Исполнение моделей с повторными попытками |
| MetadataRepository | `hyppo.metadata_repository` | Кэш результатов, общий для планировщика и исполнителя |
| Versioning | `hyppo.versioning` | Отслеживание версий гипотез/запусков |
| Actions | `hyppo.actions` | Типизированные операции, доступные вызывающим сторонам |
| MCP-сервер | `hyppo.mcp` | Поверхность MCP-инструментов и ресурсов (см. ниже) |
| GUI | `hyppo.gui` | Веб-GUI (см. выше) |

## MCP-сервер

Hyppo предоставляет 8 типизированных действий и персону `Lattice Steward`
через MCP.

```bash
# stdio (Claude Code / Desktop)
hyppo-mcp

# streamable HTTP для кросс-MCP вызовов
hyppo-mcp --transport http --port 8082
```

После подключения клиенты видят инструменты
`mcp__hyppo__BuildVirtualExperiment`, `...DiffHypothesisStates`,
`...RegisterHypothesisVersion` и т. д., а также ресурс-персону
`hyppo://personas/lattice_steward.md`.

Действия записи (`RegisterHypothesisVersion`, `MarkRunWithVersion`)
сохраняются во встроенное хранилище SQLite (aiosqlite, по умолчанию
in-memory) из коробки. Установите `DATABASE_URL`, чтобы указать на внешнюю
базу данных (например, Postgres) для продакшн-развёртываний.

## Как цитировать

См. [`CITATION.cff`](CITATION.cff) — референсная реализация,
сопровождающая диссертацию Д. Ковалёва.

## Участие в разработке

См. [`CONTRIBUTING.md`](CONTRIBUTING.md) для настройки окружения
разработки, команд линтинга/проверки типов и структуры тестов.
Golden-контракт тестов: `tests/test_golden_claims.py` фиксирует
утверждения диссертации и статей за реализацией — если вы меняете
алгоритм, его golden-тесты должны проходить; если golden-тест выглядит
устаревшим, сначала правьте текст статьи/диссертации, затем тест.

## Лицензия

MIT — см. [`LICENSE`](LICENSE).

См. также [`CHANGELOG.md`](CHANGELOG.md) для истории релизов.

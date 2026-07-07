# hyppo — заметки для агентов

## ВАЖНО для GUI-агента: построение графа гипотез

Граф гипотез (`derived_by`-рёбра) строится **только** Алгоритмом 1 платформы:

```python
from hyppo.lattice_constructor._base import HypothesisLattice
from hyppo.coa._base import Equation, Structure

# каждая гипотеза: structure = Structure([Equation(formula="out = f(...)")])
# выходная переменная берётся из ЛЕВОЙ части формулы (fix 5907005)
lattice = HypothesisLattice(hypotheses, workflow)
G = lattice.lattice  # networkx.DiGraph, рёбра h_i -> h_j = «h_j зависит от h_i»
```

Не переопределяйте вывод рёбер локально. В `scripts/gui_real_data.py`
функция `build_graph_algorithm1()` — локальная перереализация; её нужно
заменить вызовом `HypothesisLattice` (после коммита 5907005 настоящий
алгоритм даёт корректный результат: выход гипотезы = LHS уравнения,
раньше брался алфавитно-первый символ и рёбра переворачивались).

Проверка корректности: на 16 гипотезах Norne HybridCRM Алгоритм 1 обязан
давать ровно 16 узлов, 17 рёбер, DAG глубины 5 (см. `norne_alg1_lattice.py`).

## Нумерация гипотез: статья ≠ старые скрипты

В статье для журнала «Системы высокой доступности»
(`thesis/papers/sht_dostupnost_v1.tex`) узлы перенумерованы **сплошно
H1–H16**: жидкость H1–H8, обводнённость H9–H14, ГРП = H15, прогноз нефти =
H16. Старые имена в коде (H11–H15, H12b, GRP, H19) — историческая нумерация.
В GUI при показе «как в статье» используйте сплошную нумерацию или явно
указывайте соответствие имён.

## Тесты

```bash
.venv/Scripts/python -m pytest tests -q   # полный набор, ~2 мин
```

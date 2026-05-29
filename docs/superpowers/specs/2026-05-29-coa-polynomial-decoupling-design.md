# COA: полиномиальное причинно-следственное ядро + отвязка от owlready

**Дата:** 2026-05-29
**Статус:** дизайн одобрен, ожидает ревью спеки
**Репозиторий:** `hyppo-ref` (эталонная реализация Hyppo)

## Проблема

Модуль `hyppo/coa/_base.py` (причинно-следственный анализ структур уравнений) имеет три связанных дефекта:

1. **Экспоненциальная сложность.** `find_minimal_structures` перебирает `powerset` уравнений с рекурсией → `2^{Θ(s)}` по числу уравнений в структуре `s`. Замерено: 0.006→35 мс при `s`=4→16 (≈×4.7 на каждые +2). Экспонента по размеру структуры, не по `|H|`, поэтому в `|H|`-замерах не проявлялась, но это худший случай и неверная витрина для кандидатской по эффективности.
2. **Нестабильность (порча памяти).** `Structure`/`Equation` — owlready2-индивиды (`class Equation(Thing)`, `class Structure(Artefact)`) в глобальном онтологическом мире. Создание тысяч объектов раздувает нативный quadstore → segfault и бессмысленные `TypeError` (наблюдалось: `'enumerate' object is not callable`, `'code' object is not iterable`). Алгоритм при этом не использует OWL-природу этих объектов.
3. **Фантомная зависимость.** `from latex2sympy import strToSympy` — пакета с таким API/именем не существует (extra `coa` ставит `latex2sympy2` с другим API). Модуль не импортируется ни на какой версии Python без shim. antlr4/latex2sympy несовместимы с Python 3.13 (requires-python проекта).

Причинно-следственная математика (полнота структур, минимальные подструктуры, причинное отображение, транзитивное замыкание) — чистая теория графов/множеств и не нуждается ни в OWL, ни в latex2sympy.

## Цели

- Заменить экспоненциальный `find_minimal_structures` на **полиномиальную декомпозицию Дюльмажа–Мендельсона** (паросочетание + SCC).
- **Отвязать** причинно-следственный алгоритм от owlready2 → устранить порчу памяти.
- Убрать фантомную зависимость `latex2sympy` → алгоритм работает на **Python 3.13** (штатный requires-python).
- Сохранить публичный API и семантику (обратная совместимость с `examples/`, `lattice_constructor`, тестами).

## Не-цели

- НЕ трогаем онтологическую подсистему рассуждения (`core/_base.py` OWL-классы, 16 правил инвалидации, owlready/HermiT) — она про настоящий OWL.
- НЕ меняем `lattice_constructor` (он работает на duck-typed `.structure` и networkx-графе).
- НЕ оптимизируем причинно-следственную математику сверх полиномиальной (YAGNI: Хопкрофт–Карп вместо Куна не требуется при малых структурах).

## Подход (выбран пользователем)

Чистое ядро отдельным модулем + тонкие классы-данные; полная отвязка `Structure`/`Equation` от owlready.

### Компонент 1: `hyppo/coa/causal.py` — чистое ядро

Чистый Python, **только stdlib** (без owlready, sympy, networkx — исключаем любую нативную нестабильность). Представление: уравнение = `frozenset[str]` (имена переменных), структура = `list[frozenset[str]]`. Все функции полиномиальны.

| Функция | Семантика | Сложность |
|---|---|---|
| `is_complete(eqs)` | `len(eqs) == len(union)` | `O(s·v)` |
| `perfect_matching(eqs)` | паросочетание уравнение→переменная (Кун), `None` если нет | `O(V·E)` |
| `block_decomposition(eqs)` | DM: matching → орграф зависимостей → SCC (Тарьян) → топопорядок конденсации | `O(V·E)` |
| `minimal_blocks(eqs)` | inclusion-минимальные полные подмножества (source-блоки конденсации) — семантика старого `find_minimal_structures` | `O(V·E)` |
| `causal_mapping(eqs)` | назначение переменных уравнениям внутри блоков, сорт-тай-брейк по имени | `O(V·E)` |
| `transitive_closure(eqs)` | `dict[var, set[var]]`, BFS по орграфу зависимостей; собственная вершина исключается (как `nx.descendants`) | `O(V·(V+E))` |

Матчинг и SCC реализуются на чистом Python (Кун ~15 строк, Тарьян ~25 строк) — ноль нативных зависимостей в горячем пути.

### Компонент 2: `hyppo/coa/_base.py` — тонкие классы (plain, без owlready)

- **`Equation`**: `formula` (опц.) + `vars` (множество имён переменных). Парсинг формулы через `sympy.sympify` (работает на 3.13; `f_i(...)=0` даёт `f_i` как Function, не свободную переменную — проверено). Можно создавать напрямую из множества переменных без формулы.
  - Совместимость: `eq.vars` остаются sympy-символами (тест `test_equation_parses_vars` делает `v.name for v in eq.vars`). Ядро работает с `.name` (строками), Equation конвертирует на границе.
- **`Structure`**: `equations`, `vars` (производное). Методы делегируют в `causal.py`: `is_complete`, `is_minimal`, `is_structure`, `find_minimal_structures` (возвращает Structure-обёртки блоков — контракт теста сохранён), `build_full_causal_mapping`, `build_transitive_closure` (возвращает dict — контракт сохранён), `union`, `difference`, `exogenous`, `endogenous`.
  - `build_matrix` (numpy) и `build_dcg` (graphviz) — **ленивый импорт**, ядро от них не зависит.
- **Удаляется:** наследование `Thing`/`Artefact`; `from latex2sympy import strToSympy`; `powerset`; экспоненциальная рекурсия в `find_minimal_structures`; `vars=`-substitution-ветка в `Structure.__init__` (пилинг теперь внутри `causal.py`).

### Поток данных

```
formula → sympy.sympify → free_symbols → Equation.vars (символы)
Structure(equations) → [frozenset имён] → causal.* (matching+SCC) → блоки/отображение/замыкание
                      → Structure-обёртки / dict (те же формы, что и раньше)
```

### Зависимость от других модулей

`pyproject.toml`: extra `coa` теряет `latex2sympy2` (sympy уже в core-зависимостях; graphviz остаётся опциональным для `build_dcg`). `core/_base.py` (OWL-классы) не меняется; COA-ядро питает решётку (`lattice_constructor`, networkx), а та — отдельно OWL-слой; отвязка COA это не рвёт.

## Обратная совместимость

Потребители `coa.Structure/Equation`: `examples/coa_example.py`, `examples/random_structures/main.py`, `tests/test_coa.py`, тайп-хинт в `lattice_constructor` (`TYPE_CHECKING`). Требуемый интерфейс: `Equation(formula=...)`, `Structure(equations=...)`, `.vars`, `.equations`, `.is_complete()`, `.union()`, `.build_transitive_closure()`, `.find_minimal_structures()` → всё сохраняется. Адаптеры/менеджер/actions `coa.Structure` не создают.

## Тестирование и критерий приёмки (Python 3.13)

1. **`tests/test_causal.py`** (новый): юнит-тесты ядра на plain-данных — паросочетание; эквивалентность `block_decomposition`/`minimal_blocks` ↔ brute-force powerset исчерпывающе для `n≤8`; кейсы 3-cycle (воспроизводитель старого краша), 4-cycle, triangular, 7-уравнение из диссертации.
2. **Стресс-тест:** 10000 случайных полных структур в одном процессе на Python 3.13 → 0 крашей/segfault.
3. **Существующие тесты** `test_coa.py`, `test_lattice.py` — зелёные (без skip, т.к. sympify работает на 3.13).
4. **Перемер показателя:** прогон Алгоритма 1 (`build_lattice`) на реальной библиотеке по диапазону `|H|` → подтвердить `a≈2` и отсутствие экспоненты по размеру структуры. Число для диссертации — теперь от настоящего алгоритма, а не от greedy-переписывания в `asymptotic_validation.py`.
5. **Регресс-эквивалентность:** на 7-уравнении и наборе кейсов результат == ранее провалидированной DM-семантике.

## Риски

- **`Equation.vars` символы vs строки.** Ядро на строках, Equation на символах — конвертация на границе. Риск рассинхрона при коллизии имён; митигируется единым `.name`-представлением внутри ядра.
- **sympify edge-cases** (наблюдался `'<' not supported` на некоторых формах). Митигируется: ядро принимает уже-распарсенные переменные; стресс-тест ядра не парсит формулы вовсе.
- **Семантика `find_minimal_structures`.** Старое = inclusion-минимальные полные подмножества одного уровня; новое = `minimal_blocks` = source-блоки конденсации. Эквивалентность проверяется brute-force-тестом.

## Чистка

Scratch-артефакты (`scripts/poly_minimal_verify.py`, `poly_exponent_partc.py`, `run_library_dm.py`, `latex2sympy.py`, `test_powerset_fix.py`, `test_fix_deterministic.py`) — удалить либо превратить в нормальные тесты в `tests/`.

## Файлы (план изменений)

| Файл | Изменение |
|---|---|
| `hyppo/coa/causal.py` | **новый** — чистое ядро (stdlib) |
| `hyppo/coa/_base.py` | переписать: plain `Equation`/`Structure`, делегирование в `causal.py`, убрать owlready/latex2sympy/powerset |
| `hyppo/coa/__init__.py` | при необходимости обновить экспорт |
| `tests/test_causal.py` | **новый** — юнит-тесты ядра + стресс + эквивалентность |
| `tests/test_coa.py` | снять skip; адаптировать при изменении сигнатур |
| `pyproject.toml` | extra `coa`: убрать `latex2sympy2` |
| `scripts/*` (scratch) | удалить/перенести в tests |

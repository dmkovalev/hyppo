# Минимальный безопасный пересчёт: эксперименты статьи

Пакет экспериментов журнальной статьи о минимальном безопасном пересчёте (Э1--Э6).
Все команды запускаются из корня репозитория интерпретатором venv комплекса
(`.venv/Scripts/python.exe` в Windows, далее `python`), с `PYTHONPATH=.`.
Время указано для одной машины (Windows 11, Python 3.13.3) и служит ориентиром.

- `common.py`: загрузчики графов (CRM Brugge из гл. 4, 26 графов nf-core,
  171 трасса WfCommons корпуса Э2 с проверкой по манифесту).
- `simulators.py`: правила инвалидации Snakemake, DVC, Dagster, Nextflow, Bazel и
  комплекса на графе задач (события `param`, `evict`, `revise`).
- `ports/`: конвейер Brugge из `examples/research/baseline_filecache`, перенесённый
  на DVC и Dagster (плюс тонкий адаптер Snakemake) с общим интерфейсом
  `run_workload(workdir, grid, events, declared)`; нужен Э6.
- `figures.py`: все рисунки статьи (`figures/*.pdf`) по JSON из `data/`.

Наука конвейера (воркеры, хранилище, правки) не дублируется: порты вызывают
`baseline_filecache/workers.py` и `revise.py`.

## Данные

| Данные | Где | В git |
|---|---|---|
| CRM Brugge (16 гипотез, 19 рёбер) | `common.CRM_EDGES` | да |
| 26 графов nf-core | `examples/research/planning/data/hand_curated_hypothesis_graphs_26.json` | да |
| 171 трасса WfCommons (~309 МБ) | `examples/research/planning/cache/wfcommons/` | нет, фиксируется манифестом |
| ряды Brugge для реальных прогонов (Э4, Э6) | `thesis/papers/brugge_run/` | да, в репозитории диссертации |

**WfCommons.** Состав корпуса задан `planning/data/wfcommons_validation_results.json`
(171 трасса, семейства nextflow, snakemake, pegasus), содержимое -- манифестом
`data/wfcommons_manifest.json` (для каждой трассы: путь, sha256, размер, git blob sha1,
число узлов и рёбер). `common.wfcommons_graphs` сверяет каждый файл с манифестом и
при отсутствии или расхождении бросает `CorpusMismatch`. Загрузка с проверкой:

```
python -m examples.research.safe_recompute.fetch_wfcommons            # загрузить недостающие
python -m examples.research.safe_recompute.fetch_wfcommons verify     # только проверить
```

Файлы берутся из github.com/wfcommons/wfinstances (`raw.githubusercontent.com`,
пути `<семейство>/[<подкаталог>/]<имя>.json`, как в
`planning/scripts/wfcommons_validation.py`). Трассы WfInstances меняются без смены
имён (при повторной загрузке совпали число узлов и рёбер, но не байты, а у трасс
RASflow -- идентификаторы задач), поэтому загрузка с `main` может дать расхождение
хешей; тогда нужную ревизию задаёт `--ref <commit>` (её можно найти по `git_blob_sha1`
манифеста). Нужен только для Э6 `scale` и тестов корпуса.

**Brugge.** Реальные прогоны Э4 и Э6 читают `brugge_perwell.npz` (дебиты жидкости,
закачка, время) и `brugge_oilwater.npz` (нефть, вода) из каталога `brugge_run`:
месячные ряды по скважинам эталонной модели месторождения Brugge, подготовленные для
гл. 4 диссертации и хранящиеся в репозитории диссертации
(`thesis/papers/brugge_run`, github.com/dmkovalev/thesis). Контрольные суммы:

```
4a0d1a8768f24c157c6327427d02d6a629b6bb046d02a6b495a93642bc92972f  brugge_perwell.npz
9a102681064868857a07919613763e80bf7c608f49c45f22e20f00c71372e622  brugge_oilwater.npz
```

## Окружение

| Компонент | Версия |
|---|---|
| Python, venv комплекса (`hyppo-ref/.venv`) | 3.13.3 |
| Python, venv инструментов (`diss/.venv-cache`) | 3.13.3 |
| Snakemake | 9.23.1 |
| DVC | 3.67.1 |
| Dagster | 1.13.26 |
| Git | 2.49.0.windows.1 |

ОС: Windows 11 Enterprise 10.0.26200.

Установка:

```
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,data]"   # комплекс, pywaterflood
python -m venv ../.venv-cache                                         # инструменты, отдельно
../.venv-cache/Scripts/pip install snakemake==9.23.1 dvc==3.67.1 dagster==1.13.26
```

Venv инструментов отделён от venv комплекса; воркеры конвейера (numpy, pywaterflood)
всегда запускаются интерпретатором комплекса (`HYPPO_PY`). Dagster закреплён точно:
`ports/_dagster_runner.py` использует его приватный API и проверяет версию.
Venv инструментов ищут в `$HYPPO_DIAG_ROOT/.venv-cache`, затем в родительских каталогах
репозитория (или задаётся `HYPPO_TOOLS_PY`).

Переменные окружения (нужны только Э4 и реальным прогонам Э6):

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `HYPPO_BRUGGE_RUN` | каталог с `brugge_perwell.npz`, `brugge_oilwater.npz` | `<diss>/thesis/papers/brugge_run` |
| `HYPPO_PY` | интерпретатор воркеров (venv комплекса); обязателен в git worktree, где нет `.venv` | `<repo>/.venv/Scripts/python.exe` |
| `HYPPO_DIAG_ROOT` | каталог, содержащий `.venv-cache` | родитель репозитория (`F:/git-repos/diss`) |

Без данных Brugge или инструментов соответствующие тесты пропускаются.

## Э1. Вытеснение записей: безопасный режим против каскада

```
python -m examples.research.safe_recompute.e1_eviction [steps=40] [seeds=10]
```

27 графов (CRM + 26 nf-core), ёмкость репозитория 0,25--2 |V|, вытеснение LRU и
случайное. Время: около 3,5 ч (примерно 20 мин на каждую из 10 пар «политика, ёмкость»).
Выход: `data/e1_eviction.json` (перезаписывается).

При изменении только графа CRM (`common.CRM_EDGES`) достаточно пересчитать его одного
(около 5 мин) и влить в имеющийся JSON с пересчётом медиан, IQR и `crm_ratio` по 27 графам:

```
python -m examples.research.safe_recompute.e1_merge_crm [steps=40] [seeds=10]
```

## Э2. План минимальной стоимости

```
python -m examples.research.safe_recompute.e2_mincost [seeds=20] [out] [graphs_limit]
```

Время: около 10 с. Выход: `data/e2_mincost.json`.

## Э3. Причинное уточнение плана

```
python -m examples.research.safe_recompute.e3_causal
```

Время: около 5 с. Выход: `data/e3_causal.json`.

## Э4. Файловый кэш как базовая линия (Snakemake против комплекса)

Живёт в `examples/research/baseline_filecache` (см. его README):

```
bash examples/research/baseline_filecache/run_all.sh small   # дымовой
bash examples/research/baseline_filecache/run_all.sh full
# то же без bash:
python -m examples.research.baseline_filecache.experiment --grid full --cores 4
```

Нужны Snakemake в `.venv-cache` и данные Brugge. Время не измерялось: `small` --
минуты, `full` (154 задачи, 85 испытаний) -- десятки минут. Выход и рисунок:
`baseline_filecache/results/results_{grid}.json`, `results/missed_vs_p_{grid}.pdf`.

## Э5. Время планирования

```
python -m examples.research.safe_recompute.e5_planning_time
```

Время: около 20 с. Выход: `data/e5_planning_time.json` (времена машинно-зависимы).

## Э6. Сравнение с Snakemake, DVC, Dagster

```
python -m examples.research.safe_recompute.e6_systems real small    # ~6 мин
python -m examples.research.safe_recompute.e6_systems real full     # ~65 мин
python -m examples.research.safe_recompute.e6_systems scale 200     # ~2 мин
python -m examples.research.safe_recompute.e6_systems derive        # пересчёт сверки из JSON
```

`real` -- реальные прогоны Brugge (13 событий: 5 правок ridge, 5 удалений выхода,
3 ревизии хранилища) в Snakemake, DVC, Dagster и комплексе (режим 'safe') против
эталона полным пересчётом и сверка `simulators.run_sequence` с реальными
множествами (forced и native evict). Нужны инструменты, данные Brugge и переменные
окружения выше. `scale` -- симуляторы на CRM, 26 nf-core и 171 WfCommons при доле
объявленных пар p in {0, 0.5, 1}, чувствительность к P(выход изменился)
(`sensitivity`) и к seed генератора при P = 0,7 (`seed_sensitivity`, seeds 1--3);
нужен корпус WfCommons. Каждая команда дополняет `data/e6_systems.json`, сохраняя
остальные ключи (`real`, `validation`, `per_system`, `favourable_worlds`,
`sensitivity`, `seed_sensitivity`).

## Рисунки

```
python -m examples.research.safe_recompute.figures
```

Читает `data/e{1,2,3,5,6}_*.json`, пишет `figures/e1_eviction.pdf`, `e2_mincost.pdf`,
`e3_causal.pdf`, `e5_planning_time.pdf`, `e6_systems.pdf` (несколько секунд) и печатает
наименьший кегль текста при ширине рисунка в статье (0,48 textwidth, Э6 -- textwidth);
он не меньше 7 пт. Рисунок Э4 строит сам эксперимент (см. выше).

## Тесты

```
python -m pytest -p no:cacheprovider tests/test_safe_recompute_common.py \
    tests/test_safe_recompute_experiments.py tests/test_safe_recompute_ports.py tests/test_simulators.py
```

# Минимальный безопасный пересчёт: эксперименты статьи

Пакет экспериментов журнальной статьи о минимальном безопасном пересчёте.

- `common.py`: загрузчики графов (CRM Brugge из гл. 4, 26 графов nf-core,
  171 трасса WfCommons корпуса Э2).
- `simulators.py`: правила инвалидации Snakemake, DVC, Dagster, Nextflow, Bazel и
  комплекса на графе задач (события `param`, `evict`, `revise`).
- `ports/`: конвейер Brugge из `examples/research/baseline_filecache`, перенесённый
  на DVC и Dagster (плюс тонкий адаптер Snakemake) с общим интерфейсом
  `run_workload(workdir, grid, events, declared)`. Порты нужны для Э6:
  сверить симуляторы с реальными запусками систем.

Наука конвейера (воркеры, хранилище, правки) не дублируется: порты вызывают
`baseline_filecache/workers.py` и `revise.py`.

## Окружение

| Компонент | Версия |
|---|---|
| Python, venv комплекса (`hyppo-ref/.venv`) | 3.13.3 |
| Python, venv инструментов (`diss/.venv-cache`) | 3.13.3 |
| Snakemake (`.venv-cache`, `python -m snakemake --version`) | 9.23.1 |
| DVC (`.venv-cache/Scripts/dvc.exe`) | 3.67.1 |
| Dagster (`.venv-cache`) | 1.13.26 |
| Git | 2.49.0.windows.1 |

ОС: Windows 11 Enterprise 10.0.26200. Данные Brugge: `thesis/papers/brugge_run`
(или переменная `HYPPO_BRUGGE_RUN`).

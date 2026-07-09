# Hyppo GUI — запуск

Веб-интерфейс демонстрации методов виртуального эксперимента (React + Vite фронтенд,
FastAPI бэкенд). Данные — реальные (pywaterflood CRM на Brugge/Norne, граф Алгоритма 1).

Одно окружение Python (управляется `uv`, версия из `pyproject.toml` — `>=3.13`);
pywaterflood ставится extra `data`. Отдельный `.venv311` больше не нужен.

## Установка (один раз)

```bash
uv sync --extra gui --extra data --extra dev   # бэкенд + pywaterflood + pytest
cd webui && npm install && cd ..               # зависимости фронтенда
```

Все команды — из корня репозитория (`hyppo-ref`). `PYTHONUTF8=1` — для корректного
вывода кириллицы в Windows-консоли.

## Режим A — продакшн (собранный фронт под одним URL)

```bash
# 1. данные (только если менялись модель/граф/scripts/gui_real_data.py; ~1–2 мин)
PYTHONUTF8=1 uv run python scripts/gui_real_data.py      # → hyppo/gui/real_data.json

# 2. сборка фронта (только если менялся webui/src)
cd webui && npm run build && cd ..                       # → webui/dist
rm -rf hyppo/gui/static && cp -r webui/dist hyppo/gui/static  # → served + packaged in wheel

# 3. сервер + браузер
uv run hyppo-gui --port 8801 --no-browser &
start "" "http://localhost:8801/"
```

- После шага 1 **перезапустить сервер** — он кэширует `real_data.json` в памяти
  (`hyppo/gui/api/real.py`, `_CACHE`).
- После шага 2 нужно скопировать `webui/dist/*` в `hyppo/gui/static/` (команда
  выше) и **обновить страницу** (Ctrl+F5): `StaticFiles` в `hyppo/gui/app.py`
  читает `hyppo/gui/static` с диска (не `webui/dist` напрямую) — этот же каталог
  упаковывается в wheel через `pyproject.toml` → `[tool.setuptools.package-data]`.
- Без `hyppo/gui/static/index.html` сервер логирует явное предупреждение и
  работает в режиме только API (см. warning в `hyppo/gui/app.py`).

## Режим B — разработка фронта (HMR)

Vite-прокси направляет `/api` на `http://127.0.0.1:8787` (дефолтный порт CLI),
поэтому бэкенд поднимаем **на 8787**:

```bash
# терминал 1 — бэкенд на дефолтном порту 8787 (совпадает с прокси vite)
uv run hyppo-gui --no-browser

# терминал 2 — vite dev-сервер с горячей перезагрузкой
cd webui && npm run dev          # → http://localhost:5173
```

Правки в `webui/src` применяются мгновенно, `/api/*` проксируется на бэкенд.

## Тесты и перезапуск

```bash
uv run pytest tests -q                          # весь набор (~2 мин)
uv run pytest tests/test_golden_claims.py -q    # только golden (<10 c)
```

Перезапуск сервера на занятом порту (8801):

```bash
PID=$(uv run python -c "import subprocess;[print(l.split()[-1]) for l in subprocess.run(['netstat','-ano'],capture_output=True,text=True).stdout.splitlines() if ':8801' in l and 'LISTEN' in l]")
[ -n "$PID" ] && taskkill //PID $PID //F
uv run hyppo-gui --port 8801 --no-browser &
```

## Опции CLI (`hyppo-gui`)

- `--port` (по умолч. `8787`), `--host` (`127.0.0.1`), `--db` (`hyppo_gui.db`),
  `--no-browser` — не открывать браузер автоматически.

## Обычный цикл

- Правил только `webui/src` → `npm run build` + копирование в `hyppo/gui/static/`
  + Ctrl+F5 (или Режим B с HMR, без копирования).
- Правил `scripts/gui_real_data.py` (данные/граф) → шаг 1 + перезапуск сервера.

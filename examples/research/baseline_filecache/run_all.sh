#!/usr/bin/env bash
# Эксперимент «корректно настроенный файловый кэш как базовая линия».
# bash examples/research/baseline_filecache/run_all.sh [small|full]
set -euo pipefail
cd "$(dirname "$0")/../../.."
GRID="${1:-full}"
.venv/Scripts/python.exe -m examples.research.baseline_filecache.experiment --grid "$GRID" --cores "${CORES:-4}"

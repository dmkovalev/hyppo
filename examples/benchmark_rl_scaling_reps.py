"""Драйвер повторных замеров RL-материализации: N успешных прогонов на размер
(каждый~--- свежий процесс benchmark_rl_scaling.py, ретрай при флаковом сбое
rdflib/owlrl под Python 3.13), затем медиана и интервал [min; max] + std.
"""
from __future__ import annotations
import json, re, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / "benchmark_rl_scaling.py"
OUT = HERE / "research" / "planning" / "data" / "rl_scaling_reps_10k.json"

# (target_nodes, NI, NP, reps)
PLAN = [(10166, 100, 95, 5)]
MAX_RETRY = 400  # доп. попыток на точку за один запуск (резюмируется между запусками)

ROW = re.compile(r"^\s*(\d+)\s+\S+\s+([\d.]+)\s+(\d+)\s+(\d+)\s*$")


def one_run(NI: int, NP: int, spec: str):
    """Один свежий процесс; вернуть (sec, triples, stale) или None при сбое."""
    p = subprocess.run(
        [sys.executable, "-u", str(WORKER), "--sizes", spec],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        return None
    for line in p.stdout.splitlines():
        m = ROW.match(line)
        if m:
            return float(m.group(2)), int(m.group(3)), int(m.group(4))
    return None


META = {
    "description": "Повторные замеры RL-материализации (owlrl, store=SimpleMemory), "
                   "медиана и интервал по N успешным прогонам, свежий процесс на прогон.",
    "platform": "Windows 11, Python 3.13, rdflib 7.6.0, owlrl 7.6.2",
    "worker": "examples/benchmark_rl_scaling.py",
}


def _pct(srt, p):
    if not srt:
        return 0.0
    if len(srt) == 1:
        return srt[0]
    k = (len(srt) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(srt) - 1)
    return srt[lo] + (srt[hi] - srt[lo]) * (k - lo)


def make_entry(target, ninp, times, triples, stale):
    srt = sorted(times)
    return {
        "nodes": target, "NIxNP": ninp, "reps": len(times),
        "median_s": round(statistics.median(times), 3) if times else 0.0,
        "min_s": round(min(times), 3) if times else 0.0,
        "max_s": round(max(times), 3) if times else 0.0,
        "p05_s": round(_pct(srt, 0.05), 3), "p95_s": round(_pct(srt, 0.95), 3),
        "std_s": round(statistics.pstdev(times), 3) if len(times) > 1 else 0.0,
        "triples": triples, "stale": stale, "raw_s": [round(t, 3) for t in times],
    }


def load_existing():
    if OUT.exists():
        try:
            data = json.loads(OUT.read_text(encoding="utf-8"))
            return {p["nodes"]: p for p in data.get("points", [])}
        except Exception:
            return {}
    return {}


def save(entries):
    OUT.write_text(json.dumps({**META, "points": list(entries.values())},
                              ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    existing = load_existing()
    entries = dict(existing)  # nodes -> entry (резюме между запусками)
    for target, NI, NP, reps in PLAN:
        ninp = f"{NI}x{NP}"
        prev = existing.get(target, {})
        times = list(prev.get("raw_s", []))[:reps]
        triples, stale = prev.get("triples"), prev.get("stale")
        entries[target] = make_entry(target, ninp, times, triples, stale)
        save(entries)
        if times:
            print(f"[{ninp}] резюме: уже есть {len(times)}/{reps}", flush=True)
        attempts = 0
        while len(times) < reps and attempts < MAX_RETRY:
            attempts += 1
            r = one_run(NI, NP, f"{target}:{ninp}")
            if r is None:
                print(f"[{ninp}] попытка {attempts}: сбой, ретрай", flush=True)
                continue
            sec, triples, stale = r
            times.append(sec)
            entries[target] = make_entry(target, ninp, times, triples, stale)
            save(entries)  # инкрементальное сохранение: прогресс не теряется
            print(f"[{ninp}] rep {len(times)}/{reps}: {sec:.3f} s "
                  f"(triples={triples}, stale={stale})", flush=True)
        e = entries[target]
        print(f"=> {target}: медиана {e['median_s']} s, [{e['min_s']}; {e['max_s']}], "
              f"n={e['reps']}\n", flush=True)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Benchmark for Algorithm 1 (consistency check) of `iip2026_planning.tex`.

Builds synthetic OWL 2 DL ontologies of varying size |H|, runs
``hyppo.ontology.consistency.check_consistency``, and records
T_HermiT and T_StageB separately for the paper's Section 5 timing
table.

Usage:
    uv run python algorithm1_bench.py \
        --sizes 10 50 100 200 500 \
        --reps 5 \
        --out algorithm1_bench.json

Note: HermiT requires Java 11+ on PATH. Without Java, Stage A is
skipped (``run_hermit=False``) and only Stage B (C3/C4/C5) is timed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _has_java() -> bool:
    return shutil.which("java") is not None


def _build_synthetic_lattice(
    n: int, mean_deg: float, rng: np.random.Generator
) -> dict[int, set[int]]:
    """Erdős-Rényi DAG with avg out-degree ≈ mean_deg."""
    p = mean_deg / max(n - 1, 1)
    lattice: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                lattice[i].add(j)
    return lattice


def _build_synthetic_ontology(n: int, lattice: dict[int, set[int]]):
    """Programmatically build a Hyppo-compatible ontology with n hypotheses,
    n models, and the given derived_by edges."""
    from owlready2 import get_ontology

    iri = f"http://synthesis.ipi.ac.ru/bench_n{n}.owl"
    onto = get_ontology(iri)

    from hyppo.core._base import (  # noqa: F401
        Hypothesis,
        Model,
        derived_by,
        impacts,
        is_implemented_by_model,
        refers_to_hypothesis,
    )

    hyps: list = []
    mods: list = []
    with onto:
        for i in range(n):
            h = Hypothesis(f"H{i}")
            h.has_for_id = i
            h.has_for_name = f"hyp_{i}"
            h.has_for_description = f"synthetic hypothesis {i}"
            h.has_for_authors = ["bench"]
            from datetime import datetime, timezone

            h.has_for_createdate = datetime.now(tz=timezone.utc)
            h.has_for_lastupdate = h.has_for_createdate
            h.has_for_probability = 0.5
            hyps.append(h)

            m = Model(f"M{i}")
            m.has_for_id = n + i
            m.has_for_name = f"model_{i}"
            m.has_for_description = f"synthetic model {i}"
            m.has_for_authors = ["bench"]
            m.has_for_createdate = h.has_for_createdate
            m.has_for_lastupdate = h.has_for_createdate
            mods.append(m)

            h.is_implemented_by_model = [m]

        for u, succ in lattice.items():
            for v in succ:
                hyps[v].derived_by.append(hyps[u])

    return onto, hyps, mods


def benchmark(
    sizes: list[int],
    n_reps: int,
    mean_deg: float,
    seed: int,
    run_hermit: bool,
) -> list[dict[str, Any]]:
    from hyppo.ontology.consistency import check_consistency

    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    # JVM / JIT warmup: one full consistency check (with HermiT if available)
    # before any timed measurements, so that t_b and t_ab are both measured
    # on a warm JVM.  Without this, t_b (measured first) suffers cold-JVM
    # overhead, causing t_a = t_ab - t_b to be underestimated.
    if run_hermit:
        print("Warming up JVM (1 pre-run)...", flush=True)
        try:
            _warmup_lattice = _build_synthetic_lattice(sizes[0], mean_deg, rng)
            _warmup_onto, _, _ = _build_synthetic_ontology(sizes[0], _warmup_lattice)
            check_consistency(None, _warmup_onto, _warmup_lattice, run_hermit=True)
        except Exception as exc:  # pragma: no cover
            print(f"  warmup failed (non-fatal): {exc}", file=sys.stderr)

    for n in sizes:
        for rep in range(n_reps):
            lattice = _build_synthetic_lattice(n, mean_deg, rng)
            n_edges = sum(len(s) for s in lattice.values())

            t0 = time.perf_counter()
            try:
                onto, _hyps, _mods = _build_synthetic_ontology(n, lattice)
            except Exception as exc:  # pragma: no cover
                print(f"  build failed n={n} rep={rep}: {exc}", file=sys.stderr)
                continue
            t_build = time.perf_counter() - t0

            t_a = float("nan")
            t_b = float("nan")
            status = "build_only"
            try:
                ve = None
                if run_hermit:
                    # Stage A: HermiT classification only.
                    # We time only the HermiT call by running check_consistency
                    # with run_hermit=True (stages A+B combined) then subtract
                    # the independently-measured stage B duration.
                    # Both measurements are on a warm JVM (see warmup above),
                    # so the subtraction t_ab - t_b correctly isolates HermiT.
                    t_ab_start = time.perf_counter()
                    res = check_consistency(
                        ve, onto, lattice, run_hermit=True
                    )
                    t_ab = time.perf_counter() - t_ab_start

                    t_b_start = time.perf_counter()
                    res_b = check_consistency(
                        ve, onto, lattice, run_hermit=False
                    )
                    t_b = time.perf_counter() - t_b_start

                    # Stage A time = combined - stage B (both on warm JVM;
                    # guaranteed non-negative because HermiT dominates).
                    t_a = max(0.0, t_ab - t_b)
                    status = res.status
                else:
                    t_b_start = time.perf_counter()
                    res = check_consistency(
                        ve, onto, lattice, run_hermit=False
                    )
                    t_b = time.perf_counter() - t_b_start
                    status = res.status
            except Exception as exc:  # pragma: no cover
                print(
                    f"  check_consistency failed n={n} rep={rep}: {exc}",
                    file=sys.stderr,
                )
                status = f"error: {exc}"

            row = {
                "n_hypotheses": n,
                "n_edges": n_edges,
                "rep": rep,
                "t_build_s": t_build,
                "t_stage_a_s": None if np.isnan(t_a) else t_a,
                "t_stage_b_s": None if np.isnan(t_b) else t_b,
                "status": status,
            }
            results.append(row)
            print(
                f"  n={n:4d} rep={rep} | build={t_build:.3f}s "
                f"a={'--' if np.isnan(t_a) else f'{t_a:.3f}s'} "
                f"b={'--' if np.isnan(t_b) else f'{t_b:.4f}s'} {status}"
            )

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100, 200, 500])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mean-deg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", type=Path, default=Path("algorithm1_bench.json")
    )
    ap.add_argument(
        "--no-hermit",
        action="store_true",
        help="Skip Stage A (HermiT) — measure only Stage B procedural checks.",
    )
    args = ap.parse_args()

    run_hermit = not args.no_hermit and _has_java()
    if not run_hermit and not args.no_hermit:
        print(
            "Java not found on PATH — HermiT will be skipped (Stage A timing = NaN).",
            file=sys.stderr,
        )

    print(
        f"sizes={args.sizes} reps={args.reps} mean_deg={args.mean_deg} "
        f"seed={args.seed} run_hermit={run_hermit}"
    )
    rows = benchmark(
        args.sizes,
        args.reps,
        args.mean_deg,
        args.seed,
        run_hermit,
    )
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

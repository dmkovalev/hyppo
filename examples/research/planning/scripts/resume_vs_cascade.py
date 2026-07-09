#!/usr/bin/env python
"""Workflow-manager baseline: Nextflow `-resume` (file content-addressed cache)
vs hyppo's semantic `derived_by` cascade. Addresses the dissertation-council
workflow critique without installing Nextflow, by faithfully modelling the
`-resume` rule (which is fully specified and deterministic):

    a task is re-executed  <=>  the hash of (its input files + script + params)
    differs from the cached run; otherwise its cached output is reused. An input
    file's hash is the producing task's output-content hash, so a change
    propagates downstream exactly along the data flow.

Two results:

  (1) EQUIVALENCE on a file-DAG. When a task's script/params change (so its output
      bytes change), the set of tasks `-resume` re-executes equals hyppo's
      `HypothesisGraph.plan()` recompute set (downward closure of the changed
      node). Checked on 1000 random DAGs -> 0 mismatches. This is the honest basis
      for the "±" in the comparison table: on a pure file-DAG both cascade
      identically; hyppo gives no extra reuse there.

  (2) HIDDEN STALENESS that `-resume` MISSES. If an upstream hypothesis is
      invalidated semantically (e.g. reasoner marks it REFUTED / a non-file
      assumption changed) but its output FILE is byte-identical, `-resume` sees no
      changed input and silently reuses every downstream task. hyppo's
      `derived_by` cascade, keyed on the hypothesis graph rather than file bytes,
      marks the descendants stale. This is the qualitative advantage of the
      semantic graph over file hashing.

Run: PYTHONPATH=<hyppo-ref> python examples/research/planning/scripts/resume_vs_cascade.py
Out: examples/research/planning/data/resume_vs_cascade.json
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from hyppo.coa import HypothesisGraph

DATA = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------
# Faithful model of Nextflow `-resume` over a file-DAG.
# --------------------------------------------------------------------------
def nf_resume_recompute(n, edges, scripts, changed):
    """Return the set of tasks Nextflow `-resume` re-executes after `changed`
    tasks get a new script/param. A task's output content-hash = sha256 of its
    sorted parent output-hashes + its own script; it re-runs iff that hash differs
    from the cached (pre-change) hash. `scripts`: {task: str} script/param text."""
    parents = defaultdict(list)
    for u, v in edges:
        parents[v].append(u)
    order = _topo(n, edges)

    def out_hash(task_scripts):
        h = {}
        for t in order:
            material = task_scripts[t] + "|" + "".join(sorted(h[p] for p in parents[t]))
            h[t] = hashlib.sha256(material.encode()).hexdigest()
        return h

    cached = out_hash(scripts)
    new_scripts = dict(scripts)
    for c in changed:
        new_scripts[c] = scripts[c] + "#changed"      # script/param edit
    fresh = out_hash(new_scripts)
    return {t for t in range(n) if fresh[t] != cached[t]}


def _topo(n, edges):
    adj = defaultdict(set)
    indeg = {i: 0 for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        indeg[v] += 1
    q = [i for i in range(n) if indeg[i] == 0]
    order = []
    while q:
        x = q.pop()
        order.append(x)
        for w in adj[x]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return order


# --------------------------------------------------------------------------
# (1) Equivalence on file-DAGs
# --------------------------------------------------------------------------
def check_equivalence(trials=1000):
    mismatches = 0
    for t in range(trials):
        rng = random.Random(t)
        n = rng.randint(2, 40)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.25]
        scripts = {i: f"proc_{i}" for i in range(n)}
        changed = {rng.randrange(n)}
        nf = nf_resume_recompute(n, edges, scripts, changed)
        # hyppo: the changed task is "not cached", everything else cached
        cached = {i for i in range(n)} - changed
        hp = HypothesisGraph.from_edges(n, edges).plan(cached)
        if nf != hp:
            mismatches += 1
    return trials, mismatches


# --------------------------------------------------------------------------
# (2) Hidden staleness: chain A -> B -> C
# --------------------------------------------------------------------------
def hidden_staleness_demo():
    # file-DAG chain: 0 (A) -> 1 (B) -> 2 (C)
    n, edges = 3, [(0, 1), (1, 2)]
    names = {0: "A", 1: "B", 2: "C"}

    # Semantic change to A's hypothesis (e.g. reasoner marks it REFUTED) that does
    # NOT alter A's output file: model A's output as independent of the changed
    # aspect -> A's output bytes stay the same, so -resume sees no changed input.
    scripts = {0: "A_script", 1: "B_script", 2: "C_script"}
    # -resume: nothing in the FILE graph changed -> recompute set empty
    nf = nf_resume_recompute(n, edges, scripts, changed=set())  # no file/script change
    # hyppo: A invalidated semantically -> derived_by cascade marks descendants
    cached = {1, 2}            # B, C have cached results; A is invalidated (not cached)
    hp = HypothesisGraph.from_edges(n, edges).plan(cached)
    return {
        "graph": "A->B->C",
        "scenario": "A's hypothesis invalidated semantically; A's output file byte-identical",
        "nextflow_resume_recompute": sorted(names[t] for t in nf),
        "hyppo_cascade_recompute": sorted(names[t] for t in hp),
        "missed_by_resume": sorted(names[t] for t in (hp - nf)),
    }


def main():
    trials, mism = check_equivalence(1000)
    print(f"(1) equivalence on file-DAGs: {trials} random DAGs, mismatches = {mism}")
    print("    => on a pure file-DAG, Nextflow -resume and hyppo plan() recompute the "
          "SAME set (justifies the '±' in the comparison table).")

    demo = hidden_staleness_demo()
    print(f"\n(2) hidden staleness ({demo['graph']}): {demo['scenario']}")
    print(f"    Nextflow -resume re-runs: {demo['nextflow_resume_recompute']}  (file bytes unchanged -> reuses all)")
    print(f"    hyppo derived_by cascade: {demo['hyppo_cascade_recompute']}")
    print(f"    => MISSED by -resume, caught by hyppo: {demo['missed_by_resume']}")

    out = {
        "experiment": "resume_vs_cascade",
        "equivalence_on_file_dags": {"trials": trials, "mismatches": mism,
                                     "note": "Nextflow -resume == hyppo plan() on file-DAGs"},
        "hidden_staleness": demo,
        "conclusion": ("On file-DAGs both cascade identically (±); hyppo's advantage is the "
                       "semantic derived_by graph, which catches staleness invisible to "
                       "file-content hashing."),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "resume_vs_cascade.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {DATA / 'resume_vs_cascade.json'}")


if __name__ == "__main__":
    main()

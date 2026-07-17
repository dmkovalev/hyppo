"""Масштабирование RL-материализации (owlrl, Java-free) на реплицированном
CRM-графе гипотез: узлы N = NI*NP + 7*NP + 1 (пара «нагнетательная-добывающая»).

Меряется время OWL 2 RL-замыкания (тот же движок, что в
hyppo.ontology.consistency._run_owlrl_stage_a) на ABox из Hypothesis-индивидов
с derived_by (TransitiveProperty). Каскад StaleHypothesis материализуется
аксиомой  StaleHypothesis ≡ Hypothesis ⊓ ∃derived_by.InvalidHypothesis.

Замер соответствует утверждению §sect4_wf_bo диссертации: RL-материализация
полиномиальна по числу утверждений ABox.
"""
from __future__ import annotations
import sys, time, argparse
sys.setrecursionlimit(1_000_000)

from owlready2 import destroy_entity
from hyppo.core._base import virtual_experiment_onto as onto, Hypothesis
import hyppo.ontology.core_rules as cr  # регистрирует TBox: StaleHypothesis, derived_by transitive


def build_abox(NI: int, NP: int):
    """Строит реплицированный граф derived_by и сеет инвалидацию пар инжектора 1.
    Возвращает число узлов."""
    for ind in list(onto.individuals()):
        destroy_entity(ind)
    with onto:
        P = {}  # (i,j) -> hyp
        C = {}; R = {}; M = {}; F = {}; K = {}; W = {}; B = {}
        for j in range(1, NP + 1):
            R[j] = Hypothesis(f"R_{j}"); M[j] = Hypothesis(f"M_{j}")
            B[j] = Hypothesis(f"B_{j}")
            for i in range(1, NI + 1):
                P[(i, j)] = Hypothesis(f"P_{i}_{j}")
        for j in range(1, NP + 1):
            C[j] = Hypothesis(f"C_{j}")
            C[j].derived_by = [P[(i, j)] for i in range(1, NI + 1)] + [R[j]]
            K[j] = Hypothesis(f"K_{j}"); K[j].derived_by = [B[j]]
            W[j] = Hypothesis(f"W_{j}"); W[j].derived_by = [K[j]]
            F[j] = Hypothesis(f"F_{j}"); F[j].derived_by = [C[j], M[j]]
        OPR = Hypothesis("OPR")
        OPR.derived_by = [F[j] for j in range(1, NP + 1)] + [W[j] for j in range(1, NP + 1)]
        # каскад: инвалидируем пары инжектора 1 (как benchmark_341)
        for j in range(1, NP + 1):
            P[(1, j)].is_a.append(cr.InvalidHypothesis)
    return NI * NP + 7 * NP + 1


def rl_closure_time(onto):
    """Тот же путь, что consistency._run_owlrl_stage_a: bridge->copy->expand->closure.
    Возвращает (seconds, n_triples_closed, n_stale)."""
    import owlrl
    from rdflib import Graph, RDF, URIRef
    from hyppo.ontology.consistency import _expand_all_different

    world = getattr(onto, "world", None) or onto
    bridge = world.as_rdflib_graph()
    # store='SimpleMemory': обход фатального сбоя rdflib 7.6 Memory-store под Python 3.13
    graph = Graph(store="SimpleMemory")
    for tr in bridge:
        graph.add(tr)
    _expand_all_different(graph)

    t0 = time.perf_counter()
    sem = owlrl.OWLRL_Semantics(graph, axioms=True, daxioms=True, rdfs=False)
    sem.closure()
    dt = time.perf_counter() - t0

    stale_uri = URIRef(onto.base_iri + "StaleHypothesis")
    n_stale = sum(1 for _ in graph.subjects(RDF.type, stale_uri))
    return dt, len(graph), n_stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="341:10x20,1051:28x30,10166:100x95",
                    help="список target:NIxNP через запятую")
    args = ap.parse_args()
    print(f"{'nodes':>8} {'NIxNP':>10} {'RL_closure_s':>13} {'triples':>10} {'stale':>7}")
    for spec in args.sizes.split(","):
        target, ninp = spec.split(":")
        NI, NP = (int(x) for x in ninp.split("x"))
        n = build_abox(NI, NP)
        dt, ntr, nst = rl_closure_time(onto)
        print(f"{n:>8} {ninp:>10} {dt:>13.3f} {ntr:>10} {nst:>7}", flush=True)


if __name__ == "__main__":
    main()

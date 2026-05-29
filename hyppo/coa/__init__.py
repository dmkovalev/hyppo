"""The :mod:`hyppo.coa` module gathers causal-ordering algorithms.

Two layers are importable without pulling sympy/owlready:

* :mod:`hyppo.coa.causal` -- pure Dulmage-Mendelsohn core (matching, SCC, closure);
* :class:`hyppo.coa.graph.HypothesisGraph` -- Algorithms 1/2/4 over a hypothesis DAG.

The richer :class:`hyppo.coa._base.Structure` (sympy-backed) is imported lazily.
"""
from hyppo.coa.graph import HypothesisGraph

__all__ = ["HypothesisGraph"]

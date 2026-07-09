from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from hyppo.core._base import virtual_experiment_onto

if TYPE_CHECKING:
    pass


with virtual_experiment_onto:

    class HypothesisLattice:
        def __init__(self, hypotheses, workflow):
            self.hypotheses = hypotheses
            self.workflow = workflow
            self.lattice = self.build_lattice()

        def build_lattice(self):
            """Algorithm 1: Build hypothesis lattice from hypothesis structures.

            An edge (h_i, h_j) is derived when the output variable of h_i
            appears among the input variables of h_j's equations (i.e. h_j is
            derived_by h_i); see :meth:`_build_hypothesis_var_mapping`.
            """
            lattice = nx.DiGraph()
            # check if all hypotheses are in workflow
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")

            for dep in self._build_hypothesis_var_mapping():
                lattice.add_edge(dep[0], dep[1])

            return lattice

        def add_hypothesis(self, hypothesis):
            """Algorithm 2: Add a hypothesis to an existing lattice.

            Registers the new hypothesis, then rebuilds variable-level
            transitive closure with it, and maps to hypothesis-level edges.
            """
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")
            if hypothesis not in self.hypotheses:
                self.hypotheses.append(hypothesis)
            self.lattice.add_node(hypothesis)

            for dep in self._build_hypothesis_var_mapping():
                self.lattice.add_edge(dep[0], dep[1])

        def derived_by(self, hypothesis):
            """Return hypotheses that are derived by the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            return set(self.lattice.predecessors(hypothesis))

        def _build_hypothesis_var_mapping(self):
            """Derive hypothesis-level edges from equation variables.

            Edge (h_i, h_j) added iff output variable of h_i appears among
            input variables of h_j (h_j depends on h_i).
            """
            dependencies = []
            for h_i in self.hypotheses:
                if not hasattr(h_i, "structure") or not h_i.structure.equations:
                    continue
                i_out = self._output_variable(h_i)
                if i_out is None:
                    continue
                for h_j in self.hypotheses:
                    if h_i is h_j:
                        continue
                    if not hasattr(h_j, "structure"):
                        continue
                    j_out = self._output_variable(h_j)
                    for eq in h_j.structure.equations:
                        if len(eq.vars) <= 1:
                            continue
                        for v in eq.vars:
                            if v != j_out and v == i_out:
                                dependencies.append((h_i, h_j))
            return dependencies

        @staticmethod
        def _output_variable(hypothesis):
            """Return the output variable of a hypothesis.

            If an equation was built from a formula string with an explicit
            left-hand side (``out = f(...)``), the output is the LHS symbol.
            Otherwise fall back to the exogenous-variable heuristic (first
            variable not declared by a single-variable equation).
            """
            from sympy import sympify

            for eq in hypothesis.structure.equations:
                formula = getattr(eq, "formula", None)
                if formula is not None and "=" in str(formula):
                    lhs = sympify(str(formula).split("=", 1)[0], locals={})
                    syms = sorted(lhs.free_symbols, key=lambda s: s.name)
                    if len(syms) == 1:
                        return syms[0]
            exo_vars = set()
            for e in hypothesis.structure.equations:
                if len(e.vars) == 1:
                    exo_vars.add(e.vars[0])
            for eq in hypothesis.structure.equations:
                if len(eq.vars) > 1:
                    candidates = [v for v in eq.vars if v not in exo_vars]
                    return candidates[0] if candidates else eq.vars[0]
            return None

        def competes(self, hypothesis):
            """Return hypotheses that compete with the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            # Competing hypotheses share predecessors without a direct relationship
            predecessors = set(self.lattice.predecessors(hypothesis))
            competitors = set()
            for h in self.hypotheses:
                if h != hypothesis and h not in self.derived_by(hypothesis):
                    h_predecessors = set(self.lattice.predecessors(h))
                    if predecessors & h_predecessors:  # If they share any predecessors
                        competitors.add(h)
            return competitors

        def impacts(self, hypothesis):
            """Return hypotheses that are impacted by the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            # Impacted hypotheses are those that are reachable from this hypothesis
            return set(nx.descendants(self.lattice, hypothesis))

        def remove_hypothesis(self, hypothesis):
            """Remove a hypothesis from the lattice."""
            if hypothesis in self.hypotheses:
                self.hypotheses.remove(hypothesis)
                self.lattice.remove_node(hypothesis)

        def _is_correct(self):
            """Check if all hypotheses are present in the workflow."""
            workflow_hypotheses = set()
            tasks = self.workflow.get_tasks()
            for task in tasks:
                workflow_hypotheses.update(task)

            return all(h in workflow_hypotheses for h in self.hypotheses)

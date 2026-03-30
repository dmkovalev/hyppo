from __future__ import annotations

from typing import TYPE_CHECKING

from hyppo.core._base import virtual_experiment_onto
import networkx as nx
from collections import defaultdict

if TYPE_CHECKING:
    from hyppo.coa._base import Structure, Equation


with virtual_experiment_onto:
    class HypothesisLattice:

        def __init__(self, hypotheses, workflow):
            self.hypotheses = hypotheses
            self.workflow = workflow
            self.lattice = self.build_lattice()


        def build_lattice(self):
            """Algorithm 1: Build hypothesis lattice from workflow tasks.

            For each pair of hypotheses (h_i, h_j) from different tasks:
            - Unite their structures S_i ∪ S_j
            - If the union is complete, build transitive closure
            - Use the closure to derive edges in the lattice
            """
            lattice = nx.DiGraph()
            # check if all hypotheses are in workflow
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")

            transitive_closure = defaultdict(set)

            tasks = self.workflow.get_tasks()
            for i, current_task in enumerate(tasks):
                remaining_tasks = tasks[i + 1:]
                for h_i in current_task:
                    for t in remaining_tasks:
                        for h_j in t:
                            structure_i = h_i.structure
                            structure_j = h_j.structure
                            united_str = structure_i.union(structure_j)
                            if united_str.is_complete():
                                tc = united_str.build_transitive_closure()
                                for key, descendants in tc.items():
                                    transitive_closure[key].update(descendants)

            hyp_var_map = self._build_hypothesis_var_mapping(transitive_closure)

            for dep in hyp_var_map:
                lattice.add_edge(dep[0], dep[1])

            return lattice

        def add_hypothesis(self, hypothesis):
            """Algorithm 2: Add a hypothesis to an existing lattice.

            For the added hypothesis h_add and each existing h_i:
            - If S_i ∪ S_add is complete, build transitive closure
            - Add derived edges to the existing lattice
            """
            # check if all hypotheses are in workflow
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")
            tasks = self.workflow.get_tasks()
            transitive_closure = defaultdict(set)

            for current_task in tasks:
                for h_i in current_task:
                    structure_i = h_i.structure
                    united_str = structure_i.union(hypothesis.structure)
                    if united_str.is_complete():
                        tc = united_str.build_transitive_closure()
                        for key, descendants in tc.items():
                            transitive_closure[key].update(descendants)

            hyp_var_map = self._build_hypothesis_var_mapping(transitive_closure)

            for dep in hyp_var_map:
                self.lattice.add_edge(dep[0], dep[1])


        def derived_by(self, hypothesis):
            """Return hypotheses that are derived by the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            return set(self.lattice.predecessors(hypothesis))

        def _build_hypothesis_var_mapping(self, transitive_closure):
            """
            Build mapping between hypotheses based on their variable relationships.

            Args:
                transitive_closure (defaultdict): Dictionary mapping variables to their
                    transitive closure relations (sets of reachable variables)

            Returns:
                list: List of tuples representing hypothesis dependencies (h1, h2)
                    where h1 depends on h2
            """
            dependencies = []
            for h1, relations1 in transitive_closure.items():
                for h2, relations2 in transitive_closure.items():
                    if h1 != h2:
                        # If all relations in h1 are contained in h2's relations, h1 depends on h2
                        if relations1.issubset(relations2):
                            dependencies.append((h1, h2))
            return dependencies

        def competes(self, hypothesis):
            """Return hypotheses that compete with the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            # Competing hypotheses are those that share predecessors but aren't in a direct relationship
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

from hyppo.core._base import virtual_experiment_onto
from hyppo.coa._base import Structure, Equation
import networkx as nx
from collections import defaultdict


with virtual_experiment_onto:
    class HypothesisLattice:

        def __init__(self, hypotheses, workflow):
            self.hypotheses = hypotheses
            self.workflow = workflow
            self.lattice = self.build_lattice()


        def build_lattice(self):
            lattice = nx.DiGraph()
            # check if all hypotheses are in workflow
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")

            transitive_closure = defaultdict(list)

            tasks = self.workflow.get_tasks()
            while (tasks):
                current_task = tasks.get_current()
                # remove current_task from tasks
                remaining_tasks = self.workflow.get_remaining(current_task)
                for h_i in current_task:
                    for t in remaining_tasks:
                        for h_j in t:
                            structure_i = h_i.structure
                            structure_j = h_j.structure
                            united_str = structure_i.union(structure_j)
                            if united_str.is_complete():
                                tc = united_str.build_transitive_closure()
                                transitive_closure.add(tc)

            hyp_var_map = self._build_hypothesis_var_mapping(transitive_closure)

            for dep in hyp_var_map:
                lattice.add_edge(dep[0], dep[1])

            return lattice

        def add_hypothesis(self, hypothesis):
            # check if all hypotheses are in workflow
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")
            tasks = self.workflow.get_tasks()
            transitive_closure = defaultdict(list)

            while (tasks):
                current_task = tasks.get_current()
                # remove current_task from tasks
                for h_i in current_task:
                    structure_i = h_i.structure
                    united_str = structure_i.union(hypothesis.structure)
                    if united_str.is_complete():
                        tc = united_str.build_transitive_closure()
                        transitive_closure.add(tc)

            hyp_var_map = self._build_hypothesis_var_mapping(transitive_closure)

            for dep in hyp_var_map:
                self.lattice.add_edge(dep[0], dep[1])


        def derived_by(self, hypothesis):
            pass

        def _build_hypothesis_var_mapping(self, transitive_closure):
            pass


        def competes(self, hypothesis):
            pass

        def impacts(self, hypothesis):
            pass


        def remove_hypothesis(self, hypothesis):
            pass

        def _is_correct(self):
            pass




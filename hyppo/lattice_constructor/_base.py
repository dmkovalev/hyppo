from hyppo.core._base import virtual_experiment_onto
from hyppo.coa._base import Structure, Equation
import networkx as nx

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

            transitive_closure = None
            causal_mapping = None

            tasks = self.workflow.get_tasks()
            while (tasks):
                current_task = tasks.get_current()
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
                lattice.add_edge(hyp_var_map[0], hyp_var_map[1])

            return lattice
        def derived_by(self, hypothesis):
            pass

        def _build_hypothesis_var_mapping(self, transitive_closure):
            pass


        def competes(self, hypothesis):
            pass

        def impacts(self, hypothesis):
            pass


        def add_hypothesis(self):
            pass


        def remove_hypothesis(self):
            pass

        def _is_correct(self):
            pass




from hyppo.core._base import virtual_experiment_onto
from hyppo.coa._base import Structure, Equation

with virtual_experiment_onto:
    class HypothesisLattice:

        def __init__(self):
            pass

        def derived_by(self, hypothesis):
            pass

        def competes(self, hypothesis):
            pass

        def impacts(self, hypothesis):
            pass


        def add_hypothesis(self):
            pass


        def remove_hypothesis(self):
            pass


        def build_lattice(self, hypotheses, workflow):
            pass

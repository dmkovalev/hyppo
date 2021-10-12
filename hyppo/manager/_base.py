from owlready2 import *
from hyppo.core._base import virtual_experiment_onto

virtual_experiment_onto = get_ontology("https://synthesis.ipi.ac.ru/manager.owl")
namespace = virtual_experiment_onto.get_namespace()

with virtual_experiment_onto:
    namespace = namespace

    class Manager(virtual_experiment_onto.Artefact):
        def _build_lattice(self):
            pass

        def _make_plan(self):
            pass

        def run(self):
            # build lattice
            hypothesis_lattice = self._build_lattice()

            # form plan
            execution_plan = self._make_execution_plan()

            # run
            pass

    class has_for_virtual_experiment(Manager >> virtual_experiment_onto.VirtualExperiment):
        python_name = 'virtual_experiment'

    class has_for_execution_plan(Manager >> ExecutionPlan):
        python_name = 'execution_plan'

    class has_for_lattice(Manager >> HypothesisLattice):
        python_name = 'hypothesis_lattice'

    class has_for_runner(Manager >> Runner):
        python_name = 'runner'

    class has_for_domain_ontology(Manager >> Ontology):
        python_name = 'domain_ontology'

    class has_for_autogenerate(Manager >> bool, DataProperty, FunctionalProperty): pass
    class has_for_abandon_rules(Manager >> Rule): pass
    class has_for_correlation(Manager >> Correlation): pass
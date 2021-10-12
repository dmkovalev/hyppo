from owlready2 import *
from hyppo.core._base import virtual_experiment_onto

manager_onto = get_ontology("https://synthesis.ipi.ac.ru/manager.owl")

with manager_onto:
    class Manager(Thing): pass
    class has_for_experiment(Manager >> virtual_experiment_onto.VirtualExperiment): pass
    class has_for_execution_plan(Manager >> ExecutionPlan): pass
    class has_for_lattice(Manager >> HypothesisLattice): pass
    class has_for_runner(Manager >> Runner): pass
    class has_for_domain_ontology(Manager >> Ontology): pass
    class has_for_autogenerate(Manager >> bool, DataProperty, FunctionalProperty): pass



def delete(ve, id, type):
    ve.delete(type, id)
    return


def modify(ve, type, id, specification):
    ve.modify(type, id, specification)
    return

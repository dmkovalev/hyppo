import datetime

# from owlready2 import get_ontology, Thing, DataProperty, FunctionalProperty, ObjectProperty, TransitiveProperty, \
#     AllDisjoint

from owlready2 import *

virtual_experiment_onto = get_ontology("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
hcp_brain_onto = get_ontology("http://synthesis.ipi.ac.ru/hcp_brain_onto.owl")
virtual_experiment_onto.imported_ontologies.append(hcp_brain_onto)

with virtual_experiment_onto:
    # define base class and its properties
    class Artefact(Thing): pass
    class Specification(Thing): pass
    class has_for_id(Artefact >> int, DataProperty, FunctionalProperty): pass
    class has_for_name(Artefact >> str, DataProperty, FunctionalProperty): pass
    class has_for_description(Artefact >> str, DataProperty, FunctionalProperty): pass
    class has_for_authors(Artefact >> str, DataProperty): pass
    class has_for_createdate(Artefact >> datetime.datetime, DataProperty, FunctionalProperty): pass
    class has_for_lastupdate(Artefact >> datetime.datetime, DataProperty, FunctionalProperty): pass
    class has_for_specification(Artefact >> Specification): pass
    class Artefact(Thing):
        is_a = [has_for_authors.min(1)]
        is_a = [has_for_name.exactly(1)]
        is_a = [has_for_description.exactly(1)]
        is_a = [has_for_id.exactly(1)]
        is_a = [has_for_lastupdate.exactly(1)]
        is_a = [has_for_createdate.exactly(1)]
        is_a = [has_for_specification.exactly(1)]

    class Hypothesis(Artefact): pass
    class Model(Artefact): pass
    # class Mapping(Artefact): pass
    # class Relation(Artefact): pass

    # TODO probability > 0.0 and < 1.0
    class has_for_probability(Hypothesis >> float,
                              DataProperty, FunctionalProperty): pass
    class is_implemented_by_model(Hypothesis >> Model): class_property_type = ["some"]
    class refers_to_hypothesis(ObjectProperty):
        domain              = [Model]
        range               = [Hypothesis]
        inverse_property    = is_implemented_by_model
        class_property_type = ["only"]

    class competes(Hypothesis >> Hypothesis, TransitiveProperty, SymmetricProperty): pass
    class derived_by(Hypothesis >> Hypothesis, TransitiveProperty, AsymmetricProperty): pass
    class impacts(ObjectProperty, TransitiveProperty):
        domain              = [Hypothesis]
        range               = [Hypothesis]
        inverse_propert     = derived_by

    class VirtualExperiment(Artefact): pass
    class Configuration(Artefact): pass
    class Workflow(Artefact): pass
    class Task(Thing): pass

    class has_for_hypothesis(VirtualExperiment >> Hypothesis): class_property_type = ["some"]
    class has_for_model(VirtualExperiment >> Model): class_property_type = ["some"]
    class has_for_workflow(VirtualExperiment >> Workflow): class_property_type = ["only"]
    class has_for_configuration(VirtualExperiment >> Configuration): class_property_type = ["some"]
    class has_for_task(Workflow >> Task): class_property_type = ["some"]

    AllDisjoint([VirtualExperiment, Configuration, Workflow, Hypothesis, Model, Task])


if __name__ == '__main__':
    virtual_experiment_onto = get_ontology("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
    print(list(virtual_experiment_onto.classes()))
    art = Artefact("123")
    art.has_for_author = [123]
    print(has_for_authors.range)
    print(art.name)
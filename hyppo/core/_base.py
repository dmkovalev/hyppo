from abc import abstractmethod, ABCMeta
from owlready2 import *

ve = get_ontology("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
with ve:

    # define base class and its properties
    class Artefact(Thing): pass
    class has_for_id(Artefact >> int, DataProperty, FunctionalProperty): pass
    class has_for_name(Artefact >> str, DataProperty, FunctionalProperty): pass
    class has_for_description(Artefact >> str, DataProperty, FunctionalProperty): pass
    class has_for_authors(Artefact >> str, DataProperty): pass
    class has_for_createdate(Artefact >> datetime.datetime, DataProperty, FunctionalProperty): pass
    class has_for_lastupdate(Artefact >> datetime.datetime, DataProperty, FunctionalProperty): pass
    class Artefact(Thing):
        is_a = [has_for_authors.min(1)]
        is_a = [has_for_name.exactly(1)]
        is_a = [has_for_description.exactly(1)]
        is_a = [has_for_id.exactly(1)]
        is_a = [has_for_lastupdate.exactly(1)]
        is_a = [has_for_createdate.exactly(1)]

    class Hypothesis(Artefact): pass
    class Model(Artefact): pass

    # TODO probability > 0.0 and < 1.0
    class has_for_probability(Hypothesis >> float,
                              DataProperty, FunctionalProperty): pass
    class is_implemented_by_model(Hypothesis >> Model): class_property_type = ["some"]
    class refers_to_hypothesis(ObjectProperty):
        domain              = [Model]
        range               = [Hypothesis]
        inverse_property    = is_implemented_by_model
        class_property_type = ["only"]

    class competes(Hypothesis >> Hypothesis, TransitiveProperty): pass
    class derived_by(Hypothesis >> Hypothesis, TransitiveProperty): pass
    class impacts(ObjectProperty, TransitiveProperty):
        domain              = [Hypothesis]
        range               = [Hypothesis]
        inverse_propert     = derived_by

    class has_for_implementation(Model >> Model): pass

    class VirtualExperiment(Artefact):
        pass

    class Configuration(Artefact):
        pass

    class Workflow(Artefact):
        pass




if __name__ == '__main__':
    onto = get_ontology("http://synthesis.ipi.ac.ru/onto.owl")
    print(list(onto.classes()))
    art = Artefact("123")
    art.has_for_author = [123]
    print(has_for_authors.range)
    print(art.name)
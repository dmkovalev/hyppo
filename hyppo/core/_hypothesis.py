from owlready2 import get_ontology

ve = get_ontology()

class PHypothesis(Hypothesis):
    def __init__(self):
        self.probability = None
        self.article = None
        self._models = list()
        self._parameters = dict()

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        self._models = value

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @parameters.getter
    def parameters(self):
        return self._parameters

    def _parse_specification(self):
        pass


class Model(Artefact):
    def __init__(self, specification=None):
        # self.spec = self._parse_specification(specification)
        pass


if __name__ == '__main__':
    h = Hypothesis()

    h.parameters = {1: 2}
    h.id = 1
    h._parameters = 1
    print(h.parameters)
    relation = Relation

    m = Model()
    m.id = 2
    relation.add_mapping(h, m)
    m.id = 3
    relation.add_mapping(h, m)
    m.id = 2

    relation.delete_model(m)

    print(relation.get_mapping())

    relation.delete_hypothesis(h)
    print(relation.get_mapping())

from hyppo.core._base import Artefact


class Relation:
    _mapping = dict()

    @classmethod
    def add_mapping(cls, hypothesis, model):
        assert isinstance(hypothesis, Hypothesis), 'Not a hypothesis'
        assert isinstance(model, Model), 'Not a model'

        if hypothesis.id in cls._mapping.keys():
            raise ValueError('Hypothesis id is already in use')
        else:
            cls._mapping[hypothesis.id] = model.id

    @classmethod
    def delete_mapping(cls):
        pass

    @classmethod
    def get_mapping(cls):
        return cls._mapping

class Hypothesis(Artefact):
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
        self.spec = self._parse_specification(specification)


if __name__ == '__main__':
    h = Hypothesis()

    h.parameters = {1:2}
    h.id = 1
    h._parameters = 1
    print(h.parameters)

    relation = Relation
    relation.add_mapping(h, h)
    print(relation.get_mapping())

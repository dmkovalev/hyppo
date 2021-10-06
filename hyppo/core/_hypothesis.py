from hyppo.core._base import virtual_experiment_onto


class Hypothesis(virtual_experiment_onto.Artefact):
    namespace = virtual_experiment_onto.get_namespace("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
    # def __init__(self):
    #     self._parameters = dict()

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

class Model(virtual_experiment_onto.Artefact):
    namespace = virtual_experiment_onto.get_namespace("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
    def __init__(self):
        pass


if __name__ == '__main__':
    h = Hypothesis()
    h.has_for_name = 'first'
    h1 = Hypothesis()
    h.has_for_probability = 0.0
    h1.has_for_name = 'second'
    h2 = Hypothesis()
    h2.has_for_name = 'third'

    h.competes = [h1, h2]

    m = Model()


    print([i.has_for_name for i in h.competes])
    print([i.has_for_name for i in h1.INDIRECT_competes])

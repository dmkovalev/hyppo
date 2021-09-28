from abc import abstractmethod, ABCMeta

class Artefact():
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @property
    def specification(self):
        return self._specification

    def _parse_specification(self):
        pass


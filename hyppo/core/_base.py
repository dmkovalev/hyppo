from abc import abstractmethod, ABCMeta


class Artefact():
    __metaclass__ = ABCMeta

    def __init__(self):
        self.id = None
        self.name = None
        self.description = None
        self.author = None
        self.create_date = None
        self.last_update = None

    @property
    def specification(self):
        return self._specification

    @abstractmethod
    def _parse_specification(self):
        pass



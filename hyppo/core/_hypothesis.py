from ._base import Artefact

class Hypothesis(Artefact):
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = defaultdict(list)  # default dictionary to store graph
        self.real_V_names = []

    def __getattr__(self, item):
        return self.graph[item]

    def __repr__(self):
        return str(self.graph)


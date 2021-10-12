

class Configuration:
    def __init__(self):
        self.mapping = None


class Workflow(Artefact):
    self.G = Graph(connection_matrix.shape[1])
    self.G.real_V_names = connection_matrix.columns.values
    connection_matrix.columns = ["H" + str(k) for k in range(G.V)]
    connection_matrix.index = ["H" + str(k) for k in range(G.V)]

    for col in self.connection_matrix.columns:
        self.connected = self.connection_matrix.index[connection_matrix[col] == 1]
        sself.tart = int(col.split("H")[1])
        for k in range(len(self.connected)):
            self.G.addEdge(start, int(connected[k].split("H")[1]))
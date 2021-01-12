from collections import defaultdict
import itertools


# This class represents a directed graph using adjacency list representation
class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = defaultdict(list)  # default dictionary to store graph
        self.real_V_names = []

    def __getattr__(self, item):
        return self.graph[item]

    def __repr__(self):
        return str(self.graph)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

        # Use BFS to check path between s and d

    def isReachable(self, s, d):
        # Mark all the vertices as not visited
        visited = [False] * (self.V)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        while queue:

            # Dequeue a vertex from queue
            n = queue.pop(0)

            # If this adjacent node is the destination node,
            # then return true
            if n == d:
                return True

            #  Else, continue to do BFS
            for i in self.graph[n]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        # If BFS is complete without visited d
        return False


#
def create_workflow_graph(connection_matrix):
    G = Graph(connection_matrix.shape[1])
    G.real_V_names = connection_matrix.columns.values
    connection_matrix.columns = ["H" + str(k) for k in range(G.V)]
    connection_matrix.index = ["H" + str(k) for k in range(G.V)]

    for col in connection_matrix.columns:
        connected = connection_matrix.index[connection_matrix[col] == 1]
        start = int(col.split("H")[1])
        for k in range(len(connected)):
            G.addEdge(start, int(connected[k].split("H")[1]))

    return G


def find_connected_H(G):
    print(G)
    vertices = G.graph.keys()
    pairs = itertools.combinations(vertices, 2)

    connected_pairs = []
    for pair in pairs:
        print(pair)
        a = G.isReachable(pair[0], pair[1])
        if a:
            connected_pairs.append((G.real_V_names[pair[0]], G.real_V_names[pair[1]]))
            print(pair, "connected")

    return connected_pairs

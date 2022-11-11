from hyppo.core._base import virtual_experiment_onto
from utils import check_completeness
from utils import is_structure
from utils import save_default_names
from utils import set_default_names
from CreateGraph import Hypothesis_encoding

with virtual_experiment_onto:
    class HypothesisLattice:
        def find_connected_hypothetis(data, H_list):
            connected_hypothesis = []

            for i in range(1, len(H_list)):
                H1 = H_list[i][0]
                H2 = H_list[i][1]
                print("Check pair: ", (H1, H2))

                connected_h, is_connected = concat_H(data.copy(), H1, H2)
                if is_connected:
                    connected_hypothesis.append((H1, H2))

            return connected_hypothesis


        def concat_H(datafile, H1, H2):
            h1 = datafile[H1].copy()
            h2 = datafile[H2].copy()

            # comb = pd.concat((h1, h2), sort=True).fillna(0).astype(int)
            comb = h1.merge(h2, left_index=True, right_index=True, how='outer',
                            sort=False, suffixes=('_1', '_2')).fillna(0).astype(int)

            comb = comb.reindex(index=comb.index.to_series().str.rsplit('F').str[-1].astype(
                int).sort_values().index)
            print(comb)

            if is_structure(comb) and check_completeness(comb):
                E, V, def_data = save_default_names(comb)
                fd_set = Hypothesis_encoding(def_data)
                fd_set = set_default_names(fd_set, V)

                return _drop_duplicated_V(fd_set)
            else:
                print("Incomplete matrix passed")
                return None


        def single_hypothesis_graph(datafile, H):
            Hyp = datafile[H]

            if is_structure(Hyp) and check_completeness(Hyp):
                E, V, def_data = save_default_names(Hyp)
                fd_set = Hypothesis_encoding(def_data)
                fd_set = set_default_names(fd_set, V)
                return fd_set


        def _drop_duplicated_V(V_set):
            is_duplicated = False
            for i in range(len(V_set)):
                v = V_set[i]
                for j in range(len(v)):
                    if isinstance(v[j], str):
                        if len(v[j].split('_')) == 2:
                            # print("Replace ", v[j])
                            v[j] = v[j].split('_')[0]
                            is_duplicated = True
                    elif isinstance(v[j], list):
                        if len(v[j][0].split('_')) == 2:
                            # print("Replace ", v[j])
                            v[j] = [v[j][0].split('_')[0]]
                            is_duplicated = True
            return V_set, is_duplicated

        def add_hypothesis_to_lattice(self):
            pass



        def construct_lattice(hypotheses, workflow):

            for h in hypotheses:
                graph = single_hypothesis_graph(data, h)
                H_to_dot(graph, output_name=f"{h}_casual_graph", folder=save_folder)

            WF = create_workflow_graph(workflow)
            connected_by_wf = find_connected_H(WF)
            print(connected_by_wf)

            connected_hypothesis = find_connected_hypothetis(data, connected_by_wf)
            print(connected_hypothesis)
            H_list_to_dot(connected_hypothesis, output_name=workflow_file, folder=save_folder)
            python_to_dot(connected_h, output_name="H1H2", folder='./output/')

            return connected_hypothesis

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

from utils import check_completeness
from utils import is_structure
from utils import save_default_names
from utils import set_default_names
from CreateGraph import Hypothesis_encoding


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
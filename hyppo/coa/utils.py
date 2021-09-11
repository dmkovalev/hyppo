import itertools
import os


def is_structure(matrix):
    cond1 = matrix.shape[0] <= matrix.shape[1]
    if not cond1:
        return False

    cond2 = True
    for k in range(1, matrix.shape[0] + 1):
        all_sets = set(itertools.combinations(matrix.index.values, k))
        for s in all_sets:
            s = list(s)
            V = matrix.loc[s].sum(axis=0).values
            V = V[V != 0]
            if len(V) < k:
                cond2 = False
                break
    return cond1 and cond2


def check_completeness(matrix):
    if matrix.shape[0] == matrix.shape[0]:
        return True
    else:
        return False


def save_default_names(data):
    vars = list(data.columns.values)
    equations = list(data.index.values)
    df = data.copy()
    df.index = ["F" + str(k) for k in range(data.shape[0])]
    df.columns = ["x" + str(k) for k in range(data.shape[0])]
    return dict(zip(df.index.values, equations)), \
           dict(zip(df.columns.values, vars)), df


def set_default_names(fd_set, V):
    for i in range(len(fd_set)):
        combination = fd_set[i]
        for j in range(len(combination)):
            if isinstance(combination[j], str):
                combination[j] = V[combination[j]]
            elif isinstance(combination[j], list):
                combination[j] = [V[combination[j][0]]]
    return fd_set


def H_to_dot(casual_map, output_name="Direct_Casual_Graph", folder='./'):
    run_command = f"dot -Tps {folder}{output_name}.dot -o {folder}{output_name}.pdf"

    def create_body(casual_map):
        """
        vertex_name [shape=circle]
        A -> B [style=solid, color=black]


        :param casual_map:
        :return:
        """

        body_string = ""
        for element in casual_map:
            B = element[-1]
            for a in element[:-1]:
                if not isinstance(a, int):
                    body_string += f"{a} -> {B[0]} [style=solid, color=black]\n"
                else:
                    body_string += f"{B} [shape=circle]\n"

        return body_string

    body = create_body(casual_map)

    with open(folder + output_name + '.dot', 'w') as run_file:
        run_file.writelines("digraph " + output_name + '\t{\n')
        run_file.writelines(body)
        run_file.writelines("\n}")

    os.system(run_command)


def H_list_to_dot(casual_map, output_name="H_Casual_Graph", folder='./'):
    run_command = f"dot -Tps {folder}{output_name}.dot -o {folder}{output_name}.pdf"

    def create_body(casual_map):
        """
        vertex_name [shape=circle]
        A -> B [style=solid, color=black]


        :param casual_map:
        :return:
        """

        body_string = ""
        for element in casual_map:
            body_string += f"{element[0]} -> {element[1]} [style=solid, color=black]\n"

        return body_string

    body = create_body(casual_map)

    with open(folder + output_name + '.dot', 'w') as run_file:
        run_file.writelines("digraph " + output_name + '\t{\n')
        run_file.writelines(body)
        run_file.writelines("\n}")

    os.system(run_command)

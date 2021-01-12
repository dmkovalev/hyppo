import re
from TexSoup import TexSoup
import numpy as np
import pandas as pd


def tex_to_mathml():
    file = open("math_input", 'r')
    latex = file.read()
    file.close()
    print(latex)
    mathml_output = latex2mathml.converter.convert(latex)
    print(mathml_output)

    with open("mathml_system.xml", 'w') as file:
        file.writelines(mathml_output)
        file.close()


class TexParser:

    def __init__(self, path_to_file):
        self.path = path_to_file
        file = open(path_to_file, 'r')
        self.tex = file.read()
        file.close()

    def __call__(self):
        return self.parse()

    def parse(self):
        tex = TexSoup(self.tex)
        equations = list(tex.find_all('equation'))

        hypothesis = re.findall("H_{(.*?)}", str(equations))
        h_array = ["H" + h for h in hypothesis]
        output = dict(zip(h_array, [None] * len(h_array)))

        for i in range(len(h_array)):
            eq = str(equations[i]).replace(" ", '').replace('\n', '')

            system = re.findall("begin{array}{lc}(.*?)end{array}", str(eq), re.S)[0]
            # print(system)
            functions = ["F" + f for f in re.findall("f_{(.*?)}", str(system))]
            variables = ["x" + v for v in np.unique(re.findall("x_{(.*?)}", str(system)))]
            f_index = [int(f) for f in re.findall("f_{(.*?)}", str(system))]
            v_index = [int(v) for v in np.unique(re.findall("x_{(.*?)}", str(system)))]

            h_df = pd.DataFrame(np.zeros((len(variables), len(functions))).astype(int),
                                index=functions, columns=variables)

            for k in range(len(functions)):
                # print(system.split(',\\')[k])
                f = np.unique(re.findall("x_{(.*?)}", str(system.split(',\\')[k])))
                # print(f)
                for v in f:
                    h_df.loc['F' + str(f_index[k])]['x' + str(v)] = 1

            output["H" + str(i + 1)] = h_df
        return output

import os
import pandas
from DFS import create_workflow_graph, find_connected_H
from HypothesisConnection import find_connected_hypothetis
from HypothesisConnection import single_hypothesis_graph
from MathMlParser import TexParser
from utils import H_list_to_dot, H_to_dot

file = "test_latex.tex"
workflow_file = 'ExampleInput'
save_folder = "output/example/"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


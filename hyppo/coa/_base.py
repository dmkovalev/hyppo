import os
import itertools
import pandas
import numpy as np
import scipy.stats
from latex2sympy2 import latex2sympy, latex2latex
import sympy
from owlready2 import *
from latex2sympy import strToSympy


# np.random.seed(3)
from owlready2 import get_ontology

from hyppo.core._base import virtual_experiment_onto, Variable
from sympy import Function, Symbol

with virtual_experiment_onto:
    class Equation(Thing):
        namespace = virtual_experiment_onto.get_namespace(
            "http://synthesis.ipi.ac.ru/virtual_experiment.owl")

        def __init__(self, formula):
            self.formula = formula
            self.get_vars()
            self.get_equation()

        def get_vars(self):
            '''

            Args:
                latex_str: equation defined in latex

            Returns:
                set of free symbols
            '''
            equation = strToSympy(self.formula)
            self.vars = sorted(equation.free_symbols, key=lambda symbol: symbol.name)
            return self.vars

        def get_equation(self):
            self.equation = strToSympy(self.formula)
            return self.equation

    class has_for_formula(Equation >> str, FunctionalProperty):
        class_property_type = ["only"]
        python_name = "formula"

    class Structure(virtual_experiment_onto.Artefact):
        namespace = virtual_experiment_onto.get_namespace(
            "http://synthesis.ipi.ac.ru/virtual_experiment.owl")

        def __init__(self, equations):
            self.equations = equations
            # vars = [eq.get_vars() for eq in self.equations]
            self.vars = list(set().union(*(map(lambda x: set(x.vars), self.equations))))

        def is_complete(self):
            return len(self.equations) == len(self.vars)

        def build_transitive_closure(self):
            pass

        def build_full_causal_mapping(self):
            pass

        def join(self):

            pass

        def build_matrix(self) -> np.matrix:
            '''
            converts equations and variables from structure into matrix
            Returns:
                    matrix of values or None
            '''
            if not self.is_complete():
                raise Exception("Structure is not complete")
            else:
                matrix = np.matrix(np.zeros((len(self.vars), len(self.vars))))
                for i in range(len(self.vars)):
                    for j in range(len(self.equations)):
                        if self.vars[i] in self.equations[j].vars:
                            # print(self.equations[j].vars)
                            matrix[j, i] = 1
            return matrix


        # def remove_subsets(self, subset):
        #     for s in np.ravel(subset):
        #         try:
        #             cols_remove = np.argwhere(S[s] != 0)[0][1]
        #         except IndexError:
        #             # continue
        #             cols_remove = np.argwhere(S[s] != 0)
        #         S = np.delete(S, cols_remove, axis=1)
        #     S = np.delete(S, subset, axis=0)
        #     return S
        #
        #
        # def find_minimal_subsets(S, subsets_cols=[]):
        #     if matrix is empty:
        #         return final result
        #     if S.shape[0] == 0:
        #         return subsets_cols
        #
        #     for i in range(S.shape[0]):
        #
        #         all_subsets = set(itertools.combinations(range(S.shape[0]), i + 1))
        #         found_subset = []
        #
        #         for subset in all_subsets:
        #             subset = [int(s) for s in subset]
        #             s = S[subset]
        #             abs_sum = np.sum(np.amax(s, axis=0))
        #
        #             if abs_sum == i + 1:
        #                 found_subset.append(subset)
        #
        #         if len(found_subset) != 0:
        #             # delete and stop
        #
        #             found_subset = [item for sublist in found_subset for item in sublist]
        #             subsets_cols.append(found_subset)
        #
        #             # index = index - set(found_subset)
        #             S = remove_subsets(S, found_subset)
        #             break
        #
        #     return find_minimal_subsets(S, subsets_cols)
        #
        #
        # def map_subsets(subsets_cols):
        #     N = len([item for sublist in subsets_cols for item in sublist])
        #     initial_index = np.array(list(range(N)))
        #     mapped_index = []
        #     for subset in subsets_cols:
        #         # print(initial_index, subset)
        #         mp = initial_index[subset]
        #         mapped_index.append(tuple(mp))
        #         initial_index = np.delete(initial_index, subset)
        #     return mapped_index
        #
        #
        # def COA_step(data):
        #     # increment = min([int(i.split('F')[1]) for i in data.index.values])
        #     S = np.asmatrix(data.values)
        #     subsets = find_minimal_subsets(S, subsets_cols=[])
        #     subsets = map_subsets(subsets)
        #
        #     def step(subset):
        #         phi = []
        #         Sc = []
        #
        #         for s in subset:
        #             Sc.append(s)
        #             V = np.asarray(["x" + str(v) for v in s])
        #             Se = np.asarray(["F" + str(v) for v in s])
        #             # print(V)
        #             for f in Se:
        #                 x = V[np.random.randint(0, len(V))]
        #                 phi += [(f, x,)]
        #                 V = V[V != x]
        #         return Sc, phi
        #
        #     Sc, phi = step(subsets)
        #
        #     T = set(subsets).difference(set(Sc))
        #
        #     if len(T) == 0:
        #         return phi
        #     else:
        #         return phi + step(T)
        #
        # def find_correlations(graph, dataset, threshold=0.7):
        #     correlated_vars = []
        #     variables = graph.variables
        #     for var_i in variables:
        #         for var_j in variables:
        #             if var_i != var_j and not _connected(var_i, var_j):
        #                 if scipy.stats.pearsonr(dataset[var_i], dataset[var_j])[0] > threshold:
        #                     correlated_vars.append((var_i, var_j))
        #     return correlated_vars
        #
        #
        # def Hypothesis_encoding(data):
        #     Sum = []
        #     phi = COA_step(data.copy())
        #     increment = min([int(i.split('F')[1]) for i in data.index.values])
        #     variables = data.columns
        #     print(data)
        #     print(phi)
        #     for p in phi:
        #         k = int(p[0].split('F')[1])
        #         l = int(p[1].split('x')[1])
        #
        #         A = data.loc["F" + str(k)]
        #         Z = A[A == 1].index.values
        #
        #         if len(Z) == 1:
        #             update = [l, p[1]]
        #             Sum.append(update)
        #         else:
        #             update = list(set(Z).difference({p[1]}.union({k})))
        #             update.append([p[1]])
        #             Sum.append(update)
        #
        #     return Sum

if __name__ == '__main__':
    virtual_experiment_onto = get_ontology("http://synthesis.ipi.ac.ru/virtual_experiment.owl")
    # print(list(virtual_experiment_onto.classes()))
    # var = Variable(name='x1')
    # virtual_experiment_onto.save("ve.owl")
    # art = Artefact("123")
    # art.has_for_author = [123]
    # print(has_for_authors.range)
    # print(art.name)


    tex1 = r"x_1+x_2+x_3 =0"
    tex2 = r"x_1 + 6*x_2=0"
    tex3 = r"f(x_2,x_3)=0"

    e1 = Equation(formula=tex1)
    e2 = Equation(formula=tex2)
    e3 = Equation(formula=tex3)

    equations = [e1, e2, e3]
    s = Structure(equations)

    # s.has_for_equation = equations

    # all_vars = set().union(*(map(lambda x: set(x.vars), equations)))
    # s.vars = all_vars

    print(s.vars, s.is_complete(), s.build_matrix())


    # print(e.vars, e.equation)
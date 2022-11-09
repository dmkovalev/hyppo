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

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


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

        def __init__(self, equations, vars=None):
            self.equations = equations
            # vars = [eq.get_vars() for eq in self.equations]
            if vars is not None:
                self.vars = vars
            else:
                self.vars = set().union(*(map(lambda x: set(x.vars), self.equations)))

            # if not self.is_structure():
            #     raise Exception('Not a structure')

        def is_structure(self):
            is_structure = True
            all_subsets = powerset(self.equations)
            for subset in all_subsets:
                if subset:
                    eq_num = len(subset)
                    vars = set().union(*(map(lambda x: set(x.vars), subset)))
                    var_num = len(vars)
                    if eq_num > var_num:
                        is_structure = False
                    else:
                        #TODO part b
                        pass
            return is_structure

        def is_complete(self):
            return len(self.equations) == len(self.vars)

        def is_minimal(self):
            return self.is_complete() and not self.find_minimal_structures()



        def exogenous(self):
            '''
              get all exogenous variables in a structure
            Returns:
                set of exogenous variables
           '''
            exogenous = set()
            for eq in self.equations:
                if len(eq.get_vars()) == 1:
                    exogenous = exogenous.union(eq.get_vars())
            return exogenous

        def endogenous(self):
            '''
             get all endogenous variables in a structure
            Returns:
                set of endogenous variables
            '''
            return self.vars.difference(self.exogenous())

        def build_transitive_closure(self):
            pass

        def build_full_causal_mapping(self):
            if not self.is_complete():
                raise Exception('Structure is not complete')
            else:
                fcm = {}
                minimal_structures = self.find_minimal_structures()
                for min_str in minimal_structures:
                    sorted_vars = sorted(min_str.vars, key=lambda x: x.name)
                    for eq in min_str.equations:
                        fcm[eq.formula] = sorted_vars[0]
                        sorted_vars = [x for x in sorted_vars if x != sorted_vars[0]]

                left_structures = self.difference(set(minimal_structures))
                if minimal_structures and left_structures.equations:
                    fcm.update(left_structures.build_full_causal_mapping())

                if self.is_minimal():
                    sorted_vars = sorted(self.vars, key=lambda x: x.name)
                    for eq in self.equations:
                        fcm[eq.formula] = list(self.vars)[0]
                        sorted_vars = [x for x in sorted_vars if x != list(self.vars)[0]]

                return fcm



        def find_minimal_structures(self):
            '''

            Returns:

            '''
            min_str = []

            if len(self.equations) == 1 and self.is_complete():
                return min_str

            all_subsets = powerset(self.equations)
            for subset in list(all_subsets)[1:-1]:
                s = Structure(subset)
                if s.is_complete():
                    min_str.append(s)

            return min_str

        def union(self):
            pass

        def difference(self, set_structures):

            set_eq = [s.equations for s in set_structures]
            set_eq = list(itertools.chain(*set_eq))

            set_vars = [s.vars for s in set_structures]
            set_vars = list(itertools.chain(*set_vars))
            left_equations = [eq for eq in self.equations if eq not in set_eq]
            left_variables = set([v for v in self.vars if v not in set_vars])

            return Structure(equations=left_equations, vars=left_variables)

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
                # sorting of set is needed to preserve the order
                sorted_vars = sorted(self.vars, key=lambda x: x.name)
                for i in range(len(self.vars)):
                    for j in range(len(self.equations)):
                        if sorted_vars[i] in self.equations[j].vars:
                            matrix[j, i] = 1
            return matrix


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
    tex3 = r"f(x_2, x_1)=0"

    e1 = Equation(formula=tex1)
    e2 = Equation(formula=tex2)
    e3 = Equation(formula=tex3)

    equations = [e1, e2, e3]
    s = Structure(equations)

    # s.has_for_equation = equations

    # all_vars = set().union(*(map(lambda x: set(x.vars), equations)))
    # s.vars = all_vars

    # print(s.is_structure(), s.vars, s.is_complete(), s.build_matrix())
    # print(s.is_minimal())

    # print(s.exogenous(), s.endogenous())
    # print(e.vars, e.equation)
    print(s.build_full_causal_mapping())
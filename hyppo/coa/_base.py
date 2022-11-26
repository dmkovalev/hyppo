from itertools import combinations

import numpy as np
from latex2sympy import strToSympy
from owlready2 import *
from owlready2 import get_ontology
from hyppo.core._base import virtual_experiment_onto
from collections import defaultdict
import graphviz
from sympy import Symbol
import networkx as nx


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
                all_vars = set().union(*(map(lambda x: set(x.vars), self.equations)))
                subs_vars = all_vars.difference(vars)
                subs_vars_values = [(var, 0) for var in subs_vars]

                self.vars = vars
                equalities = list(map(lambda x: x.equation.subs(subs_vars_values), self.equations))
                for i in range(len(self.equations)):
                    self.equations[i].equation = equalities[i]

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
            fcm = self.build_full_causal_mapping()
            direct_dependencies = list()
            for key, value in fcm.items():
                equation = Equation(formula=key)
                dep = equation.get_vars()
                dep.remove(value)
                for d in dep:
                    direct_dependencies.append(tuple((d, value)))

            tc = self.deps_transitive_closure(direct_dependencies)
            return tc

        def deps_transitive_closure(self, direct_dependencies):

            tc = defaultdict(list)
            G = nx.DiGraph()
            for dep in direct_dependencies:
                G.add_edge(dep[0], dep[1])

            for node in G.nodes():
                tc[node] = nx.algorithms.descendants(G, node)
            return tc


        def build_full_causal_mapping(self):
            if not self.is_complete():
                raise Exception('Structure is not complete')
            else:
                left_structures = Structure(equations=self.equations, vars=self.vars)
                fcm = {}

                while(left_structures):

                    if left_structures.is_minimal():
                        sorted_vars = sorted(left_structures.vars, key=lambda x: x.name)
                        for eq in left_structures.equations:
                            fcm[eq.formula] = sorted_vars[0]
                            sorted_vars = [x for x in sorted_vars if x != list(left_structures.vars)[0]]
                        break

                    minimal_structures = left_structures.find_minimal_structures()
                    #print('minimal_structures:', minimal_structures, [eq.equation for eq in left_structures.equations])
                    for mstr in minimal_structures:
                        sorted_vars = sorted(mstr.vars, key=lambda x: x.name)
                        for eq in mstr.equations:
                            fcm[eq.formula] = sorted_vars[0]
                            sorted_vars = [x for x in sorted_vars if x != sorted_vars[0]]

                    difference = left_structures.difference(set(minimal_structures))
                    if len(difference.equations) > 0:
                        left_structures = Structure(equations=difference.equations, vars=difference.vars)
                    else:
                        left_structures = None
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
                # vars = all_vars.popleft()

                s = Structure(equations=subset)
                left_vars = [eq.equation.free_symbols for eq in subset]
                left_vars = set(itertools.chain(*left_vars))
                # if type(vars) is
                s.vars = left_vars
                # vars to be defined as number
                if s.is_complete() and not s.find_minimal_structures():
                    min_str.append(s)

            return min_str

        def union(self, set_structures):
            united = []
            for eq in self.equations:
                united.append(eq)
            for structure in set_structures:
                for eq in structure.equations:
                    united.append(eq)
            return Structure(equations=united)

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

        def build_dcg(self):
            dot = graphviz.Digraph('dcg', comment='Directed Causal Graph')
            h_encode = self.h_encode()
            for key, value in h_encode.items():
                for k in key:
                    for v in value:
                        dot.edge(str(k), str(v))

            return dot


        def h_encode(self):
            if not self.is_complete():
                raise Exception("Structure is not complete")
            else:
                fd = defaultdict(list)
                fcm = self.build_full_causal_mapping()
                # a_s = self.build_matrix()

                for key, value in fcm.items():
                    equation = Equation(formula=key)
                    dep = equation.get_vars()

                    dep.remove(value)
                    if not dep:
                        dep.append(Symbol('phi'))
                    else:
                        dep.append(Symbol('v'))
                    fd[tuple(dep)].append(value)
                return fd


if __name__ == '__main__':
    virtual_experiment_onto = get_ontology("http://synthesis.ipi.ac.ru/virtual_experiment.owl")

    tex1 = r"f_1(x_1)=0"
    tex2 = r"f_2(x_2)=0"
    tex3 = r"f_3(x_3)=0"
    tex4 = r"x_1+x_2+x_3+x_4+x_5=0"
    tex5 = r"x_1 + 6*x_3+x_4+x_5=0"
    tex6 = r"f_6(x_4, x_6)=0"
    tex7 = r"f_7(x_5, x_7)=0"

    e1 = Equation(formula=tex1)
    e2 = Equation(formula=tex2)
    e3 = Equation(formula=tex3)
    e4 = Equation(formula=tex4)
    e5 = Equation(formula=tex5)
    e6 = Equation(formula=tex6)
    e7 = Equation(formula=tex7)

    equations = [e1, e2, e3, e4, e5, e6, e7]
    s = Structure(equations)
    print(s.build_full_causal_mapping())
    print(s.build_matrix())
    print(s.find_minimal_structures())
    print(s.exogenous(), s.endogenous())

    print(s.h_encode())
    print(s.build_dcg().source)
    print(s.build_transitive_closure())

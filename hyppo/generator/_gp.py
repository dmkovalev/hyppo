import numpy as np

class Function:

    def __init__(self, function):
        self.function = function

    def calculation(self, x, y):
        if self.function == '+':
            return x + y
        if self.function == '*':
            return x * y
        if self.function == '-':
            return x - y
        if self.function == '/':
            return x / y

    @staticmethod
    def create_random_function():
        all_functions = ['*', '+', '-', '/']
        func_index = np.random.randint(len(all_functions))
        return Function(all_functions[func_index])

class Variable:

    def __init__(self):
        pass

class Tree:

    def __init__(self):
        self.key = None
        self.left = None
        self.right = None
        self.depth = 0

    def calculation(self):

        if not isinstance(self.key, Function):
            return self.key
        if isinstance(self.key, Function):
            left = self.left.calculation()
        else:
            left = self.left.key
        if isinstance(self.key, Function):
            right = self.right.calculation()
        else:
            right = self.right.key
        return self.key.calculation(left, right)

    def create_random_tree(self, min_depth, max_depth, num_variables=1):
        depth = np.random.randint(min_depth, max_depth+1)
        tree = self._create_random_subtree(depth, num_variables)
        tree.depth = depth
        return tree

    def _create_random_subtree(self, depth, num_variables):
        if depth == 1:
            num_1 = Tree()
            num_2 = Tree()
            num_1.key = np.round(np.random.rand(1)*4-2, 2)[0]
            num_2.key = np.round(np.random.rand(1)*4-2, 2)[0]
            tree = Tree()
            tree.key = Function.create_random_function()
            tree.left = num_1
            tree.right = num_2
            return tree
        else:
            tree = Tree()
            tree.key = Function.create_random_function()
            tree.left = self._create_random_subtree(depth-1)
            tree.right = self._create_random_subtree(depth-1)
            return tree
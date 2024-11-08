#####################################
# Testing code for DiffDecisionTree #
#####################################

import os
import sys
import numpy as np

sys.path.append(os.path.abspath('../src'))
from tree import DiffDecisionTree


if __name__=='__main__':

    # simple syntax test
    X = np.random.rand(1000, 5)
    Y = np.random.rand(1000, 5)
    tree = DiffDecisionTree(leaf_min_entries=200, max_depth=99)
    tree.fit(X, Y)
    print(tree.tree)
    print(tree.leaf_info)
    leaf_numbers_X = tree.eval(X)
    print(leaf_numbers_X[:10])

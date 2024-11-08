########################################
# Test for simple case of one variable #
########################################


# import external modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# import local modules
sys.path.append(os.path.abspath('../src'))
from bdt import ReweighterBDT


if __name__=='__main__':

    # generate toy data
    X = np.random.normal(loc=0.5, scale=1, size=(10000,1))
    Y = np.random.normal(loc=0.0, scale=1, size=(10000,1))

    # calculate reweighting factors
    dtparams = {
      'leaf_min_entries': 200,
      'max_depth': 99
    }
    bdt = ReweighterBDT(num_trees=5, dtparams=dtparams)
    bdt.fit(X, Y)
    reweights = bdt.eval(X)

    # make a plot
    fig, ax = plt.subplots()
    bins = np.linspace(-3, 3, num=50)
    ax.hist(Y[:,0], bins=bins, color='dodgerblue', label='Target')
    ax.hist(X[:,0], bins=bins, color='orange', label='Base', histtype='step', linewidth=2)
    ax.hist(X[:,0], weights=reweights, bins=bins, color='r', label='Reweighted', histtype='step', linewidth=2)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_xlabel('Dummy variable (a.u.)', fontsize=12)
    ax.legend()
    plt.show()

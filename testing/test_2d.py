#########################################
# Test for simple case of two variables #
#########################################


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
    cov = np.array([[1,0],[0,1]])
    X = np.random.multivariate_normal((0.5, -0.5), cov, size=10000)
    Y = np.random.multivariate_normal((0,0), cov, size=10000)

    # make a plot
    fig, ax = plt.subplots()
    ax.scatter(Y[:,0], Y[:,1], color='dodgerblue', label='Target', alpha=0.5, s=1)
    ax.scatter(X[:,0], X[:,1], color='orange', label='Base', alpha=0.5, s=1)
    ax.set_xlabel('Dummy var 1 (a.u.)', fontsize=12)
    ax.set_ylabel('Dummy var 2 (a.u.)', fontsize=12)
    ax.legend()
    plt.show()

    # calculate reweighting factors
    dtparams = {
      'leaf_min_entries': 2000,
      'max_depth': 99
    }
    bdt = ReweighterBDT(num_trees=5, dtparams=dtparams)
    bdt.fit(X, Y)
    reweights = bdt.eval(X)

    # make a plot
    fig, axs = plt.subplots(figsize=(12,6), ncols=2)
    bins = np.linspace(-3, 3, num=50)
    for idx in [0,1]:
        ax = axs[idx]
        ax.hist(Y[:,idx], bins=bins, color='dodgerblue', label='Target')
        ax.hist(X[:,idx], bins=bins, color='orange', label='Base', histtype='step', linewidth=2)
        ax.hist(X[:,idx], weights=reweights, bins=bins, color='r', label='Reweighted', histtype='step', linewidth=2)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_xlabel('Dummy var {} (a.u.)'.format(idx+1), fontsize=12)
        ax.legend()
    plt.show()

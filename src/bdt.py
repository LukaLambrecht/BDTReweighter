###########################################################
# Implementation of boosted decision tree for reweighting #
###########################################################


# import external modules
import numpy as np

# import local modules
from tree import DiffDecisionTree


class ReweighterBDT(object):

    def __init__(self, num_trees=100, dtparams=None):
        self.trees = []
        self.leaf_weights = []
        self.num_trees = num_trees
        self.dtparams = dtparams

    def fit(self, X, Y, w_X=None, w_Y=None):
        '''
        Reweight a sample X to a sample Y.
        Input arguments:
        - X: base sample; numpy array of shape (number of instances, number of variables).
        - Y: target sample; numpy array of shape (number of instances, number of variables).
        - w_X: numpy array of shape (number of instances in X) with weights.
        - w_Y: numpy array of shape (number of instances in Y) with weights.
        '''
        # initializations
        nX = len(X)
        nY = len(Y)
        # set weights to unity if not provided
        if w_X is None: w_X = np.ones(nX)
        if w_Y is None: w_Y = np.ones(nY)
        # loop over requested number of iterations
        for treeidx in range(self.num_trees):
            print('Training tree {}/{}...'.format(treeidx+1, self.num_trees))
            # train a tree
            tree = DiffDecisionTree(**self.dtparams)
            tree.fit(X, Y, w_X=w_X, w_Y=w_Y)
            self.trees.append(tree)
            # calculate weights per leaf
            leaf_weights = {}
            for key,val in tree.leaf_info.items():
                if val['wY'] == 0: leaf_weights[key] = 0.5
                if val['wX'] == 0: leaf_weights[key] = 2.
                else: leaf_weights[key] = val['wY'] / val['wX']
            self.leaf_weights.append(leaf_weights)
            # calculate weights per instance of X
            eval_X = tree.eval(X)
            weights = [leaf_weights[leaf] for leaf in eval_X]
            w_X = np.multiply(w_X, weights)

    def eval(self, X):
        '''
        Evaluate the reweighter BDT on a sample X.
        Returns the reweighting factors for each instance in X.
        '''
        reweights = np.ones(len(X))
        for tree, leaf_weights in zip(self.trees, self.leaf_weights):
            eval_X = tree.eval(X)
            weights = [leaf_weights[leaf] for leaf in eval_X]
            reweights = np.multiply(reweights, weights)
        return reweights

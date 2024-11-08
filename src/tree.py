#########################################
# Implementation of custom descion tree #
#########################################

import numpy as np


class DiffDecisionTree(object):
    '''
    Decision tree trained on maximum-difference metric between base and target distribution.
    '''

    def __init__(self,
            max_depth = 5,
            leaf_min_entries = 30 ):
        self.tree = [None]
        self.depth = 0
        self.leaf_info = {}
        self.max_depth = max_depth
        self.leaf_min_entries = leaf_min_entries

    def fit(self, X, Y, w_X=None, w_Y=None, tree_element=None):
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
        nvars = np.shape(X)[1]
        if np.shape(Y)[1] != nvars: raise Exception('X and Y have incompatible shape.')
        if tree_element is None: tree_element = self.tree
        # set weights to unity if not provided
        if w_X is None: w_X = np.ones(nX)
        if w_Y is None: w_Y = np.ones(nY)
        # check stopping conditions (for running recursively)
        is_leaf = False
        if( nX < self.leaf_min_entries or nY < self.leaf_min_entries): is_leaf = True
        if self.depth >= self.max_depth: is_leaf = True
        if is_leaf:
            leaf_index = len(self.leaf_info)
            self.leaf_info[leaf_index] = {
                    'nX': nX,
                    'nY': nY,
                    'wX': np.sum(w_X),
                    'wY': np.sum(w_Y)
            }
            tree_element[0] = leaf_index
            return
        # loop over variables
        cuts = []
        chi2s = []
        for varidx in range(nvars):
            # calculate maximum chi2 for this variable
            x = X[:,varidx]
            y = Y[:,varidx]
            cut, chi2 = self._chi2max_(x, y, w_x=w_X, w_y=w_Y)
            cuts.append(cut)
            chi2s.append(chi2)
        # select maximum chi2
        maxarg = np.argmax(chi2s)
        splitvar = maxarg
        splitval = cuts[maxarg]
        # update tree structure
        tree_element[0] = {(splitvar, splitval, 0): [None], (splitvar, splitval, 1): [None]}
        self.depth += 1
        # todo: self.depth is not yet calculated correctly (but good enough proxy for now)
        # split input samples in two chunks
        maskx = (X[:,splitvar]<=splitval)
        masky = (Y[:,splitvar]<=splitval)
        X_0 = X[maskx,:]
        w_X_0 = w_X[maskx]
        X_1 = X[~maskx,:]
        w_X_1 = w_X[~maskx]
        Y_0 = Y[masky,:]
        w_Y_0 = w_Y[masky]
        Y_1 = Y[~masky,:]
        w_Y_1 = w_Y[~masky]
        # repeat recursively
        self.fit(X_0, Y_0, w_X=w_X_0, w_Y=w_Y_0, tree_element=tree_element[0][splitvar,splitval,0])
        self.fit(X_1, Y_1, w_X=w_X_1, w_Y=w_Y_1, tree_element=tree_element[0][splitvar,splitval,1])

    def eval(self, X):
        '''
        Evaluate the decision tree on a sample X.
        Return the leaf number for each instance in X.
        '''
        res = []
        for idx in range(len(X)): res.append(self._eval_single_(X[idx,:]))
        return res

    def _chi2_(self, x, y, cut, w_x=None, w_y=None):
        '''
        Calculate custon chi2 value for given values, weights and cut.
        '''
        if w_x is None: w_x = np.ones(len(x))
        if w_y is None: w_y = np.ones(len(y))
        w_x_0 = np.sum(w_x[x<=cut])
        w_x_1 = np.sum(w_x[x>cut])
        w_y_0 = np.sum(w_y[y<=cut])
        w_y_1 = np.sum(w_y[y>cut])
        if w_x_0 + w_y_0 == 0: chi2 = 0
        elif w_x_1 + w_y_1 == 0: chi2 = 0
        else:
            chi2 = np.square(w_x_0 - w_y_0) / (w_x_0 + w_y_0)
            chi2 += np.square(w_x_1 - w_y_1) / (w_x_1 + w_y_1)
        return chi2

    def _chi2max_(self, x, y, w_x=None, w_y=None):
        '''
        Calculate custon chi2 vaue for given values and weights.
        Maximize over cut value.

        To do: this part of the code is crucial; optimize here.
        '''
        cuts = np.linspace(np.amin(x), np.amax(x), num=50)
        chi2s = np.array(list([self._chi2_(x, y, cut, w_x=w_x, w_y=w_y)] for cut in cuts))
        maxarg = np.argmax(chi2s)
        return (cuts[maxarg], chi2s[maxarg])

    def _eval_single_(self, x):
        '''
        Evaluate tree on a given instance
        '''
        tree_element = self.tree
        while not isinstance(tree_element[0], int):
            firstkey = list(tree_element[0].keys())[0]
            splitvar = firstkey[0]
            splitval = firstkey[1]
            if x[splitvar] <= splitval: tree_element = tree_element[0][(splitvar, splitval, 0)]
            else: tree_element = tree_element[0][(splitvar, splitval, 1)]
        return tree_element[0]

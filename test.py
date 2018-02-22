#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from sklearn.decomposition import PCA

def mypca(X, no_dims=50):
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

X = np.array([[-1, -10], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
print("original x:\n{}".format(X))
pca = PCA(n_components=2)
pca.fit(X)
print("after:\n{}".format(pca.transform(X)))
#print("SINGULAR VALUES:\n{}".format(pca.singular_values_))
#print("AFTER TRANSFORM:\n{}".format(pca.transform(X)))
#print("ANOTHER TRANSFORM:\n{}".format(mypca(X,no_dims=2)))

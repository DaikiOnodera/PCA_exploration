#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from sklearn.decomposition import PCA

X = np.loadtxt("mnist2500_X.txt")
print("input:{}".format(X.shape))

pca = PCA(n_components=2)
pca.fit(X)
print("after:{}".format(pca.fit_transform(X).shape))

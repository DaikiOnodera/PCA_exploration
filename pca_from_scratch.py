#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
X = np.array([[1,2],
              [2,4],
              [3,6],
              [4,8]], dtype=float)
"""
X = np.array([[1,1,2],
              [0,2,-1],
              [0,0,3]])
n, p = X.shape

# Stadardize the data
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)

# Perform PCA
cov = X_std.T @ X_std / (n-1)
W, V_pca = np.linalg.eig(cov)

# Sort eigenvectors with eigencalues
index = W.argsort()[::-1]
W = W[index]
V_pca = V_pca[:, index]

# Perform SVD
U, s, V_svd = np.linalg.svd(X_std, full_matrices=True)
V_svd = V_svd.T
S = np.zeros((n, p))
S[:p, :p] = np.diag(s)

# Print results
print("Eigenvalues from PCA")
print(W)
print("EigenValues from SVD")
print(s**2/(n-1))

print("V_pca")
print(V_pca)
print("V_svd")
print(V_svd)

# Plot results
X_pca = X_std @ V_pca[:, :2]
X_svd = X_std @ V_svd[:, :2]

plt.scatter(X_std[:, 0], X_std[:, 1], color="red")
plt.scatter(X_pca[:, 0], X_pca[:, 1], marker="+", color="blue")
plt.scatter(X_svd[:, 0], X_svd[:, 1], marker="o", color="green")
plt.show()

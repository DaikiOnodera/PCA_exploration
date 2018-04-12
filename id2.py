#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

n = 100
A0 = np.random.randn(n, n)
U0, sigma0, VT0 = la.svd(A0)
print(la.norm((U0 * sigma0).dot(VT0) - A0))

sigma = np.exp(-np.arange(n))

A = (U0 * sigma).dot(VT0)
print("A:{}".format(A.shape))
plt.semilogy(sigma)

import scipy.linalg.interpolative as sli

k = 20
idx, proj = sli.interp_decomp(A, k)
print("idx:{}".format(idx.shape))
print("proj:{}".format(proj.shape))
sort_idx = np.argsort(idx)

B = A[:, idx[:k]]
P = np.hstack([np.eye(k), proj])[:, np.argsort(idx)]
print("B:{}".format(B.shape))
print("P:{}".format(P.shape))
Aapprox = np.dot(B, P)
print("norm:{}".format(la.norm(A - Aapprox, 2)))

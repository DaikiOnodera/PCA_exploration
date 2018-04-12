#!/usr/bin/env python
# encoding:utf-8

import scipy.linalg.interpolative as sli
from scipy.linalg import hilbert
n = 1000
A = hilbert(n)

#import numpy as np
#n = 1000
#A = np.empty((n, n), order="F")
#for j in range(n):
#    for i in range(n):
#        A[i, j] = 1. / (i+j+1)

from scipy.sparse.linalg import aslinearoperator
L = aslinearoperator(A)

k, idx, proj = sli.interp_decomp(A, 0.01)
print("k:{}".format(k))
print("proj:{}".format(proj.shape))

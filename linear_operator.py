#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from scipy.sparse.linalg import LinearOperator

def mv(v):
    print("v:{}".format(v))
    print("v:{}".format(v.shape))
    return np.array([2*v[0], 3*v[1]])

A = LinearOperator((2, 3), matvec=mv)
print("A.matvec:{}".format(A.matvec(np.ones(3))))

#!/usr/bin/env python
# encoding:utf-8

import numpy as np

A = np.array([[1,2],[3,4]])
print("A:\n{}".format(A))
U, s, V = np.linalg.svd(A, full_matrices=False)
print("U:\n{}".format(U))
print("s:\n{}".format(np.diag(s)))
print("V:\n{}".format(V))


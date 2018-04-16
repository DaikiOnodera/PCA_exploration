#!/usr/bin/env python
# encoding:utf-8

import numpy as np

A = np.array([[1,2],[3,4]])
AA = np.dot(A, A)
print("A:\n{}".format(A))
U, s, V = np.linalg.svd(A, full_matrices=False)
U_, s_, V_ = np.linalg.svd(AA, full_matrices=False)
print("U:\n{}".format(U))
print("s:\n{}".format(np.diag(s)))
print("V:\n{}".format(V))
print("s_:\n{}".format(np.diag(s_)))


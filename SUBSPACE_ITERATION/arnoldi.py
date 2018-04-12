#!/usr/bin/env python
# encoding:utf-8

import numpy as np

def arnoldi_iteration(A, b, nimp):
    """
    Input
    A: (nxn matrix)
    b: (initial vector)
    k: number of iterations

    Returns Q, h
    """

    m = A.shape[0] #Shape of the input matrix

    h = np.zeros((nimp+1, nimp)) # Creates a zero matrix of shape (n+1) x n
    Q = np.zeros((m, nimp+1)) # Creates a zero matrix of shape m x n

    q = b / np.linalg.norm(b) # Normalize the input vector
    Q[:, 0] = q # Adds to the input vector

    for n in range(nimp):
        v = A.dot(q) # A*q_0
        for j in range(n+1):
            h[j, n] = Q[:, j] * v
            v = v - h[j, n] * Q[:, j]

        h[n+1, n] = np.linalg.norm(v)
        q = v / h[n+1, n]
        Q[:, n+1] = q
    return Q, h

if __name__ == "__main__":
    Q, h = arnoldi_iteration(A, b, nimp)

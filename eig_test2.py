#!/usr/bin/env python
# encoding:utf-8

import numpy as np

def main():
    A = np.array([[3,2],
                  [-2,3]])
    AA = np.dot(A,A)
    la, v = np.linalg.eig(A)
    la_, v_ = np.linalg.eig(AA)
    print(la)
    print(la_)
    print(v)
    print(v_)

if __name__=="__main__":
    main() 

#!/usr/bin/env python
# encoding:utf-8

import numpy as np

def main():
    A = np.array([[3,2],
                  [-2,3]])
    la, v = np.linalg.eig(A)
    print(la)
    print(v)

if __name__=="__main__":
    main() 

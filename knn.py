#!/usr/bin/env python
# encoding:utf-8

import numpy as np

class SingularSpectrumAnalysis(object):
    def __init__(self, signal, window_size):
        self.__signal_length = len(signal)
        self.__window_size = window_size
        X = self.__create_trajectory_matrix(signal, window_size)
        self.__U, self.__W, self.__V = np.linalg.svd(X, False)
        print("signal:{}".format(X.shape))
        print("U shape:{}".format(self.__U.shape))
        print("check:{}".format(np.linalg.svd(np.dot(self.__U.T, self.__U), full_matrices=False, compute_uv=False)))

    def __create_trajectory_matrix(self, signal, window_size):
        row = len(signal) - window_size + 1
        col = window_size
        trajectory_matrix = np.empty((row, col))
        for i in range(row):
            trajectory_matrix[i, :] = signal[i : i+window_size]
        return trajectory_matrix

    def restore_signal(self, num_component):
        X = np.zeros((self.__U.shape[0], self.__window_size))
        for i in range(num_component):
            lambda_ = self.__W[i]
            U = (self.__U[:, i])[:, np.newaxis]
            V = (self.__V[i, :])[np.newaxis, :]
            X += lambda_ * U * V
        signal = []
        for i in range(self.__signal_length):
            value = 0.0
            count = 0
            for j in range(X.shape[1]):
                row = i - j
                if (row < 0):
                    break
                elif (row >= X.shape[0]):
                    continue
                col = j
                value += X[row, col]
                count += 1
            value /= count
            signal.append(value)
        return np.array(signal)
 
np.random.seed(1)
import matplotlib.pyplot as plt
x = np.arange(100)
signal1 = x / 10
signal2 = np.random.rand(len(x))/2
signal3 = np.sin(x/2)
signal = signal1 + signal2 + signal3

window_size = 20
ssa = SingularSpectrumAnalysis(signal, window_size)
num_component = 1
restored_signal = ssa.restore_signal(num_component)

plt.hold(True)
plt.title("num_component="+str(num_component))
plt.plot(signal)
plt.plot(restored_signal)
plt.show()

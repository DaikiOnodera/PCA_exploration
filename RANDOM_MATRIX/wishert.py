#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import scipy.stats as ss

w = ss.wishart(df=3, scale=np.matrix([[1.0, 0.5], [0.5, 1.0]]))
print(w.rvs(10).shape)

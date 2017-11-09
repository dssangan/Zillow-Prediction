# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 20:11:18 2017

@author: sanga
"""

import numpy as np

# load data
y_train = np.fromfile('y.bin').reshape(-1,1)
x_train = np.fromfile('x.bin').reshape(-1,59)
x_test = np.fromfile('x_with_months.bin').reshape(-1,59)

np.savetxt('xvec_fulltrain.csv', x_train, delimiter=',')
np.savetxt('yvec_fulltrain.csv', y_train, delimiter=',')
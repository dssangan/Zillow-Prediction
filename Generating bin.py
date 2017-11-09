# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 20:28:52 2017

@author: sanga
"""

import csv
import numpy as np

xtr = np.loadtxt('xtrain.csv', delimiter=',')
xtr = xtr.reshape(-1, 59)

xts = np.loadtxt('xtest.csv', delimiter=',')
xts = xts.reshape(-1, 59)

ytr = np.loadtxt('ytrain.csv', delimiter=',')
ytr = ytr.reshape(-1, 1)

yts = np.loadtxt('ytest.csv', delimiter=',')
yts = yts.reshape(-1, 1)

xtr.tofile('xtrain.bin')
xts.tofile('xtest.bin')
ytr.tofile('ytrain.bin')
yts.tofile('ytest.bin')
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:34:30 2017

@author: kelby
"""

import csv
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def gradBoost(x_test, months, est):
    height = x_test.shape[0]
    width = x_test.shape[1]

    for rowNum in range(height):
        x_test[rowNum][width-1] = months[0]
    y = est.predict(x_test).reshape(-1, 1)

    for monIdx in range(1, len(months)):
        for rowNum in range(height):
            x_test[rowNum][width-1] = months[monIdx]
        y = np.append(y, est.predict(x_test).reshape(-1, 1), axis=1)

    return y

def toExcel(y):
    with open('sample_submission.csv', 'r') as inp, open('grad_boost_72.csv', 'w', newline='') as out:
        reader = csv.reader(inp)
        writer = csv.writer(out)
        writer.writerow(next(reader))
        for readRow, arrRow in zip(reader, y):
            rowList = [readRow[0]]
            for val in arrRow:
                rowList.append('{0:.4f}'.format(val))
            writer.writerow(rowList)

"""main"""
y_train = np.fromfile('y.bin')
x_train = np.fromfile('x.bin').reshape(-1, 59)
x_test = np.fromfile('x_with_months.bin').reshape(-1, 59)

est = GradientBoostingRegressor(n_estimators=127, learning_rate=0.075, 
    max_depth=6, random_state=10, loss='lad').fit(x_train, y_train)

y = gradBoost(x_test, [10, 11, 12, 22, 23, 24], est)
toExcel(y)
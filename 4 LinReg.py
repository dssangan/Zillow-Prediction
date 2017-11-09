# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 22:37:40 2017

@author: kelby
"""

from sklearn import linear_model
import numpy as np
import csv

x = np.fromfile('x.bin')
x = np.reshape(x, (-1, 200))

y = np.fromfile('y.bin')
y = np.reshape(y, (-1, 1))

reg = linear_model.LinearRegression()
reg.fit (x, y)
W = reg.coef_.reshape(200, 1) #reg.coef_ is the transpose of w (omega)

#LinReg: Linear Regression Function
#Inputs: data matrix, list of months (e.g. [1, 24] --> [Jan 2016, Dec 2017])
#Output: prediction matrix
def LinReg(X, months):
    XLength = len(X)

    for idx in range(XLength):
        X[idx][199] = months[0]
    Y = np.dot(X, W)

    for n in range(1, len(months)):
        for idx in range(XLength):
            X[idx][199] = months[n]
        Y = np.append(Y, np.dot(X, W), axis=1)

    return Y


#toExcel: put prediction matrix into submittable excel format
#Input: prediction matrix
#Output: submission excel file created in program directory
def toExcel(Y):
    with open('sample_submission.csv', 'r') as inp, open('test.csv', 'w', newline='') as out:
        reader = csv.reader(inp)
        writer = csv.writer(out)
        writer.writerow(next(reader))
        for readRow, arrRow in zip(reader, Y):
            rowList = [readRow[0]]
            for val in arrRow:
                rowList.append('{0:.4f}'.format(val))
            writer.writerow(rowList)

X = np.fromfile('x_with_months.bin')
X = np.reshape(X, (-1, 200))
Y = LinReg(X, [10, 11, 12, 22, 23, 24])

toExcel(Y)
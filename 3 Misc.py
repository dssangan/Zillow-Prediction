# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:25:12 2017

@author: kelby
"""

import numpy as np
import pickle
import csv

def makeTrainDict(): #make training dictionaries; map parcelid to [logerror, month]
    singles = {} #dictionary of parcels that occur one time
    doubles = {} #dictionary of parcels that occur two times
    triples = {} #dictionary of parcels that occur three times

    with open('train_2016_v2.csv') as csvfile:
        reader = csv.reader(csvfile, dialect='excel')
        next(reader)
        for row in reader:
            month = int(row[2][5:7])
            if row[0] in singles:
                if row[0] in doubles:
                    triples[row[0]] = [row[1], month]
                else:
                    doubles[row[0]] = [row[1], month]
            else:
                singles[row[0]] = [row[1], month]
    
    return [singles, doubles, triples] #return list of training dictionaries

def makeYVector(xDict, singles, doubles, triples): #make y-vector from training dictionary d
    ls = [] #list of indexes used by training data

    #fill in upper portion of y vector (a.k.a. no repeats of id's)    
    y = np.zeros((len(xDict) + len(doubles) + len(triples), 1))
    for key in singles:
        index1 = xDict[key] #row index of ID in x vector
        ls.append(index1)
        y[index1][0] = singles[key][0]
    
    #fill in lower portion of y vector (a.k.a. repeats of id's)
    index2 = len(xDict)
    for key in doubles:
        ls.append(index2)
        y[index2][0] = doubles[key][0]
        index2 += 1
    for key in triples:
        ls.append(index2)
        y[index2][0] = triples[key][0]
        index2 += 1
    
    return [y, ls]

def addMonthToXVector(xvector, xDict, singles): #add month column to x vector
    months = np.zeros((len(xDict), 1))
    for key in singles:
        index = xDict[key] #row index of ID in x vector
        months[index][0] = singles[key][1]
    return np.append(xvector, months, 1)

def appendRepeatedRows(x, y, xDict, doubles, triples):
    for key in doubles:
        idx = xDict[key]
        row = x[idx]
        row[199] = doubles[key][1] #set month for appended row
        x = np.append(x, [row], 0)
        
    for key in triples:
        idx = xDict[key]
        row = x[idx]
        row[199] = triples[key][1] #set month for appended row
        x = np.append(x, [row], 0)
        
    return x

def deleteUnusedRows(xcsv, ycsv, whiteList):
    #delete extra rows from vectors x and y
    with open(xcsv, 'r') as xinp, \
    open(ycsv, 'r') as yinp, \
    open('x_after_delete.csv', 'w', newline='') as xout, \
    open('y_after_delete.csv', 'w', newline='') as yout:
        xwriter = csv.writer(xout)
        ywriter = csv.writer(yout)
        xreader = csv.reader(xinp)
        yreader = csv.reader(yinp)
        rowNum = 0
        for xrow, yrow in zip(xreader, yreader):
            print("Loop")
            if rowNum == whiteList[0]:
                print("Found")
                xwriter.writerow(xrow)
                ywriter.writerow(yrow)
                del whiteList[0]
            rowNum += 1

xDict = pickle.load(open('ides.p', 'rb')) #maps ID to row index in x vector
x = np.fromfile('xvector_avg.bin', dtype=float, count=-1, sep='') #load 1D array from bin file
x = np.reshape(x, (len(xDict), -1))

dictList = makeTrainDict()
singles = dictList[0]
doubles = dictList[1]
triples = dictList[2]

temp = makeYVector(xDict, singles, doubles, triples)
#y = temp[0]
whiteList = sorted(temp[1])

x = addMonthToXVector(x, xDict, singles)

x.tofile('x_with_months.bin')

"""
x = appendRepeatedRows(x, y, xDict, doubles, triples)

np.savetxt('x_before_delete.csv', x, delimiter=',')
np.savetxt('y_before_delete.csv', y, delimiter=',')

deleteUnusedRows('x_before_delete.csv', 'y_before_delete.csv', whiteList)

x = np.loadtxt('x_after_delete.csv', delimiter=',')
x = x.reshape(-1, 200)

y = np.loadtxt('y_after_delete.csv', delimiter=',')
y = y.reshape(-1, 1)

x.tofile('x.bin')
y.tofile('y.bin')
"""
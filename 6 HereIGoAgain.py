# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:48:04 2017

@author: kelby
"""

import numpy as np
import pickle
import csv

def make_features():
    features = {}
    index = 0
    with open('zillow_data_dictionary.csv') as csvfile:
        reader = csv.reader(csvfile, dialect='excel')
        next(reader)
        for row in reader:
            features[row[0]] = index
            index += 1
    return features


def print_features(features):
    for feature in features:
        if type(features[feature]) == dict:
            for sub in features[feature]:
                print(sub + " : " + str(features[feature][sub]))
        else:
            print(feature + " : " + str(features[feature]))


def make_xvector(d):
    x = np.zeros((2985217, len(d)))
    for col in range(1, 58):
        with open('properties_2016.csv') as csvfile:
            reader = csv.reader(csvfile)
            feat = "'" + next(reader)[col] + "'"
            rowNum = 0
            if feat != "'propertycountylandusecode'" and feat != "'propertyzoningdesc'":
                if type(d[feat]) == dict: #if this column's feature has subfeatures
                    for row in reader:
                        data = row[col]
                        if data != "":
                            colNum = d[feat][data]
                            x[rowNum, colNum] = 1
                            print(feat)
                        rowNum += 1
                else:
                    for row in reader:
                        data = row[col]
                        if data != "":
                            if data == "true" or data == "Y":
                                data = 1
                            colNum = d[feat]
                            x[rowNum, colNum] = data
                            print(feat)
                        rowNum += 1
    return x


def make_ides():
    ides = {}
    index = 0
    with open('properties_2016.csv') as csvfile:
        reader = csv.reader(csvfile, dialect='excel')
        next(reader)
        for row in reader:
            ides[row[0]] = index
            index += 1
    return ides


def size_dict(d):
    count = 0
    for feature in d:
        if type(d[feature]) == dict:
            for sub in d[feature]:
                count += 1
        else:
            count += 1
    return count


def avg_fill(array, col, ides):
    total = 0
    numIdes = 0
    for row in array:
        if row[col] != "":
            total += row[col]
            numIdes += 1
    average = float(total)/numIdes
    print("average: " + str(average))
    for r in range(size_dict(ides)):
        if array[r, col] == 0.0:
            array[r, col] = average
            print(array[r, col])
    return array


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
        row[len(row)-1] = doubles[key][1] #set month for appended row
        x = np.append(x, [row], 0)
        
    for key in triples:
        idx = xDict[key]
        row = x[idx]
        row[len(row)-1] = triples[key][1] #set month for appended row
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
            

"""MAIN"""
"""
feat = make_features() #dictionary of features mapped to column indexes
ides = make_ides() #dictionary of parcel id's mapped to row indexes
pickle.dump(feat, open("feat.p", "wb")) #pickle features dictionary
pickle.dump(ides, open("ides.p", "wb")) #pickle id's dictionary

print("dictionaries")

x = make_xvector(feat)
x.tofile('xvector.bin')

print("xvector")

x = avg_fill(x, feat["'taxvaluedollarcnt'"], ides)
x = avg_fill(x, feat["'structuretaxvaluedollarcnt'"], ides)
x = avg_fill(x, feat["'landtaxvaluedollarcnt'"], ides)
x = avg_fill(x, feat["'taxamount'"], ides)
x.tofile('xvector_avg.bin')

print("xvector_avg")

dictList = makeTrainDict()
singles = dictList[0]
doubles = dictList[1]
triples = dictList[2]

temp = makeYVector(ides, singles, doubles, triples)
y = temp[0]
whiteList = sorted(temp[1])

x = addMonthToXVector(x, ides, singles)
x.tofile('x_with_months.bin')

print("x_with_months")

x = appendRepeatedRows(x, y, ides, doubles, triples)

np.savetxt('x_before_delete.csv', x, delimiter=',')
np.savetxt('y_before_delete.csv', y, delimiter=',')

deleteUnusedRows('x_before_delete.csv', 'y_before_delete.csv', whiteList)

x = np.loadtxt('x_after_delete.csv', delimiter=',')
x = x.reshape(-1, len(feat)+1)

y = np.loadtxt('y_after_delete.csv', delimiter=',')
y = y.reshape(-1, 1)

x.tofile('x.bin')
y.tofile('y.bin')
"""
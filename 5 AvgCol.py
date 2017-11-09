# -*- coding: utf-8 -*-
"""
Created on Wed Jul 5 11:01:33 2017

@author: kelby
"""

import csv

"""Functions"""

def listFeatures():
    """Create a list of tuples, each containing a column index and its
    corresponding feature name
    """
    with open('properties_2016.csv') as file:
        reader = csv.reader(file)
        featIndex = list(enumerate(next(reader)))
    return featIndex

def averageColumns():
    """Fill empty cells in the taxvaluedollarcnt, structuretaxvaluedollarcnt,
    landtaxvaluedollarcnt, or taxamount columns with the average value of said
    feature
    """
    featIndex = listFeatures()
    colSum = {'taxvaluedollarcnt': [0, 0],
              'structuretaxvaluedollarcnt': [0, 0],
              'landtaxvaluedollarcnt': [0, 0],
              'taxamount': [0, 0]
    }
    with open('properties_2016.csv') as inp, open('averageColumns.csv'): as out
        reader = csv.reader(inp)
        writer = csv.writer(out)
        next(reader)
        for row in reader:
            for feat in colSum:
                if row[featIndex[feat]] != ''
                    colSum[feat][0] += row[featIndex[feat]] #add to column sum
                    colSum[feat][1] += 1 #increment count of filled cells
        colAvg = {}
        for feat in colSum:
            colAvg[feat] = colSum[feat][0]/colSum[feat][1] #avg = sum/cnt
            #TEARS FOR FEARS
        
"""Main"""

averageColumns()
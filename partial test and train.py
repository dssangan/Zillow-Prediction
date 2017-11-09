# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 20:15:08 2017

@author: sanga
"""

import csv

def createdata():
    with open ('xvec_fulltrain.csv', 'r') as xinp , \
    open('yvec_fulltrain.csv','r') as yinp , \
    open('xtrain.csv','w', newline='') as xtrout, \
    open('ytrain.csv','w', newline='') as ytrout, \
    open('xtest.csv','w', newline='') as xtsout, \
    open('ytest.csv','w', newline='') as ytsout:
        xtrwriter = csv.writer(xtrout)
        ytrwriter = csv.writer(ytrout)
        xtswriter = csv.writer(xtsout)
        ytswriter = csv.writer(ytsout)
        xreader = csv.reader(xinp)
        yreader = csv.reader(yinp)
        rowNum = 0
        for xrow, yrow in zip(xreader, yreader):
            if rowNum <= 45137:
                xtrwriter.writerow(xrow)
                ytrwriter.writerow(yrow)
            else:
                xtswriter.writerow(xrow)
                ytswriter.writerow(yrow)
            rowNum += 1

createdata()
        
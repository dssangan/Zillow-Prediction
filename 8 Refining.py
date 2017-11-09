# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:19:23 2017

@author: kelby
"""

from sklearn.decomposition import PCA
import numpy as np

"""Dimensionality Reduction"""
def dimRed(X_test, X_train, n_components):
    pca = PCA(n_components=n_components, svd_solver='full', whiten=True).fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    return X_test, X_train_pca
    
def MSE(yest, yreal): #mean square error
    SE = 0
    for estRow, realRow in zip(yest, yreal):
        SE += (yest - yreal) ** 2
    return SE/len(yest)

def 
    
"""Main"""
"""
X_test = np.fromfile('x_with_months.bin').reshape(-1, 59)
X_train = np.fromfile('x.bin').reshape(-1, 59)
"""

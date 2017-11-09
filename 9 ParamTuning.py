# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:31:45 2017

@author: kelby
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingRegressor
from time import time
import numpy as np

# Load and reshape training vectors from binary files
X_train = np.fromfile('x.bin').reshape(90275, -1)
y_train = np.fromfile('y.bin')

# Split the dataset in two equal parts
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)
"""

# Set the parameters by cross-validation
tuned_parameters = [
    {'n_estimators': [79, 91, 127],
     'learning_rate': [0.070, 0.075, 0.090],
     'max_depth': [4, 5, 6],
     'random_state': [10],
     'loss': ['lad']
     }
]

scores = ['precision']#, 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    start = time()
    
    est = GridSearchCV(GradientBoostingRegressor(verbose=1), tuned_parameters, cv=5)#,
        #scoring='%s_macro' % score)
    est.fit(X_train, y_train)
    
    print("Time: " + str(time() - start))

    print("Best parameters set found on development set:")
    print()
    print(est.best_params_)
    
    """
    print()
    print("Grid scores on development set:")
    print()
    means = est.cv_results_['mean_test_score']
    stds = est.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, est.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, est.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    """
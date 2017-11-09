# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:34:42 2017

@author: darshan
"""


import numpy as np
import pandas as pd
import xgboost as xgb

properties = pd.read_csv('properties_2016.csv')

y_train = np.fromfile('y.bin')
x_train = np.fromfile('x.bin').reshape(-1,59)
x_test = np.fromfile('x_with_months.bin').reshape(-1,59)
y_mean = np.mean(y_train)

parametr = {'eta': 0.05, 'max_depth': 6, 'subsample': 0.80, 'objective': 'reg:linear','eval_metric': 'mae',
    'base_score': y_mean, 'silent': 1, 'loss': 'lad' }

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


cv_result = xgb.cv(parametr, dtrain, nfold=5, num_boost_round=30, early_stopping_rounds=5,
                   verbose_eval=2, show_stdv=False )
print(cv_result)
num_boost_rounds = 30
#print(num_boost_rounds)

# train model
model = xgb.train(dict(parametr, silent=1), dtrain, num_boost_round=num_boost_rounds, verbose_eval=True,
                  learning_rates = [0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23])
pred = model.predict(dtest)
y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})

# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

from datetime import datetime
output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
































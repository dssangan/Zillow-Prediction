# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:25:15 2017

@author: darshan
"""


import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot
#from sklearn.preprocessing import LabelEncoder

properties = pd.read_csv('properties_2016.csv')
'''
train = pd.read_csv("../input/train_2016_v2.csv")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.42 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
y_train = np.fromfile('y.bin')
x_train = np.fromfile('x.bin').reshape(-1,59)
x_test = np.fromfile('x_with_months.bin').reshape(-1,59)
y_mean = np.mean(y_train)

#y_mean = 0.0114572196068
# xgboost params
xgb_params = {
    'n_estimators':120,
    'eta': 0.15,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1,
    'loss': 'lad'
}


'''
xgb_params = {
    'eta': 0.12,
    'max_depth': 4,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1,
    'loss': 'lad'
}
'''

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)


# cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=5,
                   num_boost_round=200,
                   early_stopping_rounds=5,
                   verbose_eval=10, 
                   show_stdv=False
                  )                  
num_boost_rounds = len(cv_result)
#num_boost_rounds = 180
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
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
#output.to_csv('xgboost_78.csv')
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:57:04 2017

@author: darshan
"""

import numpy as np
#from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
import xgboost as xgb
import pandas as pd

print("Reading data")
properties = pd.read_csv('properties_2016.csv')
y_train = np.fromfile('y.bin')
x_train = np.fromfile('x.bin').reshape(-1,59)
x_test = np.fromfile('x_with_months.bin').reshape(-1,59)
y_mean = np.mean(y_train)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("Training model")
model = xgb.XGBRegressor(objective = 'reg:linear',
             nthread = 2,
             max_depth= 6, subsample = 0.75,
             silent = True,
             learning_rate = 0.07,
             n_estimators = 127,
             base_score = 0.019,
             gamma = 0.0001).fit(x_train, y_train)
             

xgb.plot_importance(model)
pyplot.savefig('aimportant_feature.png', dpi=2000, facecolor = 'r' , orientation = 'landascape')
pyplot.show()
xgb.plot_tree(model, num_trees=50)
pyplot.savefig('atree.png', dpi=3000, facecolor = 'r' , orientation = 'landscape')
pyplot.show()

print('making predictions')
#making predictions
preds = model.predict(x_test)


print("putting results into csv")
y_pred=[]

for i,predict in enumerate(preds):
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
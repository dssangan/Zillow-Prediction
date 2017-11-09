# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 13:24:12 2017

@author: darshan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
#import xgboost as xgb
import gc


#Reading data from csv files
print( "\nReading data from disk ...")
properties = pd.read_csv('properties_2016.csv')
train1 = pd.read_csv("october16y.csv")
train2 = pd.read_csv("november16y.csv")
train3 = pd.read_csv("december16y.csv")
train4 = pd.read_csv("october17y.csv")
train5 = pd.read_csv("november17y.csv")
train6 = pd.read_csv("december17y.csv")

################################################
################################################
######## Process data for 1st XGBOOST###########
################################################
################################################

#fill median for numeric columns
for c in properties.dtypes[properties.dtypes == 'float64'].index.values:
	properties[c].fillna(properties[c].median(skipna=True), inplace = True)
 
#I found this is in another kernel :fill -1 for categorical columns
print( "\nProcessing data for 1st XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

#this is making xvector for training by merging all properties value by matching the parcel id same in yvector
train_df = train1.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1) 
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
##### RUN 1st XGBOOST

print("\nSetting up data for Gradient Boosting ...")


# train model
print( "\nTraining Gradient Boosting ...")
model = GradientBoostingRegressor(n_estimators=127, learning_rate=0.075, 
    max_depth=6, random_state=10, loss='lad').fit(x_train, y_train)

print( "\nPredicting with Gradient Boosting ...")
pred = model.predict(x_test)

xgb_pred1=[]

for i,predict in enumerate(pred):
    xgb_pred1.append(str(round(predict,4)))
xgb_pred1=np.array(xgb_pred1)
print( "\nFirst predictions:" )
print( pd.DataFrame(xgb_pred1).head() )

del train_df
del x_train
del y_train
del y_mean
del x_test
del pred
gc.collect()

################################################
################################################
######## Process data for 2nd XGBOOST###########
################################################
################################################

#fill median for numeric columns
for c in properties.dtypes[properties.dtypes == 'float64'].index.values:
	properties[c].fillna(properties[c].median(skipna=True), inplace = True)
 
#I found this is in another kernel :fill -1 for categorical columns
print( "\nProcessing data for 2nd XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

#this is making xvector for training by merging all properties value by matching the parcel id same in yvector
train_df = train2.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1) 
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
##### RUN 2nd XGBOOST

print("\nSetting up data for Gradient Boosting ...")


# train model
print( "\nTraining Gradient Boosting ...")
model = GradientBoostingRegressor(n_estimators=127, learning_rate=0.075, 
    max_depth=6, random_state=10, loss='lad').fit(x_train, y_train)

#test model
print( "\nPredicting with Gradient Boosting ...")
pred = model.predict(x_test)

xgb_pred2=[]

for i,predict in enumerate(pred):
    xgb_pred2.append(str(round(predict,4)))
xgb_pred2=np.array(xgb_pred2)
print( "\nSecond XGBoost predictions:" )
print( pd.DataFrame(xgb_pred2).head() )

del train_df
del x_train
del y_train
del y_mean
del x_test
del pred
gc.collect()

################################################
################################################
######## Process data for 3rd XGBOOST###########
################################################
################################################

#fill median for numeric columns
for c in properties.dtypes[properties.dtypes == 'float64'].index.values:
	properties[c].fillna(properties[c].median(skipna=True), inplace = True)
 
#I found this is in another kernel :fill -1 for categorical columns
print( "\nProcessing data for 3rd XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

#this is making xvector for training by merging all properties value by matching the parcel id same in yvector
train_df = train3.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1) 
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
##### RUN 3rd XGBOOST

print("\nSetting up data for Gradient Boosting ...")


# train model
print( "\nTraining Gradient Boosting ...")
model = GradientBoostingRegressor(n_estimators=127, learning_rate=0.075, 
    max_depth=6, random_state=10, loss='lad').fit(x_train, y_train)

#test model
print( "\nPredicting with Gradient Boosting ...")
pred = model.predict(x_test)

xgb_pred3=[]

for i,predict in enumerate(pred):
    xgb_pred3.append(str(round(predict,4)))
xgb_pred3=np.array(xgb_pred3)
print( "\nThird XGBoost predictions:" )
print( pd.DataFrame(xgb_pred3).head() )

del train_df
del x_train
del y_train
del y_mean
del x_test
del pred
gc.collect()

################################################
################################################
######## Process data for 4th XGBOOST###########
################################################
################################################

#fill median for numeric columns
for c in properties.dtypes[properties.dtypes == 'float64'].index.values:
	properties[c].fillna(properties[c].median(skipna=True), inplace = True)
 
#I found this is in another kernel :fill -1 for categorical columns
print( "\nProcessing data for 4th XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

#this is making xvector for training by merging all properties value by matching the parcel id same in yvector
train_df = train4.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1) 
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
##### RUN 4th XGBOOST

# train model
print( "\nTraining Gradient Boosting ...")
model = GradientBoostingRegressor(n_estimators=127, learning_rate=0.075, 
    max_depth=6, random_state=10, loss='lad').fit(x_train, y_train)

#test model
print( "\nPredicting with Gradient Boosting ...")
pred = model.predict(x_test)

xgb_pred4=[]

for i,predict in enumerate(pred):
    xgb_pred4.append(str(round(predict,4)))
xgb_pred4=np.array(xgb_pred4)
print( "\nFourth XGBoost predictions:" )
print( pd.DataFrame(xgb_pred4).head() )

del train_df
del x_train
del y_train
del y_mean
del x_test
del pred
gc.collect()

################################################
################################################
######## Process data for 5th XGBOOST###########
################################################
################################################

#fill median for numeric columns
for c in properties.dtypes[properties.dtypes == 'float64'].index.values:
	properties[c].fillna(properties[c].median(skipna=True), inplace = True)
 
#I found this is in another kernel :fill -1 for categorical columns
print( "\nProcessing data for 5th XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

#this is making xvector for training by merging all properties value by matching the parcel id same in yvector
train_df = train5.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1) 
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
##### RUN 5th XGBOOST

# train model
print( "\nTraining Gradient Boosting ...")
model = GradientBoostingRegressor(n_estimators=127, learning_rate=0.075, 
    max_depth=6, random_state=10, loss='lad').fit(x_train, y_train)

#test model
print( "\nPredicting with Gradient Boosting ...")
pred = model.predict(x_test)

xgb_pred5=[]

for i,predict in enumerate(pred):
    xgb_pred5.append(str(round(predict,4)))
xgb_pred5=np.array(xgb_pred5)
print( "\nFifth XGBoost predictions:" )
print( pd.DataFrame(xgb_pred5).head() )

del train_df
del x_train
del y_train
del y_mean
del x_test
del pred
gc.collect()

################################################
################################################
######## Process data for 6th XGBOOST###########
################################################
################################################

#fill median for numeric columns
for c in properties.dtypes[properties.dtypes == 'float64'].index.values:
	properties[c].fillna(properties[c].median(skipna=True), inplace = True)
 
#I found this is in another kernel :fill -1 for categorical columns
print( "\nProcessing data for 6th XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

#this is making xvector for training by merging all properties value by matching the parcel id same in yvector
train_df = train6.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1) 
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
'''
##### RUN 6th XGBOOST

# train model
print( "\nTraining Gradient Boosting ...")
model = GradientBoostingRegressor(n_estimators=127, learning_rate=0.075, 
    max_depth=6, random_state=10, loss='lad').fit(x_train, y_train)

#test model
print( "\nPredicting with Gradient Boosting ...")
pred = model.predict(x_test)

xgb_pred6=[]

for i,predict in enumerate(pred):
    xgb_pred6.append(str(round(predict,4)))
xgb_pred6=np.array(xgb_pred6)
print( "\nSixth XGBoost predictions:" )
print( pd.DataFrame(xgb_pred6).head() )

del train_df
del x_train
del y_train
del y_mean
del x_test
del pred
gc.collect()



#Putting into filnal csv files
output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': xgb_pred1, '201611': xgb_pred2, '201612': xgb_pred3,
        '201710': xgb_pred4, '201711': xgb_pred5, '201712': xgb_pred6})
print( output.head() )

# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

print( "\nWriting results to disk ..." )
from datetime import datetime
output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")










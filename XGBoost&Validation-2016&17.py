# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:12:35 2017

@author: darshan
"""

MAKE_SUBMISSION = True          # Generate output file.
CV_ONLY = False                 # Do validation only; do not generate predicitons.
FIT_FULL_TRAIN_SET = True       # Fit model to full training set after doing validation.
VAL_SPLIT_DATE = '2016-10-01'   # Cutoff date for validation split
LEARNING_RATE = 0.007           # shrinkage rate for boosting roudns
ROUNDS_PER_ETA = 20             # maximum number of boosting rounds times learning rate
OPTIMIZE_FUDGE_FACTOR = True    # Optimize factor by which to multiply predictions.
FUDGE_FACTOR_SCALEDOWN = 0.8    # exponent to reduce optimized fudge factor for prediction

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import datetime as dt
from datetime import datetime
import gc
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg

properties = pd.read_csv('properties_2016.csv')

# Number of properties in the zip
zip_count = properties['regionidzip'].value_counts().to_dict()
# Number of properties in the city
city_count = properties['regionidcity'].value_counts().to_dict()
# Median year of construction by neighborhood
medyear = properties.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()
# Mean square feet by neighborhood
meanarea = properties.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()
# Neighborhood latitude and longitude
medlat = properties.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()
medlong = properties.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()

train = pd.read_csv("train_2016_v2.csv")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))
        
train_df = train.merge(properties, how='left', on='parcelid')
select_qtr4 = pd.to_datetime(train_df["transactiondate"]) >= VAL_SPLIT_DATE

del train
gc.collect()

# Inputs to features that depend on target variable
# (Ideally these should be recalculated, and the dependent features recalculated,
#  when fitting to the full training set.  But I haven't implemented that yet.)

# Standard deviation of target value for properties in the city/zip/neighborhood
citystd = train_df[~select_qtr4].groupby('regionidcity')['logerror'].aggregate("std").to_dict()
zipstd = train_df[~select_qtr4].groupby('regionidzip')['logerror'].aggregate("std").to_dict()
hoodstd = train_df[~select_qtr4].groupby('regionidneighborhood')['logerror'].aggregate("std").to_dict()

def calculate_features(df):
    # Nikunj's features
    # Number of properties in the zip
    df['N-zip_count'] = df['regionidzip'].map(zip_count)
    # Number of properties in the city
    df['N-city_count'] = df['regionidcity'].map(city_count)
    # Does property have a garage, pool or hot tub and AC?
    df['N-GarPoolAC'] = ((df['garagecarcnt']>0) & \
                         (df['pooltypeid10']>0) & \
                         (df['airconditioningtypeid']!=5))*1 

    # More features
    # Mean square feet of neighborhood properties
    df['mean_area'] = df['regionidneighborhood'].map(meanarea)
    # Median year of construction of neighborhood properties
    df['med_year'] = df['regionidneighborhood'].map(medyear)
    # Neighborhood latitude and longitude
    df['med_lat'] = df['regionidneighborhood'].map(medlat)
    df['med_long'] = df['regionidneighborhood'].map(medlong)

    df['zip_std'] = df['regionidzip'].map(zipstd)
    df['city_std'] = df['regionidcity'].map(citystd)
    df['hood_std'] = df['regionidneighborhood'].map(hoodstd)
dropvars = ['parcelid', 'airconditioningtypeid', 'buildingclasstypeid',
            'buildingqualitytypeid', 'regionidcity']
droptrain = ['logerror', 'transactiondate']
calculate_features(train_df)

x_valid = train_df.drop(dropvars+droptrain, axis=1)[select_qtr4]
y_valid = train_df["logerror"].values.astype(np.float32)[select_qtr4]

print('Shape full training set: {}'.format(train_df.shape))
print('Dropped vars: {}'.format(len(dropvars+droptrain)))
print('Shape valid X: {}'.format(x_valid.shape))
print('Shape valid y: {}'.format(y_valid.shape))

train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
print('\nFull training set after removing outliers, before dropping vars:')     
print('Shape training set: {}\n'.format(train_df.shape))

if FIT_FULL_TRAIN_SET:
    full_train = train_df.copy()

train_df=train_df[~select_qtr4]
x_train=train_df.drop(dropvars+droptrain, axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
n_train = x_train.shape[0]
print('Training subset after removing outliers:')     
print('Shape train X: {}'.format(x_train.shape))
print('Shape train y: {}'.format(y_train.shape))

if FIT_FULL_TRAIN_SET:
    x_full = full_train.drop(dropvars+droptrain, axis=1)
    y_full = full_train["logerror"].values.astype(np.float32)
    n_full = x_full.shape[0]
    print('\nFull trainng set:')     
    print('Shape train X: {}'.format(x_train.shape))
    print('Shape train y: {}'.format(y_train.shape))
    
if not CV_ONLY:
    test_df = properties
    calculate_features(test_df)
    x_test = test_df.drop(dropvars, axis=1)
    print('Shape test: {}'.format(x_test.shape))
    del test_df
    
del train_df
del select_qtr4
gc.collect()

xgb_params = {  # best as of 2017-09-28 13:20 UTC
    'eta': LEARNING_RATE,
    'max_depth': 7, 
    'subsample': 0.56,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 5.0,
    'alpha': 0.65,
    'colsample_bytree': 0.45,
    'base_score': y_mean,'taxdelinquencyyear'
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dvalid_x = xgb.DMatrix(x_valid)
dvalid_xy = xgb.DMatrix(x_valid, y_valid)
if not CV_ONLY:
    dtest = xgb.DMatrix(x_test)
    del x_test
    
del x_train
gc.collect()

num_boost_rounds = round( ROUNDS_PER_ETA / xgb_params['eta'] )
early_stopping_rounds = round( num_boost_rounds / 20 )
print('Boosting rounds: {}'.format(num_boost_rounds))
print('Early stoping rounds: {}'.format(early_stopping_rounds))

evals = [(dtrain,'train'),(dvalid_xy,'eval')]
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds,
                  evals=evals, early_stopping_rounds=early_stopping_rounds, 
                  verbose_eval=10)
valid_pred = model.predict(dvalid_x, ntree_limit=model.best_ntree_limit)
print( "XGBoost validation set predictions:" )
print( pd.DataFrame(valid_pred).head() )
print("\nMean absolute validation error:")
mean_absolute_error(y_valid, valid_pred)

if OPTIMIZE_FUDGE_FACTOR:
    mod = QuantReg(y_valid, valid_pred)
    res = mod.fit(q=.5)
    print("\nLAD Fit for Fudge Factor:")
    print(res.summary())

    fudge = res.params[0]
    print("Optimized fudge factor:", fudge)
    print("\nMean absolute validation error with optimized fudge factor: ")
    print(mean_absolute_error(y_valid, fudge*valid_pred))

    fudge **= FUDGE_FACTOR_SCALEDOWN
    print("Scaled down fudge factor:", fudge)
    print("\nMean absolute validation error with scaled down fudge factor: ")
    print(mean_absolute_error(y_valid, fudge*valid_pred))
else:
    fudge=1.0
    
if FIT_FULL_TRAIN_SET:
    if FIT_2017_TRAIN_SET:
        train = pd.read_csv('train_2017.csv')
        properties = pd.read_csv('properties_2017.csv')
        for c in properties.columns:
            properties[c]=properties[c].fillna(-1)
            if properties[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(properties[c].values))
                properties[c] = lbl.transform(list(properties[c].values))
        zip_count = properties['regionidzip'].value_counts().to_dict()
        city_count = properties['regionidcity'].value_counts().to_dict()
        medyear = properties.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()
        meanarea = properties.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()
        medlat = properties.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()
        medlong = properties.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()
        train_df = train.merge(properties, how='left', on='parcelid')
        citystd = train_df.groupby('regionidcity')['logerror'].aggregate("std").to_dict()
        zipstd = train_df.groupby('regionidzip')['logerror'].aggregate("std").to_dict()
        hoodstd = train_df.groupby('regionidneighborhood')['logerror'].aggregate("std").to_dict()
        calculate_features(train_df)
        x_full = train_df.drop(dropvars+droptrain, axis=1)
        y_full = train_df["logerror"].values.astype(np.float32)
        n_full = x_full.shape[0]     
    dtrain = xgb.DMatrix(x_full, y_full)
    num_boost_rounds = int(model.best_ntree_limit*n_full/n_train)
    full_model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds, 
                           evals=[(dtrain,'train')], verbose_eval=10)
    
if not CV_ONLY:
    if FIT_FULL_TRAIN_SET:
        pred = fudge*full_model.predict(dtest)
    else:
        pred = fudge*model.predict(dtest, ntree_limit=model.best_ntree_limit)
        
    print( "XGBoost test set predictions:" )
    print( pd.DataFrame(pred).head() )
    
if MAKE_SUBMISSION and not CV_ONLY:
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

   output.to_csv('xgboost&Validation396_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
   
print("Mean absolute validation error without fudge factor: ", )
print( mean_absolute_error(y_valid, valid_pred) )
if OPTIMIZE_FUDGE_FACTOR:
    print("Mean absolute validation error with fudge factor:")
    print( mean_absolute_error(y_valid, fudge*valid_pred) )
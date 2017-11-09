# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:35:22 2017
@author: sanga
#Credits:
#https://www.kaggle.com/seesee/concise-catboost-starter-ensemble-plb-0-06435
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm
import gc

print("**********************************************************************")
print("*********************All Processing is done for 2016******************")
print("**********************************************************************")
print("\nReading data.....")
train_df_2016 = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
test_df_2016 = pd.read_csv('../input/sample_submission.csv', low_memory=False)
properties_2016 = pd.read_csv('../input/properties_2016.csv', low_memory=False)
# field is named differently in submission
test_df_2016['parcelid'] = test_df_2016['ParcelId']


# similar to the1owl
def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df

print("\nProcessing data.....")
train_df_2016 = add_date_features(train_df_2016)
train_df_2016 = train_df_2016.merge(properties_2016, how='left', on='parcelid')
test_df_2016 = test_df_2016.merge(properties_2016, how='left', on='parcelid')
print("Train: ", train_df_2016.shape)
print("Test: ", test_df_2016.shape)


print("\nRemoving missing data fields.......")
missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df_2016.shape[0]
for c in train_df_2016.columns:
    num_missing = train_df_2016[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))

print("\nRemoving datafields with same datavalues.....")
# exclude where we only have one unique value :D
exclude_unique = []
for c in train_df_2016.columns:
    num_uniques = len(train_df_2016[c].unique())
    if train_df_2016[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))

print("\nDefining training features....")
exclude_other = ['parcelid', 'logerror']  # for indexing/training only
# do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
exclude_other.append('propertyzoningdesc')
train_features = []
for c in train_df_2016.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))

print("\nDefining what features are categorical.....")
cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df_2016[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature_inds.append(i)

print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])


print("\nFilling missing values in data....")
# some out of range int is a good choice
train_df_2016.fillna(-999, inplace=True)
test_df_2016.fillna(-999, inplace=True)

print("Shape of the data.....")
X_train = train_df_2016[train_features]
y_train = train_df_2016.logerror
print("\nTraining data size:")
print(X_train.shape, y_train.shape)

print("Testing data size:")
test_df_2016['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_df_2016 = add_date_features(test_df_2016)
X_test = test_df_2016[train_features]
print(X_test.shape)


print("\nTraining & Testing model......")
num_ensembles = 5
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    # TODO(you): Use CV, tune hyperparameters
    # TODO(you): Use CV, tune hyperparameters
    model = CatBoostRegressor( iterations=127, learning_rate=0.07,
    depth=5, l2_leaf_reg=3, 
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=i)
    model.fit(
        X_train, y_train,
        cat_features=cat_feature_inds)
    y_pred += model.predict(X_test)
y_pred /= num_ensembles

del train_df_2016; del missing_perc_thresh; del exclude_missing;
del test_df_2016; del num_rows; del exclude_unique
del properties_2016; del exclude_other; del train_features
del cat_feature_inds; del cat_unique_thresh; del X_train
del y_train; del X_test; del num_ensembles
gc.collect()


'''*******************************************************************************
**********************************************************************************
*****************************Everything for 2017**********************************
**********************************************************************************
**********************************************************************************
'''
print("**********************************************************************")
print("**********************************************************************")
print("*********************All Processing is done for 2017******************")
print("**********************************************************************")
print("**********************************************************************")
print("\nReading data.....")
train_df = pd.read_csv('../input/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)
test_df = pd.read_csv('../input/sample_submission.csv', low_memory=False)
properties = pd.read_csv('../input/properties_2017.csv', low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']


# similar to the1owl
def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df
    
print("\nProcessing data.....")
train_df = add_date_features(train_df)
train_df = train_df.merge(properties, how='left', on='parcelid')
test_df = test_df.merge(properties, how='left', on='parcelid')
print("Train: ", train_df.shape)
print("Test: ", test_df.shape)


print("\nRemoving missing data fields.......")
missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))

print("\nRemoving datafields with same datavalues.....")
# exclude where we only have one unique value :D
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))

print("\nDefining training features....")
exclude_other = ['parcelid', 'logerror']  # for indexing/training only
# do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
exclude_other.append('propertyzoningdesc')
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))

print("\nDefining what features are categorical.....")
cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature_inds.append(i)
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])


print("\nFilling missing values in data....")
# some out of range int is a good choice
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

print("Shape of the data.....")
X_train = train_df[train_features]
y_train = train_df.logerror
print("\nTraining data size:")
print(X_train.shape, y_train.shape)

print("Testing data size:")
test_df['transactiondate'] = pd.Timestamp('2017-10-01')  # Dummy
test_df = add_date_features(test_df)
X_test = test_df[train_features]
print(X_test.shape)


print("\nTraining & Testing model......")
num_ensembles = 5
y_pred2 = 0.0
for i in tqdm(range(num_ensembles)):
    # TODO(you): Use CV, tune hyperparameters
    # TODO(you): Use CV, tune hyperparameters
    model = CatBoostRegressor( iterations=127, learning_rate=0.07,
    depth=5, l2_leaf_reg=3, 
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=i)
    model.fit(
        X_train, y_train,
        cat_features=cat_feature_inds)
    y_pred2 += model.predict(X_test)
y_pred2 /= num_ensembles

print("\nLoading results into csv....")
submission = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred2, '201711': y_pred2, '201712': y_pred2})
print("\nBefore Adjusting the parcelids................")
print(submission.head())
    
print("\nAfter adjusting parcelID's............")
cols = submission.columns.tolist()
cols = cols[-1:] + cols[:-1]
submission = submission[cols]
print(submission.head())

submission_major = 393
submission.to_csv(
    'catboost_%03d.csv' % (submission_major),
    float_format='%.4f',
    index=False)

'''
print("\nLoading results into csv....")
submission = pd.DataFrame({
    'ParcelId': test_df_2016['parcelid'],
})
test_dates = {
    '201610': pd.Timestamp('2016-09-30'),
    '201611': pd.Timestamp('2016-10-31'),
    '201612': pd.Timestamp('2016-11-30'),
    '201710': pd.Timestamp('2017-09-30'),
    '201711': pd.Timestamp('2017-10-31'),
    '201712': pd.Timestamp('2017-11-30')
}
for label, test_date in test_dates.items():
    print("Predicting for: %s ... " % (label))
    # TODO(you): predict for every `test_date`
    submission[label] = y_pred
'''

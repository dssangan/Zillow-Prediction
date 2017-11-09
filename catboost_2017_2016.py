# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 22:20:57 2017

@author: sanga
"""

#author: dssanga

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm

print("\nReading data.....")
train_df = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2_df = pd.read_csv('../input/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)
properties = pd.read_csv('../input/properties_2016.csv', low_memory=False)
properties2 = pd.read_csv('../input/properties_2017.csv', low_memory=False)
test_df = pd.read_csv('../input/sample_submission.csv', low_memory=False)
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
#2016
train_df = add_date_features(train_df)
train_df = train_df.merge(properties, how='left', on='parcelid')
test_df = test_df.merge(properties, how='left', on='parcelid')
#2017
train2_df = add_date_features(train2_df)
train2_df = train_df.merge(properties2, how='left', on='parcelid')
test2_df = test_df.merge(properties2, how='left', on='parcelid')

print("\n2016 dataset values.......")
print("Train: ", train_df.shape)
print("Test: ", test_df.shape)
print("\n2017 dataset values......")
print("Train: ", train2_df.shape)
print("Test: ", test2_df.shape)

#all process for 2016

print("Everything is for 2016...................")
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
test_df['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_df = add_date_features(test_df)
X_test = test_df[train_features]
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

del model; del num_ensembles;

'''
************************************************
*************Everything for 2017****************
************************************************
'''

missing_perc_thresh = 0.98
exclude2_missing = []
num_rows = train2_df.shape[0]
for c in train2_df.columns:
    num2_missing = train2_df[c].isnull().sum()
    if num2_missing == 0:
        continue
    missing2_frac = num2_missing / float(num_rows)
    if missing2_frac > missing_perc_thresh:
        exclude2_missing.append(c)
print("We exclude: %s" % exclude2_missing)
print(len(exclude2_missing))


print("\nRemoving datafields with same datavalues.....")
# exclude where we only have one unique value :D
exclude2_unique = []
for c in train2_df.columns:
    num2_uniques = len(train2_df[c].unique())
    if train2_df[c].isnull().sum() != 0:
        num2_uniques -= 1
    if num2_uniques == 1:
        exclude2_unique.append(c)
print("We exclude: %s" % exclude2_unique)
print(len(exclude2_unique))

print("\nDefining training features....")
exclude2_other = ['parcelid', 'logerror']  # for indexing/training only
# do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
exclude2_other.append('propertyzoningdesc')
train2_features = []
for c in train2_df.columns:
    if c not in exclude2_missing \
       and c not in exclude2_other and c not in exclude2_unique:
        train2_features.append(c)
print("We use these for training: %s" % train2_features)
print(len(train2_features))

print("\nDefining what features are categorical.....")
cat_feature2_inds = []
cat_unique2_thresh = 1000
for i, c in enumerate(train2_features):
    num2_uniques = len(train2_df[c].unique())
    if num2_uniques < cat_unique2_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature2_inds.append(i)
        
print("Cat features are: %s" % [train2_features[ind] for ind in cat_feature2_inds])


print("\nFilling missing values in data....")
# some out of range int is a good choice
train2_df.fillna(-999, inplace=True)
test2_df.fillna(-999, inplace=True)

print("Shape of the data.....")
X2_train = train2_df[train2_features]
y2_train = train2_df.logerror
print("\nTraining data size:")
print(X2_train.shape, y2_train.shape)

print("Testing data size:")
test2_df['transactiondate'] = pd.Timestamp('2017-12-01')  # Dummy
test2_df = add_date_features(test2_df)
X2_test = test2_df[train2_features]
print(X2_test.shape)


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
        X2_train, y2_train,
        cat_features=cat_feature2_inds)
    y_pred2 += model.predict(X2_test)
y_pred2 /= num_ensembles

print("\nLoading results into csv....")
submission = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred2, '201711': y_pred2, '201712': y_pred2})
    
cols = submission.columns.tolist()
cols = cols[-1:] + cols[:-1]
submission = submission[cols]

'''
print("\nLoading results into csv....")
submission = pd.DataFrame({
    'ParcelId': test_df['parcelid'],
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
submission_major = 395
submission.to_csv(
    'catboost_%03d.csv' % (submission_major),
    float_format='%.4f',
    index=False)

#Credits:
#https://www.kaggle.com/seesee/concise-catboost-starter-ensemble-plb-0-06435

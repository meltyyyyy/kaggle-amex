#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Config:
    name = "EDA/Agg-Forward"

    n_splits = 5
    seed = 2022
    target = "target"

    # Colab Env
    upload_from_colab = True
    api_path = "/content/drive/MyDrive/workspace/kaggle.json"
    drive_path = "/content/drive/MyDrive/workspace/kaggle-amex"

    # Kaggle Env
    kaggle_dataset_path = None

    # Reka Env
    dir_path = '/home/abe/kaggle/kaggle-amex'


# In[ ]:


import os
import json
import warnings
import shutil
import logging
import joblib
import random
import datetime
import sys
import gc
import multiprocessing
import joblib
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
tqdm.pandas()
warnings.filterwarnings('ignore')


# ## Environment Settings

# In[ ]:


INPUT = os.path.join(Config.dir_path, 'input')
OUTPUT = os.path.join(Config.dir_path, 'output')
SUBMISSION = os.path.join(Config.dir_path, 'submissions')
OUTPUT_EXP = os.path.join(OUTPUT, Config.name)
EXP_MODEL = os.path.join(OUTPUT_EXP, "model")
EXP_FIG = os.path.join(OUTPUT_EXP, "fig")
EXP_PREDS = os.path.join(OUTPUT_EXP, "preds")

# make dirs
for d in [INPUT, SUBMISSION, EXP_MODEL, EXP_FIG, EXP_PREDS]:
    os.makedirs(d, exist_ok=True)


# ## Load data

# In[ ]:


train = pd.read_pickle(os.path.join(INPUT, 'train_agg.pkl'), compression='gzip')
test = pd.read_pickle(os.path.join(INPUT, 'test_agg.pkl'), compression='gzip')


# In[ ]:


train.info()


# In[ ]:


train.head()


# ## Evaluation Metric

# In[ ]:


# https://www.kaggle.com/code/inversion/amex-competition-metric-python

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(pd.DataFrame({'target': y_true}), pd.Series(y_pred, name='prediction')),
            True)


# ## Transform data type

# In[ ]:


float64_cols = [col for col in train.columns if train[col].dtype == 'float64']
int64_cols = [col for col in train.columns if train[col].dtype == 'int64']

print(train.info())
print(test.info())
print()
print("-"*50+f' data type transformation '+'-'*50)
print()

def transform_dtype(df):
  for col in df.columns:
    if df[col].dtype == 'float64':
      df[col] = df[col].astype('float16')
    if df[col].dtype == 'float32':
      df[col] = df[col].astype('float16')
    if df[col].dtype == 'int64':
      df[col] = df[col].astype('int8')
    if df[col].dtype == 'int32':
      df[col] = df[col].astype('int8')
  return df

train = transform_dtype(train)
test = transform_dtype(test)

print(train.info())
print(test.info())


# ## Select Features to Use

# In[ ]:


features = []
unuse = ['target', 'customer_ID', 'S_2']

for col in train.columns:
  if col not in unuse:
    features.append(col)

# print(features)


# ## Forward Selection

# In[1]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train[features].values, train[Config.target].values,
                 train_size=0.8, 
                 random_state=Config.seed, 
                 shuffle=True)


# In[2]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from lightgbm import LGBMClassifier, early_stopping

lgb_params = {"learning_rate": 0.01,
              'num_leaves': 127,
              'min_child_samples': 2400}

fit_params = {
    'callbacks': [early_stopping(stopping_rounds=10, verbose=0)],
    'eval_set': [(X_test, y_test)],
    'eval_metric': lgb_amex_metric,
    'verbose': 0
}

model = LGBMClassifier(**lgb_params,
                       boosting_type='gbdt',
                       objective='binary',
                       n_estimators=10000,
                       random_state=Config.seed,
                       force_col_wise=True,
                       n_jobs=16,
                       verbose=-1)

sfs = SFS(model, 
          k_features=10,
          forward=True, 
          cv=5,
          scoring='accuracy', 
          pre_dispatch=32,
          verbose=2)

sfs.fit(X_train, y_train, **fit_params)


# In[ ]:





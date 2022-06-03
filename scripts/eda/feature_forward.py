#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Config:
    name = "EDA/Feature-Forward"

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


train = pd.read_parquet(os.path.join(INPUT, 'train_small.parquet'))
test = pd.read_parquet(os.path.join(INPUT, 'test_small.parquet'))
target = pd.read_csv(os.path.join(INPUT, 'train_labels.csv'), dtype={'customer_ID': 'str', 'target': 'int8'})

with open(os.path.join(INPUT, 'data_types.json'), "r") as rf:
    dtypes = json.load(rf)
train = train.astype(dtypes)
test = test.astype(dtypes)


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


# ## Preprocess

# In[ ]:


from sklearn.preprocessing import LabelEncoder

cat_cols = [col for col in train.columns if train[col].dtype == 'category']
for col in cat_cols:
    le = LabelEncoder()
    le.fit(train[col])
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


# ## Create target

# In[ ]:


train = train.groupby('customer_ID')  .tail(1).set_index('customer_ID', drop=True)  .sort_index()  .merge(target.set_index('customer_ID', drop=True), left_index=True, right_index=True)
test =  test.groupby('customer_ID')  .tail(1).set_index('customer_ID', drop=True)  .sort_index()


# ## Select Features to Use

# In[ ]:


features = []
unuse = ['target', 'customer_ID', 'S_2']

for col in train.columns:
  if col not in unuse:
    features.append(col)

# print(features)


# ## Forward Selection

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train[features].values, train[Config.target].values,
                 train_size=0.8, 
                 random_state=Config.seed, 
                 shuffle=True)


# In[ ]:


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


# ## Training

# In[13]:


from lightgbm.plotting import plot_metric
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold

def fit_lgbm(X, y, params=None):
  models = []
  scores = []

  skf = StratifiedKFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
  
  for fold, (train_indices, valid_indices) in enumerate(tqdm(skf.split(X, y))):
    print("-"*50+f' fold{fold} '+'-'*50)
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_valid, y_valid = X.iloc[valid_indices], y.iloc[valid_indices]

    model = LGBMClassifier(**params,
                           boosting_type='gbdt',
                           objective='binary',
                           n_estimators=10000,
                           random_state=Config.seed,
                           n_jobs=-1,
                           verbose=0)
    
    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              eval_names=['train', 'valid'],
              eval_metric=lgb_amex_metric,
              callbacks=[early_stopping(stopping_rounds=10, verbose=1)],
              verbose=50)
    
    # ------------------- prediction -------------------
    pred = model.predict_proba(X_valid)[:, 1]
    score = amex_metric(pd.DataFrame({'target': y_valid.values}), pd.Series(pred, name='prediction'))

    # ------------------- plot -------------------
    plot_metric(model)

    # ------------------- save -------------------
    file = f'{EXP_MODEL}/lgbm_fold{fold}.pkl'
    joblib.dump(model, file)
    scores.append(score)
    models.append(model)
    print(f'fold{fold} amex meric: {score}')
    print()

  print(f"OOF Score: {np.mean(scores):.5f}")
  return models

def inference_lgbm(models, X):
    pred = np.array([model.predict_proba(X) for model in models])
    pred = np.mean(pred, axis=0)[:, 1]
    return pred


# In[ ]:





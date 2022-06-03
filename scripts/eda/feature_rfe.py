#!/usr/bin/env python
# coding: utf-8

# In[87]:


class Config:
    name = "EDA/Feature-RFE"

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


# In[88]:


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

# In[89]:


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

# In[90]:


train = pd.read_parquet(os.path.join(INPUT, 'train.parquet'))
test = pd.read_parquet(os.path.join(INPUT, 'test.parquet'))
target = pd.read_csv(os.path.join(INPUT, 'train_labels.csv'), dtype={'customer_ID': 'str', 'target': 'int8'})

with open(os.path.join(INPUT, 'data_types.json'), "r") as rf:
    dtypes = json.load(rf)
train = train.astype(dtypes)
test = test.astype(dtypes)


# In[91]:


train.info()


# In[92]:


train.head()


# ## Evaluation Metric

# In[93]:


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

# In[94]:


from sklearn.preprocessing import LabelEncoder

cat_cols = [col for col in train.columns if train[col].dtype == 'category']
for col in cat_cols:
    le = LabelEncoder()
    le.fit(train[col])
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


# ## Create target

# In[95]:


train = train.groupby('customer_ID')  .tail(1).set_index('customer_ID', drop=True)  .sort_index()  .merge(target.set_index('customer_ID', drop=True), left_index=True, right_index=True)
test =  test.groupby('customer_ID')  .tail(1).set_index('customer_ID', drop=True)  .sort_index()
train.reset_index(drop=False, inplace=True)
test.reset_index(drop=False, inplace=True)


# ## Select Features to Use

# In[96]:


features = []
unuse = ['target', 'customer_ID', 'S_2']

for col in train.columns:
  if col not in unuse:
    features.append(col)

# print(features)


# ## Forward Selection

# In[97]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train[features].values, train[Config.target].values,
                 train_size=0.8,
                 random_state=Config.seed,
                 shuffle=True)


# In[98]:


from sklearn.feature_selection import RFE
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
                       n_jobs=32,
                       verbose=-1)

rfe = RFE(model,
          n_features_to_select=100,
          step=1,
          verbose=1)

rfe.fit(X_train, y_train, **fit_params)


# ## Generate new train data

# In[ ]:


train_new = pd.DataFrame(rfe.transform(train[features]),
                     columns=train[features].columns.values[rfe.get_support()])
result = pd.DataFrame(rfe.get_support(), index=train[features].columns.values, columns=['used'])
result['ranking'] = rfe.ranking_
result.sort_values('ranking', ascending=True).reset_index(drop=False, inplace=True)
result.to_csv(f'{EXP_MODEL}/rfe_features.csv', index=False)


# ## Training

# In[ ]:


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
                           force_col_wise=True,
                           n_jobs=32,
                           verbose=-1)

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              eval_names=['train', 'valid'],
              eval_metric=lgb_amex_metric,
              callbacks=[early_stopping(stopping_rounds=10, verbose=0)],
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


lgb_params = {"learning_rate": 0.01,
              'num_leaves': 127,
              'min_child_samples': 2400}

features = list(train_new.columns)
models = fit_lgbm(train[features], train[Config.target], params=lgb_params)
# models = [joblib.load(f'{EXP_MODEL}/lgbm_fold{i}.pkl') for i in range(Config.n_splits)]
pred = inference_lgbm(models, test[features])


# ## Plot importance

# In[ ]:


def plot_importances(models):
    importance_df = pd.DataFrame(models[0].feature_importances_,
                                 index=features,
                                 columns=['importance'])\
                        .sort_values("importance", ascending=False)

    plt.subplots(figsize=(len(features) // 4, 5))
    plt.bar(importance_df.index, importance_df.importance)
    plt.grid()
    plt.xticks(rotation=90)
    plt.ylabel("importance")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{EXP_FIG}/importance.png')

plot_importances(models)


# ## Submission

# In[ ]:


sub = pd.DataFrame({'customer_ID': test.index,
                    'prediction': pred})
sub.to_csv(f'{EXP_PREDS}/submission.csv', index=False)


# In[ ]:


get_ipython().system(' kaggle competitions submit -c amex-default-prediction -f /home/abe/kaggle/kaggle-amex/submissions/submission.csv -m "Recuresive Feature Elimination for Baseline"')


# In[ ]:





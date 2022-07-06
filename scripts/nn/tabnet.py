#!/usr/bin/env python
# coding: utf-8

# ### Basic configuration
"""Summery
fold0 amex meric: 0.7871165449748297
fold1 amex meric: 0.7897137328076916
fold2 amex meric: 0.7877178072893678
fold3 amex meric: 0.7895787409341312
fold4 amex meric: 0.7904035770918295
OOF Score: 0.78891
"""
# In[1]:


class Config:
    notebook = "NN/TabNet"
    script = "nn/tabnet"

    n_splits = 5
    seed = 2020
    target = "target"

    batch_size = 512
    max_epochs = 60

    # Colab Env
    api_path = "/content/drive/MyDrive/workspace/kaggle.json"
    drive_path = "/content/drive/MyDrive/workspace/kaggle-amex"

    # Kaggle Env
    kaggle_dataset_path = None

    # Reka Env
    dir_path = '/home/abe/kaggle/kaggle-amex'

    def is_notebook():
        if 'get_ipython' not in globals():
            return False
        env_name = get_ipython().__class__.__name__  # type: ignore
        if env_name == 'TerminalInteractiveShell':
            return False
        return True


# ### Import basic libraries

# In[2]:


from tqdm.auto import tqdm
import seaborn as sns
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
import subprocess
from subprocess import PIPE
import ntpath

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
sns.set_palette("winter_r")

tqdm.pandas()
warnings.filterwarnings('ignore')


# ### Seeding

# In[3]:


random.seed(Config.seed)
np.random.seed(Config.seed)
os.environ['PYTHONHASHSEED'] = str(Config.seed)


# ### Path configuration

# In[4]:


INPUT = os.path.join(Config.dir_path, 'input')
OUTPUT = os.path.join(Config.dir_path, 'output')
SUBMISSION = os.path.join(Config.dir_path, 'submissions')
OUTPUT_EXP = os.path.join(OUTPUT, Config.script)
EXP_MODEL = os.path.join(OUTPUT_EXP, "model")
EXP_FIG = os.path.join(OUTPUT_EXP, "fig")
NOTEBOOK = os.path.join(Config.dir_path, "Notebooks")
SCRIPT = os.path.join(Config.dir_path, "scripts")

# make dirs
for dir in [
        INPUT,
        OUTPUT,
        SUBMISSION,
        OUTPUT_EXP,
        EXP_MODEL,
        EXP_FIG,
        NOTEBOOK,
        SCRIPT]:
    os.makedirs(dir, exist_ok=True)

if Config.is_notebook():
    notebook_path = os.path.join(NOTEBOOK, Config.notebook + ".ipynb")
    script_path = os.path.join(SCRIPT, Config.script + ".py")
    dir, _ = ntpath.split(script_path)
    subprocess.run(f"mkdir -p {dir}; touch {script_path}",
                   shell=True,
                   stdout=PIPE,
                   stderr=PIPE,
                   text=True)
    subprocess.run(
        f"jupyter nbconvert --to python {notebook_path} --output {script_path}",
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        text=True)


# ## Load data

# In[5]:


train = pd.read_parquet(os.path.join(INPUT, 'train.parquet'))
target = pd.read_csv(
    os.path.join(
        INPUT,
        'train_labels.csv'),
    dtype={
        'customer_ID': 'str',
        'target': 'int8'})
# train = pd.read_parquet(os.path.join(INPUT, 'train_small.parquet') if COLAB else 'train_small.parquet')
test = pd.read_parquet(os.path.join(INPUT, 'test.parquet'))


# In[6]:


train.info()


# In[7]:


train.head()


# ## Amex metric

# In[8]:


def amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)


# In[9]:


from pytorch_tabnet.metrics import Metric

class AmexTabnet(Metric):

  def __init__(self):
    self._name = 'amex_tabnet'
    self._maximize = True

  def __call__(self, y_true, y_pred):
    amex = amex_metric(y_true, y_pred[:, 1])
    return max(amex, 0.)


# ## Feature Eng

# In[10]:


from sklearn.preprocessing import OneHotEncoder

features_avg = ['B_11', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_2',
                'B_20', 'B_28', 'B_29', 'B_3', 'B_33', 'B_36', 'B_37', 'B_4', 'B_42',
                'B_5', 'B_8', 'B_9', 'D_102', 'D_103', 'D_105', 'D_111', 'D_112', 'D_113',
                'D_115', 'D_118', 'D_119', 'D_121', 'D_124', 'D_128', 'D_129', 'D_131',
                'D_132', 'D_133', 'D_139', 'D_140', 'D_141', 'D_143', 'D_144', 'D_145',
                'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48',
                'D_49', 'D_50', 'D_51', 'D_52', 'D_56', 'D_58', 'D_62', 'D_70', 'D_71',
                'D_72', 'D_74', 'D_75', 'D_79', 'D_81', 'D_83', 'D_84', 'D_88', 'D_91',
                'P_2', 'P_3', 'R_1', 'R_10', 'R_11', 'R_13', 'R_18', 'R_19', 'R_2', 'R_26',
                'R_27', 'R_28', 'R_3', 'S_11', 'S_12', 'S_22', 'S_23', 'S_24', 'S_26',
                'S_27', 'S_5', 'S_7', 'S_8', ]
features_min = ['B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_2', 'B_20', 'B_22',
                'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_33', 'B_36', 'B_4', 'B_42',
                'B_5', 'B_9', 'D_102', 'D_103', 'D_107', 'D_109', 'D_110', 'D_111',
                'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128',
                'D_129', 'D_132', 'D_133', 'D_139', 'D_140', 'D_141', 'D_143', 'D_144',
                'D_145', 'D_39', 'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51',
                'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70',
                'D_71', 'D_74', 'D_75', 'D_78', 'D_79', 'D_81', 'D_83', 'D_84', 'D_86',
                'D_88', 'D_96', 'P_2', 'P_3', 'P_4', 'R_1', 'R_11', 'R_13', 'R_17', 'R_19',
                'R_2', 'R_27', 'R_28', 'R_4', 'R_5', 'R_8', 'S_11', 'S_12', 'S_23', 'S_25',
                'S_3', 'S_5', 'S_7', 'S_9', ]
features_max = ['B_1', 'B_11', 'B_13', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_2',
                'B_22', 'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_31', 'B_33', 'B_36',
                'B_4', 'B_42', 'B_5', 'B_7', 'B_9', 'D_102', 'D_103', 'D_105', 'D_109',
                'D_110', 'D_112', 'D_113', 'D_115', 'D_121', 'D_124', 'D_128', 'D_129',
                'D_131', 'D_139', 'D_141', 'D_144', 'D_145', 'D_39', 'D_41', 'D_42',
                'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_52',
                'D_53', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_72', 'D_74',
                'D_75', 'D_79', 'D_81', 'D_83', 'D_84', 'D_88', 'D_89', 'P_2', 'P_3',
                'R_1', 'R_10', 'R_11', 'R_26', 'R_28', 'R_3', 'R_4', 'R_5', 'R_7', 'R_8',
                'S_11', 'S_12', 'S_23', 'S_25', 'S_26', 'S_27', 'S_3', 'S_5', 'S_7', 'S_8', ]
features_last = ['B_1', 'B_11', 'B_12', 'B_13', 'B_14', 'B_16', 'B_18', 'B_19', 'B_2',
                 'B_20', 'B_21', 'B_24', 'B_27', 'B_28', 'B_29', 'B_3', 'B_30', 'B_31',
                 'B_33', 'B_36', 'B_37', 'B_38', 'B_39', 'B_4', 'B_40', 'B_42', 'B_5',
                 'B_8', 'B_9', 'D_102', 'D_105', 'D_106', 'D_107', 'D_108', 'D_110',
                 'D_111', 'D_112', 'D_113', 'D_114', 'D_115', 'D_116', 'D_117', 'D_118',
                 'D_119', 'D_120', 'D_121', 'D_124', 'D_126', 'D_128', 'D_129', 'D_131',
                 'D_132', 'D_133', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142',
                 'D_143', 'D_144', 'D_145', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45',
                 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_55',
                 'D_56', 'D_59', 'D_60', 'D_62', 'D_63', 'D_64', 'D_66', 'D_68', 'D_70',
                 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_77', 'D_78', 'D_81', 'D_82',
                 'D_83', 'D_84', 'D_88', 'D_89', 'D_91', 'D_94', 'D_96', 'P_2', 'P_3',
                 'P_4', 'R_1', 'R_10', 'R_11', 'R_12', 'R_13', 'R_16', 'R_17', 'R_18',
                 'R_19', 'R_25', 'R_28', 'R_3', 'R_4', 'R_5', 'R_8', 'S_11', 'S_12',
                 'S_23', 'S_25', 'S_26', 'S_27', 'S_3', 'S_5', 'S_7', 'S_8', 'S_9', ]
features_categorical = ['B_30_last', 'B_38_last', 'D_114_last', 'D_116_last',
                        'D_117_last', 'D_120_last', 'D_126_last',
                        'D_63_last', 'D_64_last', 'D_66_last', 'D_68_last']

encoder = OneHotEncoder(drop='first', sparse=False, dtype=np.float32, handle_unknown='ignore')

def add_features(df, is_train):
    cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    last = (cid != np.roll(cid, -1))
    df_avg = (df
              .groupby(cid)
              .mean()[features_avg]
              .rename(columns={f: f"{f}_avg" for f in features_avg})
             )
    gc.collect()
    df_min = (df
              .groupby(cid)
              .min()[features_min]
              .rename(columns={f: f"{f}_min" for f in features_min})
             )
    gc.collect()
    df_max = (df
              .groupby(cid)
              .max()[features_max]
              .rename(columns={f: f"{f}_max" for f in features_max})
             )
    gc.collect()
    df_last = (df.loc[last, features_last]
          .rename(columns={f: f"{f}_last" for f in features_last})
          .set_index(np.asarray(cid[last]))
         )
    gc.collect()

    df_categorical = df_last[features_categorical].astype(object)
    features_not_cat = [f for f in df_last.columns if f not in features_categorical]
    if is_train:
        encoder.fit(df_categorical)
    df_categorical = pd.DataFrame(encoder.transform(df_categorical).astype(np.float16),
                                  index=df_categorical.index).rename(columns=str)

    df = pd.concat([df_last[features_not_cat], df_categorical, df_avg, df_min, df_max], axis=1)
    df.fillna(value=0, inplace=True)


    del df_avg, df_max, df_min, df_last, df_categorical, cid, last, features_not_cat
    return df

train = add_features(train, True)
test = add_features(test, False)


# ## Create target

# In[11]:


train = train.join(target.set_index('customer_ID'), how='left')


# ## Select features to use

# In[12]:


features = []
unuse = ['customer_ID', 'S_2', 'target']

features = [col for col in train.columns if col not in unuse]


# ## Training

# In[13]:


from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.model_selection import StratifiedKFold


def fit_tabnet(X, y):
    models = []
    scores = []
    feature_importances = pd.DataFrame()
    feature_importances["feature"] = X.columns.tolist()
    stats = pd.DataFrame()
    explain_matrices = []
    masks = []

    skf = StratifiedKFold(
        n_splits=Config.n_splits,
        shuffle=True,
        random_state=Config.seed)

    for fold, (train_indices, valid_indices) in enumerate(
            skf.split(X, y)):
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_valid, y_valid = X.iloc[valid_indices], y.iloc[valid_indices]
        print('#'*25)
        print('### Training data shapes', X_train.shape, y_train.shape)
        print('### Validation data shapes', X_valid.shape, y_valid.shape)
        print('#'*25)


        model = TabNetClassifier(n_d = 32,
                             n_a = 32,
                             n_steps = 3,
                             gamma = 1.3,
                             n_independent = 2,
                             n_shared = 2,
                             momentum = 0.02,
                             clip_value = None,
                             lambda_sparse = 1e-3,
                             optimizer_fn = torch.optim.Adam,
                             optimizer_params = dict(lr = 1e-3, weight_decay=1e-3),
                             scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                             scheduler_params = {'T_0':5,
                                                 'eta_min':1e-4,
                                                 'T_mult':1,
                                                 'last_epoch':-1},
                             mask_type = 'entmax',
                             seed = Config.seed)

        model.fit(np.array(X_train),
              np.array(y_train.values.ravel()),
              eval_set = [(np.array(X_valid), np.array(y_valid.values.ravel()))],
              max_epochs = Config.max_epochs,
              patience = 50,
              batch_size = Config.batch_size,
              eval_metric = ['auc', 'accuracy', AmexTabnet])

        # ------------------- prediction -------------------
        pred = model.predict_proba(X_valid.values)[:,1]
        score = amex_metric(y_valid, pred)


        # Loss ,metric, improtances tracking
        stats[f'fold{fold+1}_train_loss'] = model.history['loss']
        stats[f'fold{fold+1}_val_metric'] = model.history['val_0_amex_tabnet']
        feature_importances[f"importance_fold{fold}+1"] = model.feature_importances_

        # model explanability
        explain_matrix, mask = model.explain(X_valid.values)
        explain_matrices.append(explain_matrix)
        masks.append(mask[0])
        masks.append(mask[1])

        scores.append(score)
        models.append(model)
        print(f'fold{fold} amex meric: {score}')
        print()

    print(f"OOF Score: {np.mean(scores):.5f}")
    return models, explain_matrix, masks, stats, feature_importances


def inference_tabnet(models, X):
    pred = np.array([model.predict_proba(X.values) for model in models])
    pred = np.mean(pred, axis=0)[:, 1]
    return pred


# In[14]:


models, explain_matrix, masks, stats, feature_importances = fit_tabnet(train[features], train[Config.target])
pred = inference_tabnet(models, test[features])


# ## Plot metric

# In[15]:


for i in stats.filter(like='train', axis=1).columns.tolist():
    plt.plot(stats[i], label=str(i))
plt.title('Train loss')
plt.legend()
plt.savefig(f'{EXP_FIG}/loss.png')
plt.show()
plt.close()


# In[16]:


for i in stats.filter(like='val', axis=1).columns.tolist():
    plt.plot(stats[i], label=str(i))
plt.title('Train RMSPE')
plt.legend()
plt.savefig(f'{EXP_FIG}/rmspe.png')
plt.show()
plt.close()


# In[18]:


feature_importances['mean_importance']=feature_importances[['importance_fold0+1','importance_fold1+1']].mean(axis=1)
feature_importances.sort_values(by='mean_importance', ascending=False, inplace=True)
sns.barplot(y=feature_importances['feature'][:50],x=feature_importances['mean_importance'][:50])
plt.title('Mean Feature Importance by Folds')
plt.savefig(f'{EXP_FIG}/importance.png')
plt.show()
plt.close()


# In[19]:


sub = pd.DataFrame({'customer_ID': test.index,
                    'prediction': pred})
sub.to_csv(f'{SUBMISSION}/tabnet.csv', index=False)


# In[ ]:





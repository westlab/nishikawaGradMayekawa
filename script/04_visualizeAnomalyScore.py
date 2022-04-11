import os
import codecs
import copy
import sys
import time

import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import mplConf

RESULT_DIR = './../result'

rcParams["figure.subplot.left"] = 0.1
rcParams["figure.subplot.bottom"] = 0.1
rcParams["figure.subplot.right"] = 0.98
rcParams["figure.subplot.top"] = 0.98

### anomaly score of channel 00
all_data = np.load(f'{RESULT_DIR}/ch00/result_data_all.npz')
pred_normal_valid = all_data['valid_pred_normal']
pred_anomaly_valid = all_data['valid_pred_anomaly']

fig , axes = plt.subplots(1, 1)
axes.hist(pred_normal_valid, bins=50, color='c', alpha=0.9, label='normal')
axes.hist(pred_anomaly_valid, bins= 300, color='r', alpha=0.9, label='abnormal')
axes.set_xlim(0.8, 1.5)
axes.set_xlabel("anomaly score", weight = "light")
axes.set_ylabel("num sample", weight = "light")
axes.legend()
plt.savefig(f'{RESULT_DIR}/anomaly_score_one.png')


### anomaly score of all channel
df = pd.read_csv(f'{RESULT_DIR}/final_prediction.csv')
label = df['label'].values
score_naive = df['score'].values
score_naive_normal = df[df['label'] == 0]['score'].values
score_naive_anomaly = df[df['label'] == 1]['score'].values
pred_naive = df['pred']
### naive-sum
fig , axes = plt.subplots(1, 1)
axes.hist(score_naive_normal, bins=50, color='c', alpha=0.9, label='normal')
axes.hist(score_naive_anomaly, bins=200, color='r', alpha=0.9, label='abnormal')
axes.set_xlim(10.5, 16)
axes.set_xlabel("anomaly score", weight = "light")
axes.set_ylabel("num sample", weight = "light")
axes.legend()
plt.savefig(f'{RESULT_DIR}/anomaly_score_naive.png')
### regularized-sum
score_reg = df['score_regularized'].values
score_reg_normal = df[df['label'] == 0]['score_regularized'].values
score_reg_anomaly = df[df['label'] == 1]['score_regularized'].values
pred_reg = df['pred_regularized']
fig , axes = plt.subplots(1, 1)
axes.hist(score_reg_normal, bins=100, color='c', alpha=0.9, label='normal')
axes.hist(score_reg_anomaly, bins=400, color='r', alpha=0.9, label='abnormal')
axes.set_xlim(-20, 200)
axes.set_xlabel("anomaly score", weight = "light")
axes.set_ylabel("num sample", weight = "light")
axes.legend()
plt.savefig(f'{RESULT_DIR}/anomaly_score_reg.png')


### ROC curve
fpr_naive, tpr_naive, _ = roc_curve(label, score_naive)
fpr_reg, tpr_reg, _ = roc_curve(label, score_reg)

fig , axes = plt.subplots(1, 1)

axes.plot(fpr_naive, tpr_naive, label='naive (AUC: 0.97)', alpha=0.7)
axes.plot(fpr_reg, tpr_reg, label='regularized (AUC: 0.98)', alpha=0.7)

axes.set_xlabel("False Positive Rate", weight = "light")
axes.set_ylabel("True Positve Rate", weight = "light")
axes.legend()

plt.savefig(f'{RESULT_DIR}/ROC_curve.png')
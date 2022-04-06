import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import gamma
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

RESULT_DIR = "./../result"

def myGamma(z, alpha, scale):
    return gamma.pdf(z, alpha, scale=scale)

def myGammaFit(pred_normal_valid):
    y, _x = np.histogram(pred_normal_valid, bins=100, density=True)
    x = []
    for i in range(len(_x) - 1):
        x.append((_x[i+1] + _x[i]) / 2)
    x = np.array(x)

    popt, _ = curve_fit(myGamma, x, y, maxfev=50000)
    mean, var, _, _ = gamma.stats(popt[0], scale=popt[1], moments='mvsk')
    std = np.sqrt(var)

    print(f'{popt[0]} | {popt[1]} | {mean} | {var}')
    return mean, std

if __name__ == '__main__':
    ### log
    logFile = f'{RESULT_DIR}/ensamble_log.txt'
    if os.path.exists(logFile):
        os.remove(logFile)
    sys.stdout = open(logFile, 'a+')

    ### calculate Youden`s Index
    mean = np.empty(8)
    std = np.empty(8)
    sum_pred_regularized_valid = None
    sum_pred_valid = None
    label_valid = None

    print('alpha | scale | mean | var')
    threshold_each = []
    for channel in range(8):
        ### load data
        data = np.load(f'{RESULT_DIR}/ch0{channel}/result_data_all.npz')
        pred_normal_valid = data['valid_pred_normal']
        pred_anomaly_valid = data['valid_pred_anomaly']
        if channel == 0:
            label_valid = np.concatenate([
                np.zeros_like(pred_normal_valid),
                np.ones_like(pred_anomaly_valid)
            ])
            sum_pred_regularized_valid = np.zeros_like(label_valid)
            sum_pred_valid = np.zeros_like(label_valid)
        ### fit Gamma distribution
        res_mean, res_std = myGammaFit(pred_normal_valid)
        mean[channel] = res_mean
        std[channel] = res_std
        ### regularize && sum
        pred = np.concatenate([pred_normal_valid, pred_anomaly_valid])
        sum_pred_regularized_valid += (pred - res_mean) / res_std
        sum_pred_valid += pred
        ### caluculate Youden Index of each channel
        fpr, tpr, thres = roc_curve(label_valid, pred)
        idx = np.argmax(tpr - fpr)
        threshold_each.append(thres[idx])


    ### calculate Yoden Index for naive
    fpr, tpr, thres = roc_curve(label_valid, sum_pred_valid)
    idx = np.argmax(tpr - fpr)
    threshold_youden = thres[idx]
    ### calculate Yoden Index for regularized
    fpr, tpr, thres = roc_curve(label_valid, sum_pred_regularized_valid)
    idx = np.argmax(tpr - fpr)
    threshold_youden_regularized = thres[idx]

    ### predict
    sum_pred_test = None
    sum_pred_regularized_test = None
    sum_pred_each_test = None
    filenames = None
    for channel in range(8):
        ### load data
        data = np.load(f'{RESULT_DIR}/ch0{channel}/result_data_all.npz')
        prediction = data['test_prediction']
        if channel == 0:
            filenames = data['test_filename']
            sum_pred_test = np.zeros_like(prediction)
            sum_pred_regularized_test = np.zeros_like(prediction)
            sum_pred_each_test = np.zeros_like(prediction)
        ### redularize & sum
        sum_pred_test += prediction
        sum_pred_regularized_test += (prediction - mean[channel]) / std[channel]
        sum_pred_each_test += (prediction > threshold_each[channel]).astype(np.int)

    ### save as csv
    df_pred = pd.DataFrame()
    df_pred['filename'] = filenames
    df_pred['label'] = (df_pred['filename'].str[1:5] > '8817').astype(np.int)
    df_pred['score'] = sum_pred_test
    df_pred['score_regularized'] = sum_pred_regularized_test
    df_pred['pred'] = (df_pred['score'] > threshold_youden).astype(np.int)
    df_pred['pred_regularized'] = (df_pred['score_regularized'] > threshold_youden_regularized).astype(np.int)
    df_pred['pred_comp'] = (sum_pred_each_test > 0).astype(np.int)
    df_pred.to_csv(f'{RESULT_DIR}/final_prediction.csv', index=None)

    ### logging for naive
    logFile = f'{RESULT_DIR}/report_naive.txt'
    if os.path.exists(logFile):
        os.remove(logFile)
    sys.stdout = open(logFile, 'a+')
    
    print('ROC AUC score with score             :', roc_auc_score(df_pred['label'], df_pred['score']))
    
    print('\n', '=' * 20, 'confusion matrix', '=' * 20)
    tn, fp, fn, tp = confusion_matrix(df_pred['label'], df_pred['pred']).ravel()
    print(f'              prediction')
    print(f'          | normal | anomaly |')
    print(f' normal   |    {tn:3} |    {fp:3}  |')
    print(f' anomaly  |    {fn:3} |    {tp:3}  |')
    print('\n')
    print('Accuracy : ', (tp + tn) / 847)
    print('Recall   : ', tp / (tp + fn))
    print('Precision: ', tp / (tp + fp))
    print('F1-score : ', 2 * tp / (2*tp + fp + fn))
    
    print('\n', '=' * 20, 'classification report', '=' * 20)
    print(classification_report(df_pred['label'], df_pred['pred']))

    ### logging for regularized
    logFile = f'{RESULT_DIR}/report_regularized.txt'
    if os.path.exists(logFile):
        os.remove(logFile)
    sys.stdout = open(logFile, 'a+')
    
    print('ROC AUC score with reguralized-score :', roc_auc_score(df_pred['label'], df_pred['score_regularized']))
    
    print('\n', '=' * 20, 'confusion matrix', '=' * 20)
    tn, fp, fn, tp = confusion_matrix(df_pred['label'], df_pred['pred_regularized']).ravel()
    print(f'              prediction')
    print(f'          | normal | anomaly |')
    print(f' normal   |    {tn:3} |    {fp:3}  |')
    print(f' anomaly  |    {fn:3} |    {tp:3}  |')
    print('\n')
    print('Accuracy : ', (tp + tn) / (tp + tn + fp + fn))
    print('Recall   : ', tp / (tp + fn))
    print('Precision: ', tp / (tp + fp))
    print('F1-score : ', 2 * tp / (2*tp + fp + fn))
    
    print('\n', '=' * 20, 'classification report', '=' * 20)
    print(classification_report(df_pred['label'], df_pred['pred_regularized']))

    ### logging for comp
    logFile = f'{RESULT_DIR}/report_comp.txt'
    if os.path.exists(logFile):
        os.remove(logFile)
    sys.stdout = open(logFile, 'a+')

    print('\n', '=' * 20, 'confusion matrix', '=' * 20)
    tn, fp, fn, tp = confusion_matrix(df_pred['label'], df_pred['pred_comp']).ravel()
    print(f'              prediction')
    print(f'          | normal | anomaly |')
    print(f' normal   |    {tn:3} |    {fp:3}  |')
    print(f' anomaly  |    {fn:3} |    {tp:3}  |')
    print('\n')
    print('Accuracy : ', (tp + tn) / 847)
    print('Recall   : ', tp / (tp + fn))
    print('Precision: ', tp / (tp + fp))
    print('F1-score : ', 2 * tp / (2*tp + fp + fn))
    
    print('\n', '=' * 20, 'classification report', '=' * 20)
    print(classification_report(df_pred['label'], df_pred['pred_comp']))
import os
import random
import sys

import librosa
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data(data_dir_path, data_type):
    if data_type == 'train':
        return np.load(data_dir_path + 'train_data.npy')
    elif data_type == 'valid':
        return np.load(data_dir_path + 'valid_normal.npy'), np.load(data_dir_path + 'valid_anomaly.npy')
    elif data_type == 'test':
        return np.load(data_dir_path + 'test_normal.npy'), np.load(data_dir_path + 'test_anomaly.npy')
    else:
        print(f'Error: data_type:{data_type} is undefined', file=sys.stderr)
        sys.exit(1)

def visualize_list(target_list, save_path, xlabel, ylabel):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    plt.plot(target_list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.clf()
    plt.close()

def visualize_anomaly_score(normal_pred, anomaly_pred, save_dir):
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    sns.histplot(normal_pred, kde=False, label='normal', binwidth=1, color='c')
    sns.histplot(anomaly_pred, kde=False, label='anomaly', binwidth=1, color='r')
    plt.legend()
    plt.xlabel('anomaly score')
    plt.ylabel('count')
    # plt.xlim(0, 100)
    plt.savefig(save_dir + 'error_density.png')
    plt.clf()
    plt.close()

def calucurate_AUC(normal_pred, anomaly_pred, save_dir):
    y_true = np.concatenate([np.zeros_like(normal_pred), np.ones_like(anomaly_pred)])
    y_score = np.concatenate([normal_pred, anomaly_pred])
    if save_dir != None:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Set1')
        plt.plot(fpr, tpr)
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.grid()
        plt.savefig(save_dir + '/roc_curve.png')
        plt.clf()
        plt.close()
    return roc_auc_score(y_true, y_score)

def save_image_magma(x, xhat, file_name, height, width, save_dir):
    magma = cm.get_cmap('magma')
    x = x.to('cpu').reshape(x.shape[0], 1, height, width)#.detach().numpy().copy()
    xhat = xhat.to('cpu').reshape(x.shape[0], 1, height, width)#.detach().numpy().copy()
    save_image(x, f"{save_dir}/{file_name}_input.png", nrow=1, padding=20, pad_value=255, normalize=True)
    save_image(xhat, f"{save_dir}/{file_name}_output.png", nrow=1, padding=20, pad_value=255, normalize=True)
    save_image(x-xhat, f"{save_dir}/{file_name}_diff.png", nrow=1, padding=20, pad_value=255, normalize=True)
    # x_trans = []
    # for i in range(len(x)):
    #     x_trans.append(magma(x[i, 0, :, :]).reshape(4, height, width))

def getSpecPhaseFromTimeseries(data, sec, nFFT, returnPhase=False):
    stftMat = librosa.stft(
        y = data[:16000 * sec],
        n_fft = nFFT
    )
    spectrogram = np.abs(stftMat)
    spectrogram = np.log2(spectrogram)
    if returnPhase:
        phase = np.angle(stftMat)
        return spectrogram, phase
    else:
        return spectrogram

class myDataset(Dataset): 
    def __init__(self, path_list, channel, sec, n_fft, transforms, isTest=False):
        super().__init__()
        self.path_list = path_list
        self.channel = channel
        self.sec = sec
        self.n_fft = n_fft
        self.transforms = transforms
        self.isTest = isTest
    def __len__(self):
        return len(self.path_list)
    def __getitem__(self, idx):
        file_name = self.path_list[idx, self.channel]
        path = f"./../data/splitData/{file_name}.npy"
        data = np.load(path).astype(np.float).reshape(-1)
        if self.isTest:
            spectrogram, phase = getSpecPhaseFromTimeseries(data, self.sec, self.n_fft, returnPhase=True)
            spectrogram = self.transforms(spectrogram)
            return file_name, spectrogram.type(torch.FloatTensor), phase
        else:
            spectrogram = getSpecPhaseFromTimeseries(data, self.sec, self.n_fft)
            spectrogram = self.transforms(spectrogram)
            return file_name, spectrogram.type(torch.FloatTensor)
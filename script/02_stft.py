import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import scipy as sp

import numpy as np
from tqdm import tqdm
import librosa
import librosa.display

### STFT parameters
SEC = 5
N_FFT = 512

### input & output data
PATH_DATA = np.load('./../data/train_test_paths.npz')
DATA_DIR = './../data/splitData'
SAVE_DIR = f'./../data/stftData'
# os.makedirs(SAVE_DIR)
# os.makedirs(f"{SAVE_DIR}/spectrogram")
# os.makedirs(f"{SAVE_DIR}/phase")

def create_stftMat_from_path(path_list):
    for i in tqdm(range(len(path_list))):
        for channel in range(8):
            path = f"{DATA_DIR}/{path_list[i][channel]}.npy"
            y = np.load(path).astype(np.float).reshape(-1)
            stftMat = librosa.stft(
                y = y[:16000*SEC],
                n_fft = N_FFT
            )
            spectrogram = np.abs(stftMat)
            phase = np.angle(stftMat)
            np.save(f"{SAVE_DIR}/spectrogram/{path_list[i][channel]}.npy", spectrogram)
            np.save(f"{SAVE_DIR}/phase/{path_list[i][channel]}.npy", phase)

if __name__ == '__main__':

    print('#' * 50)
    print(f'create {SAVE_DIR}')
    print('#' * 50)

    print('\n >>> loading data ...')
    train_data = PATH_DATA['normal_train']
    valid_normal = PATH_DATA['normal_valid']
    test_normal = PATH_DATA['normal_test']
    valid_anomaly = PATH_DATA['anomaly_valid']
    test_anomaly = PATH_DATA['anomaly_test']

    print(' >>> creating spectrogram...')
    print(' >>>>>> createing train_data')
    create_stftMat_from_path(train_data)

    print(' >>>>>> createing valid_normal')
    create_stftMat_from_path(valid_normal)

    print(' >>>>>> createing test_normal')
    create_stftMat_from_path(test_normal)

    print(' >>>>>> createing valid_anomaly')
    create_stftMat_from_path(valid_anomaly)
    
    print(' >>>>>> createing test_anomaly')
    create_stftMat_from_path(test_anomaly)

    print(' >>> finished !!!')
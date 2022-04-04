"""
TAMAGOで録音した生データを10秒ごとに分割するためのプログラム
分割された音響データはチャンネルごとに連番になっている
"""

import os
import glob
import re
import warnings

from sympy import O
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wavio

### データの保存先
ROW_DATA_DIR = './../data/rawData'
SAVE_DATA_DIR = './../data'
if(os.path.exists(f"{SAVE_DATA_DIR}/split_data") == False):
    os.mkdir(f"{SAVE_DATA_DIR}")

def wavToNpy(path, cnt, anomaly=False):
    output_blk = [None] * 8
    for channel in range(8):
        output = []
        wav = wavio.read(f"{ROW_DATA_DIR}/{path[channel][18:]}")
        rep = int(wav.data.shape[0]/wav.rate/10)
        for j in range(rep):
            new_file_name = '{:05}_{:02}'.format(cnt+j, channel)
            new_wave_data = wav.data[j*wav.rate:(j+10)*wav.rate, :]
            np.save(f"{SAVE_DATA_DIR}/split_data/{new_file_name}.wav", new_wave_data)
            output.append(new_file_name)
        output_blk[channel] = output
    return output_blk, len(output_blk[0])

if __name__ == "__main__":

    ### load data
    df = pd.read_csv('../data/rawData_df.csv')
    df = df.sort_values('path')

    """
    正常データと異常データのパスを
    二次元配列 [データ数 * チャンネル数] にする
    """
    normal_paths = []    # list of ndarray[normal_path * 8]
    anomaly_paths = []   # list of ndarray[anomaly_path * 8]
    for i in range(8):
        normal_paths.append(df[(df['test'].isnull()) & (df['channel']==i)].sort_values('path')['path'].values)
        anomaly_paths.append(df[(df['test'].notna()) & (df['channel']==i)].sort_values('path')['path'].values)
    normal_paths = np.transpose(np.array(normal_paths))
    anomaly_paths = np.transpose(np.array(anomaly_paths))
    print(f'num normal files  : {len(normal_paths)}')
    print(f'num anomaly files : {len(anomaly_paths)}')

    """
    音響データごとに10秒ずつに分割する
    分割したデータは適宜保存し，ファイル名は連番にする
    """
    cnt = 0
    normal_length = []
    normal_blks = []
    print("spliting normal data ...")
    for i in tqdm(range(len(normal_paths))):
        output_blk, rep = wavToNpy(normal_paths[i], cnt)
        normal_blks.append(np.transpose(np.array(output_blk)))
        normal_length.append(rep)
        cnt += rep
    output_normal_paths = np.concatenate(normal_blks)
    anomaly_length = []
    anomaly_blks = []
    print("spliting anomaly data ...")
    for i in tqdm(range(len(anomaly_paths))):
        output_blk, rep = wavToNpy(anomaly_paths[i], cnt, anomaly=True)
        anomaly_blks.append(np.transpose(np.array(output_blk)))
        anomaly_length.append(rep)
        cnt += rep
    output_anomaly_paths = np.concatenate(anomaly_blks)
    print("split acoustic data finished\n")

    """
    パスの配列を保存する
    """
    print(f"num of normal data is\t{output_normal_paths.shape[0]}")
    print(f"num of anomaly data is\t{output_anomaly_paths.shape[0]}")
    np.save(f"{SAVE_DATA_DIR}/normal_path.npy", output_normal_paths)
    np.save(f"{SAVE_DATA_DIR}/anomaly_path.npy", output_anomaly_paths)

    """
    train_valid_testにランダムに分割する
    """
    normal_train, normal_test =  train_test_split(output_normal_paths, test_size=0.1, shuffle=True)
    normal_test, normal_valid =  train_test_split(normal_test, test_size=0.33, shuffle=True)
    anomaly_test, anomaly_valid = train_test_split(output_anomaly_paths, test_size=0.33, shuffle=True)
    print(f"shape of noramal_train is\t{normal_train.shape}")
    print(f"shape of normal_valid is\t{normal_valid.shape}")
    print(f"shape of normal_test is\t{normal_test.shape}")
    print(f"shape of anomaly_valid is\t{anomaly_valid.shape}")
    print(f"shape of anomaly_test is\t{anomaly_test.shape}")
    np.savez(
        f"{SAVE_DATA_DIR}/train_test_paths.npz",
        normal_train = normal_train,
        normal_valid = normal_valid,
        normal_test = normal_test,
        anomaly_valid = anomaly_valid,
        anomaly_test = anomaly_test
    )
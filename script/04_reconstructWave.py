#!/opt/conda/bin/python
import argparse
import os
import sys

import numpy as np
from torch.functional import norm
from tqdm import tqdm
import librosa
import wavio

### load path list
PATH_LIST_NORMAL = np.load("./../data/train_test_paths.npz")["normal_test"]
PATH_LIST_ANOMALY = np.load("./../data/train_test_paths.npz")["anomaly_test"]
PATH_LIST = np.concatenate([PATH_LIST_NORMAL, PATH_LIST_ANOMALY], 0)

RESULT_DIR = "./../result"

def get_phase(data, n_fft):
    stftMat = librosa.stft(
        y = data,
        n_fft = n_fft
    )
    phase = np.angle(stftMat)
    return phase

def reconstruct_wave_from_path(spec_path, save_path, ts_path):
    if not os.path.exists(save_path[:-13]):
        os.mkdir(save_path[:-13])
    spec = np.load(spec_path)
    timesereies = np.load(ts_path)
    phase = get_phase(timesereies, 512)
    stft_mat = (2 ** spec) * (np.cos(phase) + 1j * np.sin(phase))
    reconstructed_wav = librosa.istft(stft_mat)
    wavio.write(
        file = save_path,
        data = reconstructed_wav,
        rate = 16000,
        sampwidth = 2
    )


def main():
    print(f"result directory is {RESULT_DIR}")
    ### run main loop
    for path in tqdm(PATH_LIST):
        for channel in range(8):
            path_spec_input = f"{RESULT_DIR}/ch0{channel}/inputSpectrogram/{path[channel]}.npy"
            path_spec_output = f"{RESULT_DIR}/ch0{channel}/outputSpectrogram/{path[channel]}.npy"
            path_spec_diff = f"{RESULT_DIR}/ch0{channel}/diffSpectrogram/{path[channel]}.npy"

            path_wave_input = f"{RESULT_DIR}/ch0{channel}/inputWave/{path[channel]}.npy"
            path_wave_output = f"{RESULT_DIR}/ch0{channel}/outputWave/{path[channel]}.npy"
            path_wave_diff = f"{RESULT_DIR}/ch0{channel}/diffWave/{path[channel]}.npy"

            path_timeseries = f"{RESULT_DIR}/ch0{channel}/diffTimeSeries/{path[channel]}.npy"
            reconstruct_wave_from_path(path_spec_input, path_wave_input, path_timeseries)
            reconstruct_wave_from_path(path_spec_output, path_wave_output, path_timeseries)
            reconstruct_wave_from_path(path_spec_diff, path_wave_diff, path_timeseries)

if __name__ == "__main__":
    main()
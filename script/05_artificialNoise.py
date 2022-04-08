import os
import wave

import librosa
import numpy as np
import torch
from torchvision import transforms
import wavio

from models import AutoEncoder as AE
from trainer import Trainer
from utils import seed_everything

### meta parameters
SEED = 42
seed_everything(seed=SEED)
GPU = "cuda:0"
DEVICE = torch.device(GPU if torch.cuda.is_available() else "cpu")

RESULT_DIR = "./../result"
DATA_DIR = "./../data"
path_data = np.load(f"{DATA_DIR}/train_test_paths.npz")
PATH = path_data["normal_test"][0]

def get_spec(data, n_fft):
    stftMat = librosa.stft(
        y = data,
        n_fft = n_fft
    )
    spectrogram = np.abs(stftMat)
    spectrogram = np.log2(spectrogram)
    phase = np.angle(stftMat)
    return spectrogram, phase

def reconstruct_wave(spec, phase):
    diffStftMat = (2 ** spec) * (np.cos(phase) + 1j * np.sin(phase))
    reconstructed_wav = librosa.istft(diffStftMat)
    return reconstructed_wav

def predict(ANGLE, Hz):
    
    wave_data = []
    for channel in range(8):
        ### load data & add noise & get spec
        wav = np.load(f"{DATA_DIR}/splitData/{PATH[channel]}.npy").astype(np.float)
        noise_data = wavio.read(f"{DATA_DIR}/noise{Hz}/{ANGLE}_{channel}.wav")
        noise = noise_data.data# * 10000
        input_wav = (wav[:80000] + noise[:80000]).reshape(-1)
        spec, phase = get_spec(input_wav, 512)

        ### define model & load parmeter
        model = AE(spec.shape[0], spec.shape[1])
        model_path = f"{RESULT_DIR}/ch0{channel}/best_model_parameter.pth"
        model.load_state_dict(torch.load(model_path))

        ### predict
        model.to(DEVICE).eval()
        input_spec = torch.FloatTensor(spec).reshape(-1).to(DEVICE)
        output = model(input_spec)
        output = output.to("cpu").detach().numpy().copy().reshape(spec.shape)

        ### save
        SAVE_DIR = f"{RESULT_DIR}/noisedData{Hz}/{ANGLE}_degree"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        diff = spec - output
        diff_wav = reconstruct_wave(diff, phase)
        output_wav = reconstruct_wave(output, phase)
        np.savez(f"{SAVE_DIR}/{ANGLE}_{channel}.npz",
            input_spec = spec,
            output_spec = output,
            diff_spec = diff,
            reconstructed_wav = diff_wav
        )
        wavio.write(f"{SAVE_DIR}/{ANGLE}_{channel}_in.wav", input_wav, 16000, sampwidth=2)
        wavio.write(f"{SAVE_DIR}/{ANGLE}_{channel}_out.wav", output_wav, 16000, sampwidth=2)
        wavio.write(f"{SAVE_DIR}/{ANGLE}_{channel}_diff.wav", diff_wav, 16000, sampwidth=2)
        wave_data.append(diff_wav)
        # break
    wave_data = np.transpose(np.array(wave_data))
    wavio.write(f"{SAVE_DIR}/{ANGLE}.wav", wave_data, 16000, sampwidth=2)

if __name__ == "__main__":
    angles = [45, 135, 225, 315]
    for angle in angles:
        print(angle)
        predict(angle, 6000)
        # break
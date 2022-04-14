import argparse
import os
import codecs
import copy
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

from utils import seed_everything
from models import AutoEncoder as AE
from trainer import Trainer

def trainAllChannel(seed, device, result_dir):
    os.mkdir(result_dir)
    default_stdout = sys.stdout

    ### logging test score
    scores = []

    ### define transform
    myTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    ### trainer loop
    for channel in range(8):
        seed_everything(seed=seed)
        save_dir = f'{result_dir}/ch0{channel}'
        print(f'model training ... channel: {channel}/ 8')
        trainer = Trainer(
            model = AE,
            save_dir = save_dir,
            device = device,
            channel = channel,
            transform = myTransform,
            sec = 5,
            n_fft = 512
        )
        auc_score = trainer.all(num_epochs=20)
        scores.append(auc_score)
        del trainer
        sys.stdout = default_stdout

    ### save result
    df = pd.DataFrame()
    df['channel'] = range(8)
    df['score'] = scores
    df.to_csv(f"{RESULT_DIR}/testAUCScore.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-d", "--device", default=0, type=int)
    args = parser.parse_args()

    SEED = args.seed
    GPU = f"cuda:{args.device}"
    device = torch.device(GPU if torch.cuda.is_available() else "cpu")
    RESULT_DIR = f"./../result_seed_{SEED}/"

    print(SEED, device, RESULT_DIR)
    trainAllChannel(seed=SEED, device=device, result_dir=RESULT_DIR)
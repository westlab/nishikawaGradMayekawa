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

SEED = 42
GPU = "cuda"
device = torch.device(GPU if torch.cuda.is_available() else "cpu")
RESULT_DIR = f"./../result/"

def trainAllChannel():
    default_stdout = sys.stdout

    ### logging test score
    scores = []

    ### define transform
    myTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    ### trainer loop
    for channel in range(8):
        seed_everything(seed=SEED)
        save_dir = f'{RESULT_DIR}/ch0{channel}'
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
    trainAllChannel()
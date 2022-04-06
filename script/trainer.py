#!/opt/conda/bin/python
import os
import codecs
import copy
import sys
import time
import librosa

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.nn.modules import module
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

from models import AutoEncoder as AE
from utils import seed_everything, visualize_list, calucurate_AUC, save_image_magma
from utils import myDataset as Dataset

LR_DEF = 0.00001
BS_DEF = 8 # max:8
EP_DEF = 20

TRAIN_TEST_PATHS = './../data/train_test_paths.npz'

class Trainer():
    def __init__(self, model, save_dir, device, channel, transform, sec, n_fft):
        self.save_dir = save_dir
        if(os.path.exists(save_dir) == True):
            print(f'\nsave data dir:"{save_dir}" is already exists!!\n set another directory!!', file=sys.stderr)
            sys.exit(1)
        os.mkdir(save_dir)
        sys.stdout = open(f"{save_dir}/stdout.log", 'a+')
        print('\n>>> initializing trainer')
        print(f'\n>>> all results will be saved in {save_dir}')

        self.path_lists = np.load(TRAIN_TEST_PATHS)
        self.channel = channel
        self.transform = transform
        self.device = device
        self.sec = sec
        self.n_fft = n_fft

        ### define model
        print('\n>>> defining model')
        self.model = model(height=257, width=626)
        print(self.model, sep="\n", end="\n", file=codecs.open(f"{save_dir}/archteqture.txt", 'w', 'utf-8'))

        ### define result
        self.loss_list = []
        self.valid_score_list = []
        self.valid_pred_normal = []
        self.valid_pred_anomaly = []
        self.test_prediction = []
        self.test_filename = []
        self.valid_score = None
        self.test_score = None

        print('\n>>> Ready to train !!')

    def train(self, learning_rate=LR_DEF, num_epochs=EP_DEF, batch_size=BS_DEF):
        print('\n>>> start training')
        train_set = Dataset(
            path_list = self.path_lists['normal_train'],
            channel = self.channel, sec = self.sec, n_fft = self.n_fft,
            transforms = self.transform
        )
        train_loader = DataLoader(
            dataset = train_set,
            batch_size = batch_size, shuffle = True, num_workers = 2
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        self.model.to(self.device)
        self.loss_list = []
        self.valid_score = []
        print('-' * 65)
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            start_time = time.time()
            for i, (_, img) in enumerate(train_loader):
                x = img.to(self.device).reshape(len(img), 1, -1)
                xhat = self.model(x)
                loss = criterion(xhat, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if i % 100 == 0:
                #     print(f'### iteration:{i:5} ### loss:{loss.data.item()} ###')
            ### validation
            self.valid_score.append(self.valid())
            ### logging
            self.loss_list.append(loss.data.item())
            print(f'| epoch [{epoch:2}/{num_epochs}] | loss: {loss.data.item():03.4f} | score: {self.valid_score[-1]:.4f} | rap: {time.time()-start_time:3.4f} |')
            print('-' * 65)
            if (np.argmax(self.valid_score) == len(self.valid_score) - 1):
                best_model = copy.deepcopy(self.model)
        visualize_list(self.loss_list, save_path=self.save_dir+'/loss_list.png', xlabel='epoch', ylabel='train loss')
        visualize_list(self.valid_score, save_path=self.save_dir+'/valid_score.png', xlabel='epoch', ylabel='AUC score')
        torch.save(best_model.to('cpu').state_dict(), self.save_dir+'/best_model_parameter.pth')
        self.model = best_model
        print('>>>>>> training finished')
        print(f'>>>>>> validation score is {max(self.valid_score)}')

    def valid(self):
        normal_set = Dataset(
            path_list = self.path_lists['normal_valid'],
            channel = self.channel, sec = self.sec, n_fft = self.n_fft,
            transforms = self.transform
        )
        anomaly_set = Dataset(
            path_list = self.path_lists['anomaly_valid'],
            channel = self.channel, sec = self.sec, n_fft = self.n_fft,
            transforms = self.transform
        )
        normal_loader =  DataLoader(normal_set, batch_size=1, shuffle=False, num_workers=1)
        anomaly_loader =  DataLoader(anomaly_set, batch_size=1, shuffle=False, num_workers=1)
        _, self.valid_pred_normal = self.predict(normal_loader)
        _, self.valid_pred_anomaly = self.predict(anomaly_loader)
        auc_score = calucurate_AUC(self.valid_pred_normal, self.valid_pred_anomaly, save_dir=None)
        return auc_score

    def test(self):
        print('\n>>> start test')
        if not os.path.exists(self.save_dir+'diffTimeSeries'):
            os.mkdir(self.save_dir + '/inputSpectrogram')
            os.mkdir(self.save_dir + '/outputSpectrogram')
            os.mkdir(self.save_dir + '/diffSpectrogram')
            os.mkdir(self.save_dir + '/diffTimeSeries')
        ### load data
        normal_set = Dataset(
            path_list = self.path_lists['normal_test'],
            channel = self.channel, sec = self.sec, n_fft = self.n_fft,
            transforms = self.transform, isTest=True
        )
        anomaly_set = Dataset(
            path_list = self.path_lists['anomaly_test'],
            channel = self.channel, sec = self.sec, n_fft = self.n_fft,
            transforms = self.transform, isTest=True
        )
        normal_loader =  DataLoader(normal_set, batch_size=1, shuffle=False, num_workers=1)
        anomaly_loader =  DataLoader(anomaly_set, batch_size=1, shuffle=False, num_workers=1)
        ### prediction  
        normal_filename, normal_pred = self.predict(normal_loader, isTest=True)
        anomaly_filename, anomaly_pred = self.predict(anomaly_loader, isTest=True)
        self.test_filename = np.concatenate([normal_filename, anomaly_filename])
        self.test_prediction = np.concatenate([normal_pred, anomaly_pred])
        ### visualize result
        print('>>>>>> visualize Input & Output')
        # self.visualizeInputOutput(normal_set, 'normal')
        # self.visualizeInputOutput(anomaly_set, 'anomaly')
        print('>>>>>> calcurate test score')
        self.test_score = calucurate_AUC(normal_pred, anomaly_pred, save_dir=self.save_dir)
        return self.test_score

    def predict(self, dataloader, isTest = False):
        anomaly_score = []
        file_name_list = []
        criterion = nn.MSELoss()
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for dataTuple in dataloader:
                if isTest:
                    file_name, x, phase = dataTuple
                else:
                    file_name, x = dataTuple
                file_name = file_name[0]
                file_name_list.append(file_name)
                x = x.to(self.device).reshape(len(x), 1, -1)
                xhat = self.model(x)
                loss = criterion(xhat, x)
                anomaly_score.append(loss.item())
                if isTest:
                    x = x.to('cpu').detach().numpy().copy().reshape(257, 626)
                    xhat = xhat.to('cpu').detach().numpy().copy().reshape(257, 626)
                    diffSpec = x - xhat
                    phase = np.squeeze(phase.to('cpu').detach().numpy().copy())
                    diffStftMat = (2 ** diffSpec) * (np.cos(phase) + 1j * np.sin(phase))
                    reconstructedTimeSeries = librosa.istft(diffStftMat)
                    np.save(self.save_dir + '/inputSpectrogram/' + file_name, x)
                    np.save(self.save_dir + '/outputSpectrogram/' + file_name, xhat)
                    np.save(self.save_dir + '/diffSpectrogram/' + file_name, diffSpec)
                    np.save(self.save_dir + '/diffTimeSeries/' + file_name, reconstructedTimeSeries)
        return np.array(file_name_list), np.array(anomaly_score)

    def visualizeInputOutput(self, dataset, file_name):
        dataloader =  DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, x, _ = iter(dataloader).__next__()
            x = x.to(self.device).reshape(len(x), 1, -1)
            xhat = self.model(x)
        save_image_magma(x, xhat, file_name, 257, 626, save_dir=self.save_dir)

    def loadModel(self, model_path):
        print(f'\n>>> load pretrained model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))
        print('>>>>>> loading model finished')

    def saveResult(self):
        print('\n>>> Saving all result')
        np.savez(
            self.save_dir + '/result_data_all.npz',
            train_loss_list = self.loss_list,
            valid_score_list = self.valid_score,
            valid_pred_normal = self.valid_pred_normal,
            valid_pred_anomaly = self.valid_pred_anomaly,
            test_prediction = self.test_prediction,
            test_filename = self.test_filename,
            valid_AUC_score = np.max(self.valid_score),
            test_AUC_score = self.test_score
        )
        print('\n>>> all result saved !!')

    def all(self, learning_rate=LR_DEF, num_epochs=EP_DEF, batch_size=BS_DEF):
        self.train(learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size)
        self.valid_score = self.valid()
        test_score = self.test()
        print(f'>>>>>> valid score is {self.valid_score}')
        print(f'>>>>>> test score is {self.test_score}')
        self.saveResult()
        print('\n>>> all process completed !!')
        return test_score


if __name__ == "__main__":
    ### meta parameters
    SEED = 42
    GPU = "cuda"
    seed_everything(seed=SEED)

    ### trainer parameters
    SAVE_DIR = './../tmp_result'
    DEVICE = torch.device(GPU if torch.cuda.is_available() else "cpu")
    CHANNEL = 0
    myTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    ### define Trainer && train
    MyModel = Trainer(
        model = AE,
        save_dir = SAVE_DIR,
        device = DEVICE,
        channel = CHANNEL,
        transform = myTransform,
        sec = 5,
        n_fft = 512
    )
    MyModel.all(num_epochs=3)
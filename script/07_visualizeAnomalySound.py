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

import mplConf

RESULT_DIR = "./../result"

def forModelDiff():
    df = pd.read_csv(f"{RESULT_DIR}/final_prediction.csv")

    ### sample random file
    normal_filename = df[df["pred"] == 0]["filename"].values[-200] + ".npy"
    anomaly_filename = df[df["pred"] == 1]["filename"].values[-100] + ".npy"

    ### load data
    data = []
    data.append(np.load(f"{RESULT_DIR}/ch00/inputSpectrogram/{normal_filename}"))
    data.append(np.load(f"{RESULT_DIR}/ch00/inputSpectrogram/{anomaly_filename}"))
    data.append(np.load(f"{RESULT_DIR}/ch00/outputSpectrogram/{normal_filename}"))
    data.append(np.load(f"{RESULT_DIR}/ch00/outputSpectrogram/{anomaly_filename}"))
    data.append(np.load(f"{RESULT_DIR}/ch00/diffSpectrogram/{normal_filename}"))
    data.append(np.load(f"{RESULT_DIR}/ch00/diffSpectrogram/{anomaly_filename}"))

    ### inverse preprocessing
    for i in range(6):
        data[i] = data[i] * np.log10(2)

    ### visualize input & output & difference of AE
    xticks = [0, 125, 250, 375, 500, 625]
    xlabels = [0, 1, 2, 3, 4, 5]
    yticks = [0, 64, 128, 192, 256]
    ylabels = [0, 2, 4, 6, 8]
    titles = [
        "input (normal)",
        "input (anomaly)",
        "output (normal)",
        "output (anomaly)",
        "difference (normal)",
        "difference (anomaly)",
    ]

    rcParams["figure.figsize"] = [6, 4.5]
    rcParams["font.size"] = 8
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams["axes.labelsize"] = 7
    rcParams["figure.subplot.left"] = 0.07
    rcParams["figure.subplot.bottom"] = 0.05
    rcParams["figure.subplot.right"] =0.95
    rcParams["figure.subplot.top"] = 0.97
    fig, axes = plt.subplots(3, 2)
    ### ax before imshow ###
    for ax, i in zip(axes.flat, range(6)):
        axpos = ax.get_position()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if i < 2:
            img = ax.imshow(data[i], origin="lower", cmap="magma", vmin=-1, vmax=6)
        elif i < 4:
            img = ax.imshow(data[i], origin="lower", cmap="magma", vmin=2, vmax=6)
        else:
            img = ax.imshow(data[i], origin="lower", cmap="magma", vmin=-4, vmax=2)
        cbar = fig.colorbar(img, cax=cax)

        ax.set_title(titles[i])
        ax.set_xlabel("time [sec]", weight = "light")
        ax.set_ylabel("frequency [kHz]", weight = "light")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

    plt.savefig(f"{RESULT_DIR}/diff_AE.png", transparent=True)

def forNoiseDiff():
    data = np.load(f"{RESULT_DIR}/noisedData{6000}/45_degree/45_0.npz")
    titles = [
        "input of anomaly data",
        "output of anomaly data",
        "difference of anomaly data",
    ]

    xticks = [0, 125, 250, 375, 500, 625]
    xlabels = [0, 1, 2, 3, 4, 5]
    yticks = [0, 64, 128, 192, 256]
    ylabels = [0, 2, 4, 6, 8]

    rcParams["figure.figsize"] = [4, 6]
    rcParams["font.size"] = 8
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams["axes.labelsize"] = 7
    rcParams["figure.subplot.left"] = 0.07
    rcParams["figure.subplot.bottom"] = 0.05
    rcParams["figure.subplot.right"] =0.95
    rcParams["figure.subplot.top"] = 0.97
    fig, axes = plt.subplots(3, 1)

    ### ax before imshow ###
    for ax, i in zip(axes.flat, range(3)):
        axpos = ax.get_position()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if i < 1:
            img = ax.imshow(data["input_spec"] * np.log10(2), origin="lower", cmap="magma", vmin=-1, vmax=6)
        elif i < 2:
            img = ax.imshow(data["output_spec"] * np.log10(2), origin="lower", cmap="magma", vmin=2, vmax=6)
        else:
            img = ax.imshow(data["diff_spec"] * np.log10(2), origin="lower", cmap="magma", vmin=-4, vmax=2)
        cbar = fig.colorbar(img, cax=cax)

        ax.set_title(titles[i])
        ax.set_xlabel("time[sec]", weight = "light")
        ax.set_ylabel("frequency[kHz]", weight = "light")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

    plt.savefig(f"{RESULT_DIR}/diff_AE_artNoise.png", transparent=True)

if __name__ == "__main__":
    forModelDiff()
    # forNoiseDiff()
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
import numpy as np

import mplConf

if __name__ == "__main__":
    angles = np.load("./../result/sourcePosition.npy")

    rcParams["figure.figsize"] = [4, 3]
    rcParams["figure.subplot.left"] = 0.12
    rcParams["figure.subplot.bottom"] = 0.12
    rcParams["figure.subplot.right"] =0.99
    rcParams["figure.subplot.top"] = 0.99

    fig , axes = plt.subplots(1, 1)
    axes.hist(angles, bins=50, alpha=0.9)
    axes.set_xlim(8, 18)
    axes.set_xlabel("angle [°]", weight = "light")
    axes.set_ylabel("count", weight = "light")
    axes.set_xlim(-180, 180)

    plt.savefig('./../result/anomaly_direction.png')
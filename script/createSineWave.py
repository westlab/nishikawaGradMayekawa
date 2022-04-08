import numpy as np
import matplotlib.pyplot as plt
import wavio

FREQ = 440
SEC = 200
RATE = 44100

t = np.arange(0, RATE * SEC)
wav = np.sin(2 * np.pi * FREQ * t/RATE)

wavio.write(f"./../data/sine{FREQ}.wav", wav, RATE, sampwidth=2)
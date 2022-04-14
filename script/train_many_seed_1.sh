#!/bin/bash

python 02_trainAllChannel.py -s5 -d1
python 02_trainAllChannel.py -s6 -d1
python 02_trainAllChannel.py -s7 -d1
python 02_trainAllChannel.py -s8 -d1
python 02_trainAllChannel.py -s9 -d1

python 03_ensambleAllChannel.py -r./../result_seed_5
python 03_ensambleAllChannel.py -r./../result_seed_6
python 03_ensambleAllChannel.py -r./../result_seed_7
python 03_ensambleAllChannel.py -r./../result_seed_8
python 03_ensambleAllChannel.py -r./../result_seed_9
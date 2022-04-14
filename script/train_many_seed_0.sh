#!/bin/bash

python 02_trainAllChannel.py -s0 -d0
python 02_trainAllChannel.py -s1 -d0
python 02_trainAllChannel.py -s2 -d0
python 02_trainAllChannel.py -s3 -d0
python 02_trainAllChannel.py -s4 -d0

python 03_ensambleAllChannel.py -r./../result_seed_0
python 03_ensambleAllChannel.py -r./../result_seed_1
python 03_ensambleAllChannel.py -r./../result_seed_2
python 03_ensambleAllChannel.py -r./../result_seed_3
python 03_ensambleAllChannel.py -r./../result_seed_4
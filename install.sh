#!/bin/bash

git clone https://github.com/m-toman/Tacotron-2 tacotron2
#cp config/hparams.py tacotron2/hparams.py
touch tacotron2/__init__.py

git clone https://github.com/m-toman/WaveRNN-Pytorch.git wavernn_alt
touch wavernn_alt/__init__.py

#!/bin/bash

git clone https://github.com/m-toman/Tacotron-2 tacotron2
#cp config/hparams.py tacotron2/hparams.py
touch tacotron2/__init__.py

git clone https://github.com/m-toman/WaveRNN-Pytorch.git wavernn_alt
touch wavernn_alt/__init__.py

#mkdir -p model_checkpoints
#cd model_checkpoints
#wget https://www.dropbox.com/s/86fiurfwbv4b2hp/tacorn_wavernn.pyt
#wget https://www.dropbox.com/s/y49ppvkkto79hct/tacorn_wavernn_step.npy
#cd ..

#!/bin/bash

git clone https://github.com/m-toman/Tacotron-2 tacotron2
#cp config/hparams.py tacotron2/hparams.py
touch tacotron2/__init__.py

# get pretrained model
# this is temporary and shall be moved to the python scripts
cd tacotron2
wget https://www.dropbox.com/s/5svv16eolba0i7o/logs-Tacotron-2.zip
unzip logs-Tacotron-2.zip
rm logs-Tacotron-2.zip
cd ..

mkdir -p model_checkpoints
cd model_checkpoints
wget https://www.dropbox.com/s/86fiurfwbv4b2hp/tacorn_wavernn.pyt
wget https://www.dropbox.com/s/y49ppvkkto79hct/tacorn_wavernn_step.npy
cd ..
